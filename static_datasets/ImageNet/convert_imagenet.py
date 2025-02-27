import argparse
import os
import random

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import csv
from tqdm import tqdm
import math
import torch.distributed as dist

from ...general_utils.calibration import bias_corr_model, weights_cali_model
from ...general_utils.spiking_layer import SpikeModule
from ...general_utils.fold_bn import search_fold_and_remove_bn
from models.vit import *
from models.resnet import res_spcials as res_spcials_res34, res_spcials_res50 #...
from models.resnet import spiking_resnet18, res_spcials, SpikeResModule, spiking_resnet101, spiking_resnet50, spiking_resnet34
from models.vgg import vgg16, vgg16_bn, vgg_specials
from ...general_utils.spiking_layer import SpikeModule, SpikeModel, get_maximum_activation, sensitivity_anylysis, sensitivity_anylysis_threshold, Energy
from tqdm import tqdm
import time
import os
import pandas as pd
import time as timep
from scipy.stats import spearmanr, kendalltau
from matplotlib import pyplot as plt
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def build_imagenet_data(data_path: str = '', input_size: int = 224, batch_size: int = 64, workers: int = 4,
                        dist_sample: bool = False):
    # print('==> Using Pytorch Dataset')

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # torchvision.set_image_backend('accimage')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]))

    if dist_sample:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)
    return train_loader, val_loader


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def throughput_test(model, test_loader, device, dynamic=True,
                    threshold=0.2, metric='entropy', timesteps=4, measure_energy=False):
    correct = 0
    total = 0
    model.eval()
    overall_time = 0
    energy_calculator = Energy(model)
    energy_calculator.register_hooks()
    model.reccurent = True
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        s_time = timep.time()
        model.init_membrane_potential()
        if not dynamic:
            logits = 0
            for ts in range(timesteps):
                logits += model(inputs)
            _, predicted = logits.cpu().max(1)
            total += 1
            correct += float(predicted.eq(targets).sum().item())
        else:
            partial_logits = 0
            # evaluate sample-by-sample
            for s_i in range(timesteps):
                t = s_i + 1
                partial_logits += model(inputs)
                if metric == 'entropy':
                    probs = torch.log_softmax(partial_logits, dim=1)
                    entropy = - torch.sum(probs * torch.exp(probs), dim=1) / math.log(partial_logits.shape[1])
                    # Reverse the entropy for consistent thresholding implementation
                    entropy = 1 - entropy
                elif metric == 'confidence':
                    probs = torch.softmax(partial_logits, dim=1)
                    confidence, _ = torch.sort(probs, dim=1, descending=True)
                    confidence = confidence[:, 0]
                    entropy = confidence
                else:
                    raise NotImplementedError

                if entropy > threshold[s_i] or t == timesteps:
                    _, predicted = partial_logits.cpu().max(1)
                    total += 1
                    correct += float(predicted.eq(targets).sum().item())
                    break
        e_time = timep.time()
        overall_time += (e_time - s_time)
    final_acc = (100 * correct / total)

    print("Throughput: {}, accuracy: {}, Energy: {}".format(total/overall_time, final_acc, energy_calculator.energy_meter.avg))


@torch.no_grad()
def test(model, test_loader, device, dynamic=True, threshold=0.2, stable_steps_threshold=12, T=4, metric='entropy', save_image=False):
    correct = 0
    total = 0
    time = 0
    time_vec = np.array([0 for t in range(T)])
    model.eval()
    spikecount = 0

    num_saved_hard_images = 0
    num_saved_easy_images = 0

    # Creating lists to store examples of easy and hard images and their corresponding labels
    score_list = []

    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device).float()
        outputs = model(inputs)
        for m in ann.modules():
                if isinstance(m, SpikeModule) or isinstance(m, SpikeResModule):
                    spikecount += m.spike_counter
        prev_predictions = torch.zeros(targets.size(0)).to(device) -1 
        stable_steps = torch.zeros((T, targets.size(0))).to(device)

        if not dynamic:
            mean_out = outputs.mean(1)
            _, predicted = mean_out.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
        else:
            metric_list = []
            batch_dim = targets.size(0)
            total += float(batch_dim)
            # calculate entropy
            for s_i in range(T):
                t = s_i + 1
                partial_logits = outputs[:, :t].sum(1) if s_i > 0 else outputs[:, 0]
                if metric == 'entropy':
                    probs = torch.log_softmax(partial_logits, dim=1)
                    entropy = - torch.sum(probs * torch.exp(probs), dim=1) / math.log(partial_logits.shape[1])
                    metric_list += [1 - entropy]
                elif metric == 'confidence':
                    probs = torch.softmax(partial_logits, dim=1)
                    confidence, _ = torch.sort(probs, dim=1, descending=True)
                    confidence = confidence[:, 0]
                    metric_list += [confidence]
                elif metric == 'combined':
                    probs_entropy = torch.log_softmax(partial_logits, dim=1)
                    entropy = - torch.sum(probs_entropy * torch.exp(probs_entropy), dim=1) / math.log(partial_logits.shape[1])
                    metric_list += [1 - entropy]

                    probs = torch.softmax(partial_logits, dim=1)
                    _, predicted = probs.max(1)
                    prediction_changes = predicted != prev_predictions
                    stable_steps[s_i, prediction_changes] = 0
                    if s_i > 0:
                        stable_steps[s_i, ~prediction_changes] = stable_steps[s_i-1, ~prediction_changes] + 1
                    prev_predictions = predicted
                else:
                    raise NotImplementedError
            score_list += [torch.stack(metric_list, dim=0)]
            # compute accuracy sample-by-sample
            final_logits = []
            for b_i in range(batch_dim):
                for s_i in range(T):
                    if metric_list[s_i][b_i] > threshold_list[s_i] or s_i == T - 1:
                        t = s_i + 1             # current time step
                        time_vec[s_i] = time_vec[s_i] + 1
                        final_logits += [outputs[b_i, :t].sum(0) if s_i > 0 else outputs[b_i, 0]]
                        break
            final_logits = torch.stack(final_logits, dim=0)
            _, predicted = final_logits.cpu().max(1)
            correct += float(predicted.eq(targets).sum().item())

            if num_saved_easy_images >= 25 and num_saved_hard_images >= 25:
                break

    if save_image:
        score_list = torch.cat(score_list, dim=1)
        print(score_list.shape)
        np.save("raw/conf_dist.npy", score_list.cpu().numpy())

    final_acc = (100 * correct / total)
    if dynamic:
        time = np.dot(time_vec, np.array(range(1, T + 1)))
        avg_time = (time / total)
        time_ratio = time_vec / total
        return final_acc, avg_time, time_ratio, spikecount
    else:
        return final_acc


@torch.no_grad()
def validate_model(test_loader, ann):
    correct = 0
    total = 0
    spikecount = 0
    ann.eval()
    device = next(ann.parameters()).device
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        outputs = ann(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
        for m in ann.modules():
                if isinstance(m, SpikeModule) or isinstance(m, SpikeResModule):
                    spikecount += m.spike_counter

    return 100 * correct / total, spikecount


@torch.no_grad()
def calculate_thresholds(model, train_loader, T, device, metric='entropy', num_batches=10):
    model.eval()
    thresholds = []

    with torch.no_grad():
        correct_scores_accumulator = [[] for _ in range(T)]  # Assuming model has a 'time' attribute
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total += targets.size(0)

            for t in range(T):
                # Calculate partial sum of logits up to time step t
                partial_logits = outputs[:, :t+1].sum(dim=1)
                # Compute probabilities
                probs = torch.softmax(partial_logits, dim=1)
                # Determine predictions and check correctness
                _, predicted = torch.max(probs, dim=1)
                correct = predicted.eq(targets)

                if metric == 'entropy':
                    probs = torch.log_softmax(partial_logits, dim=1)
                    entropy = -torch.sum(probs * torch.exp(probs), dim=1) / math.log(partial_logits.shape[1])
                    for i, is_correct in enumerate(correct):
                        if is_correct:
                            correct_scores_accumulator[t].append(entropy[i].item())
                elif metric == 'confidence':
                    probs = torch.softmax(partial_logits, dim=1)
                    confidence, _ = torch.sort(probs, dim=1, descending=True)
                    confidence = confidence[:, 0]
                    for i, is_correct in enumerate(correct):
                        if is_correct:
                            correct_scores_accumulator[t].append(confidence[i].item())
                else:
                    raise NotImplementedError("Metric not supported")

            if num_batches is not None and batch_idx + 1 >= num_batches:
                break

        # Calculate dynamic thresholds based on correct predictions
        entropy_thresholds = []
        for t in range(T):
            if correct_scores_accumulator[t]:
                entropy_thresholds.append(np.mean(correct_scores_accumulator[t]))
            else:
                entropy_thresholds.append(1)
        min_entropy = min(entropy_thresholds)
        for t in range(T):
            if correct_scores_accumulator[t]:
                thresholds.append(0.995 + 0.005*np.exp(min_entropy- entropy_thresholds[t]))
            else:
                thresholds.append(1)  # Default threshold if no correct predictions

    return thresholds

def init_distributed():
    """
    Initialize distributed training environment
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        # set device for each process
        torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--arch', default='res34', type=str, help='network architecture')
    parser.add_argument('--dpath', default='/home/dataset/imagenet', type=str, help='dataset directory')
    parser.add_argument('--seed', default=1000, type=int, help='random seed to reproduce results')
    parser.add_argument('--batch_size', default=256, type=int, help='minibatch size')

    parser.add_argument('--calib', default='light', type=str, help='calibration methods',
                        choices=['none', 'light', 'advanced'])
    parser.add_argument('--T', default=32, type=int, help='snn simulation length')
    parser.add_argument('--maxspike', default=8, type=int, help='max fire times')
    parser.add_argument('--device', default='', type=str, help='device select')
    parser.add_argument('--minspike', default=2, type=int, help='max fire times')
    parser.add_argument('--initialspike', default=16, type=int, help='max fire times')
    parser.add_argument('--local-rank', default=-1, type=int, help='snn simulation length')
    parser.add_argument('--usebn', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--search', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--search_threshold', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--desired_maxspike', default=1, type=int, help='max fire times')
    parser.add_argument('--searchtime', default=1, type=int, help='max fire times')
    parser.add_argument('--threshold_ratio', default=0.95, type=float, help='max fire times')
    parser.add_argument('--maxspike_ratio', default=0.95, type=float, help='max fire times')
    parser.add_argument('--method', default='dp', type=str, help='network architecture')
    parser.add_argument('--metric', default='kl', type=str, help='network architecture')
    parser.add_argument('--t', default=1., type=float, metavar='N',
                    help='threshold for entropy or confidence.')
    parser.add_argument('--T_metric', default='entropy', type=str, help='network architecture')
    parser.add_argument('--stable_step', default=4., type=float, metavar='N',
                    help='threshold for entropy or confidence.')

    # Initialize distributed envrionments
    # Note: If this doesn't work, you may use the method in official torch example:
    # https://pytorch.org/tutorials/intermediate/dist_tuto.html
    try:
        initialized, local_rank = init_distributed()
        if initialized:
            torch.cuda.set_device(local_rank)
    except:
        print('For some reason, your distributed environment is not initialized, this program may run on separate GPUs')
        initialized = False

    args = parser.parse_args()
    # print(args)
    results_list = []
    acc = 0
    spikecount = 0
    use_bn = args.usebn

    device = args.device
    if args.device == '':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run one time imagenet experiment.
    for i in range(1):

        seed_all(seed=args.seed + i)
        sim_length = args.T

        train_loader, test_loader = build_imagenet_data(data_path=args.dpath, dist_sample=initialized, batch_size=args.batch_size)

        if args.arch == 'VGG16':
            ann = vgg16_bn(pretrained=True) if args.usebn else vgg16(pretrained=True)
        elif args.arch == 'res34':
            ann = spiking_resnet34(pretrained=True, use_bn=args.usebn)
        elif args.arch == 'res18':
            ann = spiking_resnet18(pretrained=True)
        elif args.arch == 'res50':
            ann = spiking_resnet50(pretrained=True)
        elif args.arch == 'res101':
            ann = spiking_resnet101(pretrained=True)
        elif args.arch == 'vit':
            ann = vit()
        else:
            raise NotImplementedError

        search_fold_and_remove_bn(ann)
        ann.cuda()

        # print(sum(p.numel() for p in ann.parameters() if p.requires_grad))
        if args.search:
            args.desired_maxspike = args.maxspike
            args.maxspike = args.initialspike
        if args.arch.startswith('res'):
            if args.arch=='res34':
                snn = SpikeModel(model=ann, sim_length=sim_length,
                            specials=res_spcials_res34, maxspike=args.maxspike)
            elif args.arch=='res50':
                snn = SpikeModel(model=ann, sim_length=sim_length,
                            specials=res_spcials_res50, maxspike=args.maxspike)
            # elif TO-DO
        else:
            snn = SpikeModel(model=ann, sim_length=sim_length,
                        specials=vgg_specials if args.arch == 'VGG16' else res_spcials, maxspike=args.maxspike)
        snn.cuda()
        mse = True
        start = time.time()
        get_maximum_activation(train_loader, model=snn, momentum=0.9, iters=5, mse=mse, percentile=None, maxspike=args.maxspike,
                            sim_length=sim_length, channel_wise=True, dist_avg=initialized)
        end = time.time()

        if args.search:
            for i in range(args.searchtime):
                start = time.time()
                optimal_maxspike_list, node_list = sensitivity_anylysis(train_loader, model=snn, maxspike=args.maxspike, maxspike_ratio=args.maxspike_ratio, sim_length=sim_length, dist_avg=initialized, disred_maxspike=args.desired_maxspike, minspike=args.minspike, method=args.method, metric=args.metric)
                index = 0
                for m in snn.modules():
                    if isinstance(m, SpikeModule):
                        m.maxspike = optimal_maxspike_list[index]
                        # m.maxspike = 2
                        index += 1
                get_maximum_activation(train_loader, model=snn, momentum=0.9, iters=5, mse=mse, percentile=None, maxspike=args.maxspike,
                                            sim_length=sim_length, channel_wise=True, dist_avg=initialized)
            if args.calib == 'light':
                bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False, dist_avg=initialized)
            if args.calib == 'advanced':
                weights_cali_model(model=snn, train_loader=train_loader, num_cali_samples=1024, learning_rate=1e-5,
                                dist_avg=initialized)
                bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False, dist_avg=initialized)
            if args.search_threshold:
                optimal_maxspike_list, node_list = sensitivity_anylysis_threshold(train_loader, model=snn, maxspike=args.maxspike, threshold_ratio=args.threshold_ratio, sim_length=sim_length, method=args.method, metric=args.metric)
                # print(f"ratio per layer: {optimal_maxspike_list}")
            index = 0
            for m in snn.modules():
                if isinstance(m, SpikeModule) or isinstance(m, SpikeResModule):
                    # m.count = True
                    m.spike_counter = 0
                    if args.search_threshold:
                        m.threshold = m.threshold * optimal_maxspike_list[index]
                    index += 1
            snn.set_spike_state(use_spike=True)
            bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False)
            snn.set_spike_state(use_spike=True)

            threshold_list = [args.t] * sim_length
            print(threshold_list)
            print(np.mean(threshold_list))
            facc, time, time_ratio, spikecount = test(snn, test_loader, device, dynamic=True, threshold=args.t, stable_steps_threshold=args.stable_step,T=sim_length, metric=args.T_metric,
                                        save_image=False)
            print('Threshold: {}, time: {}, acc: {}, portion: {}, spikecount: {}'.format(args.t, time, facc, time_ratio, spikecount))
        else:
            if args.calib == 'light':
                bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False, dist_avg=initialized)
            if args.calib == 'advanced':
                bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False, dist_avg=initialized)
            if args.search_threshold:
                optimal_maxspike_list, node_list = sensitivity_anylysis_threshold(train_loader, model=snn, maxspike=args.maxspike, threshold_ratio=args.threshold_ratio, sim_length=sim_length, method=args.method, metric=args.metric)
            index = 0
            for m in snn.modules():
                if isinstance(m, SpikeModule) or isinstance(m, SpikeResModule):
                    # m.count = True
                    m.spike_counter = 0
                    if args.search_threshold:
                        m.threshold = m.threshold * optimal_maxspike_list[index]
                    index += 1

            threshold_list = calculate_thresholds(snn, train_loader, T=sim_length, device=device, metric=args.T_metric, num_batches=100)

            print(threshold_list)
            print(np.mean(threshold_list))
            facc, time, time_ratio, spikecount = test(snn, test_loader, device, dynamic=True, threshold=args.t, stable_steps_threshold=args.stable_step,T=sim_length, metric=args.T_metric,
                                        save_image=False)
            print('Threshold: {}, time: {}, acc: {}, portion: {}, spikecount: {}'.format(args.t, time, facc, time_ratio, spikecount))


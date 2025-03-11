import argparse
import os
import random
from unittest import result

import numpy as np
import torch
import math
from train_cifar import build_data

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from general_utils.calibration import bias_corr_model
from models.resnet import res_specials

from models.resnet import SpikeResModule
from general_utils.spiking_layer import *
from models.resnet import resnet20 as resnet20_cifar
from models.resnet import resnet32 as resnet32_cifar
from models.vgg import VGG
from general_utils.fold_bn import search_fold_and_remove_bn
from tqdm import tqdm
import time
import time as timep


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
    time = 0
    model.eval()
    overall_time = 0
    energy_calculator = Energy(model)
    energy_calculator.register_hooks()
    model.reccurent = True
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
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
                    time += t
                    correct += float(predicted.eq(targets).sum().item())
                    break
        e_time = timep.time()
        overall_time += (e_time - s_time)
    final_acc = (100 * correct / total)
    avg_time = (time / total)

    print("Throughput: {}, accuracy: {}, avg_timesteps: {}, Energy: {}".format(total/overall_time, final_acc, avg_time, energy_calculator.energy_meter.avg))


@torch.no_grad()
def test(model, test_loader, device, dynamic=True, threshold=0.2, stable_steps_threshold=12, confidence_threshold=0.001, T=4, metric='entropy', save_image=False):
    correct = 0
    total = 0
    time = 0
    time_vec = np.array([0 for t in range(T)])
    model.eval()

    num_saved_hard_images = 0
    num_saved_easy_images = 0

    # Creating lists to store examples of easy and hard images and their corresponding labels
    score_list = []

    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device).float()
        outputs = model(inputs)
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
                elif 'confidence' in metric:
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
                    if metric_list[s_i][b_i] > threshold[s_i] or s_i == T - 1:
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
        return final_acc, avg_time, time_ratio
    else:
        return final_acc


@torch.no_grad()
def validate_model(test_loader, ann):
    correct = 0
    total = 0
    ann.eval()
    device = next(ann.parameters()).device
    spikecount = 0
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
    model.singleT = True

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
                thresholds.append(0.6 + 0.4*np.exp(min_entropy- entropy_thresholds[t]))

            else:
                thresholds.append(1)  # Default threshold if no correct predictions
    model.singleT = False
    return thresholds


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--arch', default='VGG16', type=str, help='network architecture', choices=['VGG16', 'res20'])
    parser.add_argument('--dpath', default='/home/dataset', type=str, help='dataset directory')
    parser.add_argument('--model', default='', type=str, help='model path')
    parser.add_argument('--seed', default=1000, type=int, help='random seed to reproduce results')
    parser.add_argument('--batch_size', default=256, type=int, help='minibatch size')
    parser.add_argument('--device', default='', type=str, help='device select')
    parser.add_argument('--calib', default='light', type=str, help='calibration methods',
                        choices=['none', 'light', 'advanced'])
    parser.add_argument('--T', default=16, type=int, help='snn simulation length')
    parser.add_argument('--usebn', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--maxspike', default=4, type=int, help='max fire times')
    parser.add_argument('--minspike', default=1, type=int, help='max fire times')
    parser.add_argument('--initialspike', default=8, type=int, help='max fire times')
    parser.add_argument('--desired_maxspike', default=1, type=int, help='max fire times')
    parser.add_argument('--search', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--search_threshold', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--searchtime', default=1, type=int, help='max fire times')
    parser.add_argument('--threshold_ratio', default=0.95, type=float, help='max fire times')
    parser.add_argument('--maxspike_ratio', default=0.95, type=float, help='max fire times')
    parser.add_argument('--method', default='pruning', type=str, help='network architecture')
    parser.add_argument('--metric', default='kl', type=str, help='network architecture')
    parser.add_argument('--t', default=1., type=float, metavar='N',
                    help='threshold for entropy or confidence.')
    parser.add_argument('--T_metric', default='entropy', type=str, help='network architecture')

    args = parser.parse_args()
    # results_list = []
    acc = 0
    spikecount = 0
    use_bn = args.usebn

    device = args.device
    if args.device == '':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_val_result = 0

    # we run the experiments for 5 times, with different random seeds
    for i in range(1):
        start = time.time()
        seed_all(seed=args.seed + i)
        sim_length = args.T

        use_cifar10 = args.dataset == 'CIFAR10'

        train_loader, test_loader = build_data(dpath=args.dpath, cutout=True, use_cifar10=use_cifar10, auto_aug=True, batch_size=args.batch_size)

        if args.arch == 'VGG16':
            ann = VGG('VGG16', use_bn=use_bn, num_class=10 if use_cifar10 else 100)
        elif args.arch == 'res20':
            ann = resnet20_cifar(use_bn=use_bn, num_classes=10 if use_cifar10 else 100)
        elif args.arch == 'res32':
            ann = resnet32_cifar(use_bn=use_bn, num_classes=10 if use_cifar10 else 100)
        else:
            raise NotImplementedError

        load_path = 'raw/' + args.dataset + '/' + args.arch + '_wBN_wd5e4_state_dict.pth' if use_bn else \
            'raw/' + args.dataset + '/' + args.arch + '_woBN_wd1e4_state_dict.pth'
        if args.model != '':
            load_path = args.model

        state_dict = torch.load(load_path, map_location=device)['model']
        ann.load_state_dict(state_dict, strict=True)
        search_fold_and_remove_bn(ann)
        ann.cuda()
        # validate_model(test_loader, ann)

        if args.search:
            args.desired_maxspike = args.maxspike
            args.maxspike = args.initialspike

        snn = SpikeModel(model=ann, sim_length=sim_length, specials=res_specials, maxspike=args.maxspike)
        snn.cuda()
        # print(snn)

        mse = False if args.calib == 'none' else True
        end = time.time()

        start = time.time()
        get_maximum_activation(train_loader, model=snn, momentum=0.9, iters=5, mse=mse, percentile=None, maxspike=args.maxspike,
                            sim_length=sim_length, channel_wise=True)
        end = time.time()

        if args.search:
            for i in range(args.searchtime):

                optimal_maxspike_list, node_list = sensitivity_anylysis(train_loader, model=snn, maxspike=args.maxspike, maxspike_ratio=args.maxspike_ratio, sim_length=sim_length, disred_maxspike=args.desired_maxspike, minspike=args.minspike, method=args.method, metric=args.metric)
                # print(f"Timesteps per layer: {optimal_maxspike_list}")
                index = 0
                for m in snn.modules():
                    if isinstance(m, SpikeModule):
                        m.maxspike = optimal_maxspike_list[index]
                        index += 1
                get_maximum_activation(train_loader, model=snn, momentum=0.9, iters=5, mse=mse, percentile=None, maxspike=args.maxspike,
                                            sim_length=sim_length, channel_wise=True)
            if args.calib == 'light':
                bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False)
            if args.search_threshold:
                optimal_maxspike_list, node_list = sensitivity_anylysis_threshold(train_loader, model=snn, maxspike=args.maxspike, threshold_ratio=args.threshold_ratio, sim_length=sim_length, method=args.method, metric=args.metric)
                # print(f"ratio per layer: {optimal_maxspike_list}")
            index = 0
            for m in snn.modules():
                if isinstance(m, SpikeModule) or isinstance(m, SpikeResModule):
                    m.count = True
                    m.spike_counter = 0
                    if args.search_threshold:
                        m.threshold = m.threshold * optimal_maxspike_list[index]
                    index += 1
            snn.set_spike_state(use_spike=True)
            bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False)
            snn.set_spike_state(use_spike=True)
            threshold_list = calculate_thresholds(snn, train_loader, T=sim_length, device=device, metric=args.T_metric, num_batches=100)

            print(threshold_list)
            print(np.mean(threshold_list))
            snn.singleT = True
            facc, time, time_ratio = test(snn, test_loader, device, dynamic=True, threshold=threshold_list, T=sim_length, metric=args.T_metric, save_image=False)
            print('Threshold: {}, time: {}, acc: {}, portion: {}'.format(args.t, time, facc, time_ratio))

        else:
            if args.calib == 'light':
                bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False)
            if args.calib == 'advanced':
                bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False)
            if args.search_threshold:
                optimal_maxspike_list, node_list = sensitivity_anylysis_threshold(train_loader, model=snn, maxspike=args.maxspike, threshold_ratio=args.threshold_ratio, sim_length=sim_length, method=args.method, metric=args.metric)
                # print(f"ratio per layer: {optimal_maxspike_list}")
            index = 0
            for m in snn.modules():
                if isinstance(m, SpikeModule) or isinstance(m, SpikeResModule):
                    m.count = True
                    m.spike_counter = 0
                    if args.search_threshold:
                        m.threshold = m.threshold * optimal_maxspike_list[index]
                    # m.threshold = m.threshold * optimal_maxspike_list[index]
                    index += 1
            snn.set_spike_state(use_spike=True)
            bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False)
            snn.set_spike_state(use_spike=True)
            threshold_list = calculate_thresholds(snn, train_loader, T=sim_length, device=device, metric=args.T_metric, num_batches=100)

            print(threshold_list)
            print(np.mean(threshold_list))
            snn.singleT = True
            facc, time, time_ratio = test(snn, test_loader, device, dynamic=True, threshold=threshold_list, T=sim_length, metric=args.T_metric, save_image=False)
            print('Threshold: {}, time: {}, acc: {}, portion: {}'.format(args.t, time, facc, time_ratio))

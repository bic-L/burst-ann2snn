import argparse
import os
import random

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import csv
from tqdm import tqdm

from distributed_utils import get_local_rank, initialize
from models.calibration import bias_corr_model
from models.ImageNet.models.resnet import resnet34_snn
from models.ImageNet.models.resnet import res_spcials as res_spcials_res34
from models.ImageNet.models.spiking_resnet import spiking_resnet18, res_spcials, SpikeResModule, SpikeModule
from models.ImageNet.models.vgg import vgg16, vgg16_bn, vgg_specials
from models.spiking_layer import SpikeModel, get_maximum_activation, sensitivity_anylysis, sensitivity_anylysis_threshold
from tqdm import tqdm
import time
import os
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Burst-ANN2SNN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--arch', default='res18', type=str, help='network architecture')
    parser.add_argument('--dpath', default='/home/dataset/imagenet', type=str, help='dataset directory')
    parser.add_argument('--seed', default=1000, type=int, help='random seed to reproduce results')
    parser.add_argument('--batch_size', default=256, type=int, help='minibatch size')

    parser.add_argument('--calib', default='none', type=str, help='calibration methods',
                        choices=['none', 'light'])
    parser.add_argument('--T', default=8, type=int, help='snn simulation length')
    parser.add_argument('--maxspike', default=4, type=int, help='max fire times')
    parser.add_argument('--minspike', default=1, type=int, help='min fire times')
    parser.add_argument('--initialspike', default=8, type=int, help='initial fire times')

    parser.add_argument('--usebn', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--search', action='store_true', help='enable BSR')
    parser.add_argument('--search_threshold', action='store_true', help='enable SSR')
    parser.add_argument('--desired_maxspike', default=1, type=int, help='target BS')
    parser.add_argument('--threshold_ratio', default=0.95, type=float, help='ratio of SSR')
    parser.add_argument('--maxspike_ratio', default=0.95, type=float, help='ratio of BSR')
    parser.add_argument('--method', default='dp', type=str, help='serach method')
    parser.add_argument('--metric', default='kl', type=str, help='search metric')

    try:
        initialize()
        initialized = True
        torch.cuda.set_device(get_local_rank())
    except:
        print('For some reason, your distributed environment is not initialized, this program may run on separate GPUs')
        initialized = False

    args = parser.parse_args()
    print(args)
    results_list = []
    acc = 0
    spikecount = 0
    use_bn = args.usebn

    # run one time imagenet experiment.
    for i in range(1):

        seed_all(seed=args.seed + i)
        sim_length = args.T

        train_loader, test_loader = build_imagenet_data(data_path=args.dpath, dist_sample=initialized, batch_size=args.batch_size)

        if args.arch == 'VGG16':
            ann = vgg16_bn(pretrained=True) if args.usebn else vgg16(pretrained=True)
        elif args.arch == 'res34':
            ann = resnet34_snn(pretrained=True, use_bn=args.usebn)
        elif args.arch == 'res18':
            ann = spiking_resnet18(pretrained=True)
        else:
            raise NotImplementedError

        search_fold_and_remove_bn(ann)
        ann.cuda()

        if args.search:
            args.desired_maxspike = args.maxspike
            args.maxspike = args.initialspike
        if args.arch == 'res34':
            snn = SpikeModel(model=ann, sim_length=sim_length,
                        specials=res_spcials_res34, maxspike=args.maxspike)
        else:
            snn = SpikeModel(model=ann, sim_length=sim_length,
                        specials=vgg_specials if args.arch == 'VGG16' else res_spcials, maxspike=args.maxspike)
        snn.cuda()
        mse = True
        get_maximum_activation(train_loader, model=snn, momentum=0.9, iters=5, mse=mse, percentile=None, maxspike=args.maxspike,
                            sim_length=sim_length, channel_wise=True, dist_avg=initialized)
        end = time.time()
        if args.search:
            optimal_maxspike_list, node_list = sensitivity_anylysis(train_loader, model=snn, maxspike=args.maxspike, maxspike_ratio=args.maxspike_ratio, sim_length=sim_length, dist_avg=initialized, disred_maxspike=args.desired_maxspike, minspike=args.minspike, method=args.method, metric=args.metric)
            index = 0
            for m in snn.modules():
                if isinstance(m, SpikeModule):
                    m.maxspike = optimal_maxspike_list[index]
                    index += 1
            get_maximum_activation(train_loader, model=snn, momentum=0.9, iters=5, mse=mse, percentile=None, maxspike=args.maxspike,
                                        sim_length=sim_length, channel_wise=True, dist_avg=initialized)
            if args.calib == 'light':
                bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False, dist_avg=initialized)
            if args.search_threshold:
                optimal_maxspike_list, node_list = sensitivity_anylysis_threshold(train_loader, model=snn, maxspike=args.maxspike, threshold_ratio=args.threshold_ratio, sim_length=sim_length, method=args.method, metric=args.metric)
            index = 0
            for m in snn.modules():
                if isinstance(m, SpikeModule) or isinstance(m, SpikeResModule):
                    m.spike_counter = 0
                    if args.search_threshold:
                        m.threshold = m.threshold * optimal_maxspike_list[index]
                    index += 1
            snn.set_spike_state(use_spike=True)
            acc, spikecount = validate_model(test_loader, snn)
        else:
            if args.calib == 'light':
                bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False, dist_avg=initialized)
            if args.search_threshold:
                optimal_maxspike_list, node_list = sensitivity_anylysis_threshold(train_loader, model=snn, maxspike=args.maxspike, threshold_ratio=args.threshold_ratio, sim_length=sim_length, method=args.method, metric=args.metric)
            index = 0
            for m in snn.modules():
                if isinstance(m, SpikeModule) or isinstance(m, SpikeResModule):
                    m.spike_counter = 0
                    if args.search_threshold:
                        m.threshold = m.threshold * optimal_maxspike_list[index]
                    index += 1
            snn.set_spike_state(use_spike=True)
            acc, spikecount = validate_model(test_loader, snn)
    print(acc)
    print(spikecount)
import argparse
import os
import random
from unittest import result

import numpy as np
import torch

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from general_utils.calibration import bias_corr_model
from general_utils.spiking_layer import SpikeModule, SpikeModel, get_maximum_activation, sensitivity_anylysis, sensitivity_anylysis_threshold
from general_utils.fold_bn import search_fold_and_remove_bn

from models.dataset import ShapeNetDataset, ModelNetDataset
from models.model import PointNetDenseCls, PointNetCls

from tqdm import tqdm
import time
import matplotlib.pyplot as plt

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
def validate_model(test_loader, ann, args, num_classes=4):
    total_correct = 0
    for i,data in tqdm(enumerate(test_loader, 0)):
        with torch.no_grad():
            points, target = data
            points, target = points.cuda(), target.cuda()
            pred = ann(points)
            pred_choice = pred.data.max(1)[1]
            total_correct += pred_choice.eq(target[:, 0].data).cpu().sum()
    test_accuracy = 100. * total_correct / len(test_dataset)
    print(f'Test Acc: {test_accuracy}%')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dpath', default='/home/dataset', type=str, help='dataset directory')
    parser.add_argument('--ckptpath', default='/home/output', type=str, help='checkpoint path')
    parser.add_argument('--model', default='', type=str, help='model path')
    parser.add_argument('--seed', default=1000, type=int, help='random seed to reproduce results')
    parser.add_argument('--batch_size', default=32, type=int, help='minibatch size')
    parser.add_argument('--device', default='', type=str, help='device select')
    parser.add_argument('--calib', default='light', type=str, help='calibration methods',
                        choices=['none', 'light', 'advanced'])
    parser.add_argument('--T', default=16, type=int, help='snn simulation length')
    parser.add_argument('--usebn', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--maxspike', default=4, type=int, help='max fire times')
    parser.add_argument('--minspike', default=1, type=int, help='max fire times')
    parser.add_argument('--initialspike', default=8, type=int, help='max fire times')
    parser.add_argument('--desired_maxspike', default=4, type=int, help='max fire times')
    parser.add_argument('--search', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--search_threshold', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--searchtime', default=1, type=int, help='max fire times')
    parser.add_argument('--threshold_ratio', default=1, type=float, help='max fire times')
    parser.add_argument('--maxspike_ratio', default=1, type=float, help='max fire times')
    parser.add_argument('--method', default='pruning', type=str, help='network architecture')
    parser.add_argument('--metric', default='kl', type=str, help='network architecture')
    parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
    parser.add_argument('--feature_transform', type=bool, default=True, help="use feature transform")
    parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    parser.add_argument(
        '--num_points', type=int, default=2500, help='input batch size')

    args = parser.parse_args()
    acc = 0
    spikecount = 0
    use_bn = args.usebn

    device = args.device
    if args.device == '':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_val_result = 0

    # run experiments with different random seeds, 
    number_of_seeds = 5
    for i in range(number_of_seeds):
        start = time.time()
        seed_all(seed=args.seed + i)
        sim_length = args.T

        if args.dataset_type == 'shapenet':
            dataset = ShapeNetDataset(
                root=args.dpath,
                classification=True,
                npoints=args.num_points)
            test_dataset = ShapeNetDataset(
                root=args.dpath,
                classification=True,
                split='test',
                npoints=args.num_points,
                data_augmentation=False)
        elif args.dataset_type == 'modelnet40':
            dataset = ModelNetDataset(
                root=args.dpath,
                split='trainval',
                npoints=args.num_points)
            test_dataset = ModelNetDataset(
                root=args.dpath,
                split='test',
                data_augmentation=False,
                npoints=args.num_points)
            
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(args.workers))


        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(args.workers))
        num_classes = len(dataset.classes)
        ann = PointNetCls(k=num_classes, feature_transform=args.feature_transform)
        load_path = args.ckptpath

        state_dict = torch.load(load_path, map_location=device)
        ann.load_state_dict(state_dict, strict=True)
        search_fold_and_remove_bn(ann)
        ann.cuda()

        # performance of the ANN pretrained model
        # validate_model(test_loader, ann)

        if args.search:
            args.desired_maxspike = args.maxspike
            args.maxspike = args.initialspike

        snn = SpikeModel(model=ann, sim_length=sim_length, maxspike=args.maxspike)
        snn.cuda()
        print(snn)

        mse = False if args.calib == 'none' else True
        end = time.time()
        # print(f"Time for loading model: {end - start}")


        start = time.time()
        get_maximum_activation(train_loader, model=snn, momentum=0.9, iters=5, mse=mse, percentile=None, maxspike=args.maxspike,
                            sim_length=sim_length, channel_wise=True)
        end = time.time()
        # print(f"Time for get_maximum_activation: {end - start}")

        # snn.set_spike_state(use_spike=True)
        # results_list.append(validate_model(test_loader, snn))
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
                if isinstance(m, SpikeModule):
                    m.count = True
                    m.spike_counter = 0
                    if args.search_threshold:
                        m.threshold = m.threshold * optimal_maxspike_list[index]
                    index += 1
            bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False)
            snn.set_spike_state(use_spike=True)
            # results_list.append(validate_model(test_loader, snn))
            validate_model(test_loader, snn, args, num_classes)

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
                if isinstance(m, SpikeModule):
                    m.count = True
                    m.spike_counter = 0
                    if args.search_threshold:
                        m.threshold = m.threshold * optimal_maxspike_list[index]
                    # m.threshold = m.threshold * optimal_maxspike_list[index]
                    index += 1

            # snn.set_spike_state(use_spike=True)
            # acc, spikecount = validate_model(test_loader, snn)
            bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False)
            snn.set_spike_state(use_spike=True)
            validate_model(test_loader, snn, args, num_classes)

    # print(acc)
    # print(spikecount)

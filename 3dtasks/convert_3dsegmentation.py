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
from general_utils.spiking_layer import SpikeModule
from general_utils.fold_bn import search_fold_and_remove_bn
from general_utils.spiking_layer import SpikeModel, get_maximum_activation, sensitivity_anylysis

from models.dataset import ShapeNetDataset
from models.model import PointNetDenseCls

from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from show3d_balls import showpoints
from torch.autograd import Variable


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
    shape_ious = []
    for i,data in tqdm(enumerate(test_loader, 0)):
        points, target = data
        points, target = points.cuda(), target.cuda()
        ann = ann.eval()
        pred = ann(points)
        pred_choice = pred.data.max(2)[1]

        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy() - 1

        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)#np.unique(target_np[shape_idx])
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))

    print("mIOU for class {}: {}".format(args.class_choice, np.mean(shape_ious)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--arch', default='VGG16', type=str, help='network architecture', choices=['VGG16', 'res20'])
    parser.add_argument('--dpath', default='/home/yuetong/ziqing/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0', type=str, help='dataset directory')
    parser.add_argument('--model', default='', type=str, help='model path')
    parser.add_argument('--seed', default=1000, type=int, help='random seed to reproduce results')
    parser.add_argument('--batch_size', default=32, type=int, help='minibatch size')
    parser.add_argument('--device', default='', type=str, help='device select')
    parser.add_argument('--calib', default='light', type=str, help='calibration methods',
                        choices=['none', 'light', 'advanced'])
    parser.add_argument('--T', default=8, type=int, help='snn simulation length')
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

        dataset = ShapeNetDataset(
            root=args.dpath,
            classification=False,
            class_choice=[args.class_choice])
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(args.workers))

        test_dataset = ShapeNetDataset(
            root=args.dpath,
            classification=False,
            class_choice=[args.class_choice],
            split='test',
            data_augmentation=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(args.workers))
        num_classes = dataset.num_seg_classes
        ann = PointNetDenseCls(k=num_classes, feature_transform=args.feature_transform)
        # print(ann)

        state_dict = torch.load(args.model, map_location=device)
        ann.load_state_dict(state_dict, strict=True)
        search_fold_and_remove_bn(ann)
        ann.cuda()

        # validate_model(test_loader, ann)

        if args.search:
            args.desired_maxspike = args.maxspike
            args.maxspike = args.initialspike

        snn = SpikeModel(model=ann, sim_length=sim_length, maxspike=args.maxspike)
        snn.cuda()
        # print(snn)

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
            index = 0
            for m in snn.modules():
                if isinstance(m, SpikeModule):
                    m.count = True
                    m.spike_counter = 0
                    if args.search_threshold:
                        m.threshold = m.threshold * optimal_maxspike_list[index]
                    index += 1
            snn.set_spike_state(use_spike=True)
            bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False)
            snn.set_spike_state(use_spike=True)

            if not os.path.exists(f"snn_maxspike{args.maxspike}_T{args.T}"):
                os.makedirs(f"snn_maxspike{args.maxspike}_T{args.T}")
            for idx in tqdm(range(len(dataset))):
                point, seg = dataset[idx]
                point_np = point.numpy().transpose(1, 0)
                point = Variable(point.view(1, point.size()[0], point.size()[1]))
                pred = snn(point.cuda())
                pred_choice = pred.data.max(2)[1]
                cmap = plt.cm.get_cmap("hsv", 10)
                cmap = np.array([cmap(i) for i in range(10)])[:, :3]
                gt = cmap[seg.numpy() - 1, :]
                pred_color = cmap[pred_choice.cpu().numpy()[0], :]
                filename = f"snn_maxspike{args.maxspike}_T{args.T}/{args.class_choice}_{idx}.png"
                showpoints(point_np, gt, pred_color, filename=filename)

        else:
            bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False)
            snn.set_spike_state(use_spike=True)

            if not os.path.exists(f"snn_maxspike{args.maxspike}_T{args.T}"):
                os.makedirs(f"snn_maxspike{args.maxspike}_T{args.T}")
            for idx in tqdm(range(len(dataset))):
                point, seg = dataset[idx]
                point_np = point.numpy().transpose(1, 0)
                point = Variable(point.view(1, point.size()[0], point.size()[1]))
                pred = snn(point.cuda())
                pred_choice = pred.data.max(2)[1]
                cmap = plt.cm.get_cmap("hsv", 10)
                cmap = np.array([cmap(i) for i in range(10)])[:, :3]
                gt = cmap[seg.numpy() - 1, :]
                pred_color = cmap[pred_choice.cpu().numpy()[0], :]
                filename = f"snn_maxspike{args.maxspike}_T{args.T}/{args.class_choice}_{idx}.png"
                showpoints(point_np, gt, pred_color, filename=filename)


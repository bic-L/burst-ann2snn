import argparse
import os
import random
from unittest import result
from torch import nn
import numpy as np
import torch

from main_train_cifardvs import ActionRecognition
from models.calibration import bias_corr_model
from models.CIFAR10DVS.models.resnet import res_spcials, SpikeResModule
from models.CIFAR10DVS.models.resnet import spiking_resnet18, spiking_resnet34
from models.spiking_layer import SpikeModule
# from torchvision.models import resnet101, ResNet101_Weights, resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, VGG16_BN_Weights, vgg16_bn
import torchvision.models as models
from models.fold_bn import search_fold_and_remove_bn
from models.spiking_layer import SpikeModel, get_maximum_activation, sensitivity_anylysis, sensitivity_anylysis_threshold
from tqdm import tqdm
from dataset import Cifar10DVS, Dvs128Gesture
from main_train_cifardvs import NCars, NCaltech101

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
    ann.eval()
    device = next(ann.parameters()).device
    spikecount = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        print('targets:', targets)
        inputs = inputs.to(device)
        outputs = ann(inputs)
        _, predicted = outputs.cpu().max(1)
        print('predicted:', predicted)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
        for m in ann.modules():
            if isinstance(m, SpikeModule) or isinstance(m, SpikeResModule):
                spikecount += m.spike_counter
            # print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    # print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    # print(spikecount)
    return 100 * correct / total, spikecount


# def load_model(name, num_classes):
#     # 获取对应的预训练模型
#     if name == "resnet18":
#         resnet = models.resnet18()
#     elif name == "resnet34":
#         resnet = models.resnet34()
#     elif name == "resnet50":
#         resnet = models.resnet50()
#     # ... 可以继续添加其他ResNet类型
#     else:
#         raise ValueError(f"Unknown ResNet type: {name}")
    
#     # 替换ResNet的最后一个全连接层，以适应新的类别数
#     num_ftrs = resnet.fc.in_features
#     resnet.fc = nn.Linear(num_ftrs, num_classes)
    
#     # 定义一个新的模型
#     class ResNet_DVS(nn.Module):
#         def __init__(self, base_model):
#             super(ResNet_DVS, self).__init__()
#             self.reduction = nn.Conv2d(18, 3, kernel_size=3, padding=1)
#             self.base_model = base_model
        
#         def forward(self, x):
#             x = self.reduction(x)
#             x = self.base_model(x)
#             return x

#     return ResNet_DVS(resnet)



def load_data(dataset, batch_size, distributed=False):
    # Data loading code
    # print("Loading data")

    if dataset == 'cifar10dvs':
        dataset_train, dataset_test = Cifar10DVS(root="/home/dataset/Cifar10DVS", resolution=(128, 128))
        nb_classes = 10
        # origin_set = cifar10_dvs.CIFAR10DVS(root="/home/dataset/Cifar10DVS", data_type='frame', frames_number=T, split_by='number')
        # dataset_train, dataset_test = split_to_train_test_set(0.9, origin_set, 10)
    elif dataset == 'dvs128gesture':
        dataset_train, dataset_test = Dvs128Gesture(root="/home/hostm-10/dataset/DvsGesture", resolution=(128, 128))
        nb_classes = 4
        # dataset_train = DVS128Gesture(root="/home/dataset/DVSGesture", train=True, data_type='frame', frames_number=T, split_by='number')
        # dataset_test = DVS128Gesture(root="/home/dataset/DVSGesture", train=False, data_type='frame', frames_number=T, split_by='number')
    elif dataset == 'ncars':
        event_resolution = (100, 120)
        train_dataset = "/home/dataset/N-Cars/train"
        validation_dataset = "/home/dataset/N-Cars/test"
        dataset_train = NCars(train_dataset, True, event_resolution)
        dataset_test = NCars(validation_dataset, False, event_resolution)
        nb_classes = 2
    elif dataset == 'ncaltech101':
        event_resolution = (180, 240)
        train_dataset = "/home/dataset/N-Caltech101/training"
        validation_dataset = "/home/dataset/N-Caltech101/validation"
        dataset_train = NCaltech101(train_dataset, True, event_resolution)
        dataset_test = NCaltech101(validation_dataset, False, event_resolution)
        nb_classes = 101
    elif dataset == 'actionrecognition':
        event_resolution = (260,346)
        train_dataset = "/home/dataset/falldetection/Action_Recognition/train"
        validation_dataset = "/home/dataset/falldetection/Action_Recognition/test"
        # validation_dataset = "/home/yuetong/ziqing/SNN_Calibration/our_dataset"
        dataset_train = ActionRecognition(train_dataset, False, (260,346))
        dataset_test =  ActionRecognition(validation_dataset, False, (260,346))
        nb_classes = 10
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        sampler=sampler_train,  # Use the sampler here
        num_workers=8,
        drop_last=True,
        pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        sampler=sampler_test,  # Use the sampler here
        num_workers=8,
        drop_last=False,
        pin_memory=True)

    return data_loader, data_loader_test, nb_classes



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name')
    parser.add_argument('--arch', default='VGG16', type=str, help='network architecture')
    parser.add_argument('--dpath', default='/home/dataset', type=str, help='dataset directory')
    parser.add_argument('--model', default='', type=str, help='model path')
    parser.add_argument('--seed', default=1000, type=int, help='random seed to reproduce results')
    parser.add_argument('--batch_size', default=90, type=int, help='minibatch size')
    parser.add_argument('--device', default='', type=str, help='device select')
    parser.add_argument('--calib', default='none', type=str, help='calibration methods',
                        choices=['none', 'light', 'advanced'])
    parser.add_argument('--T', default=16, type=int, help='snn simulation length')
    parser.add_argument('--usebn', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--maxspike', default=1, type=int, help='max fire times')
    parser.add_argument('--minspike', default=2, type=int, help='max fire times')
    parser.add_argument('--initialspike', default=8, type=int, help='max fire times')
    parser.add_argument('--desired_maxspike', default=4, type=int, help='max fire times')
    parser.add_argument('--search', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--search_threshold', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--searchtime', default=1, type=int, help='max fire times')
    parser.add_argument('--threshold_ratio', default=0.95, type=float, help='max fire times')
    parser.add_argument('--maxspike_ratio', default=0.95, type=float, help='max fire times')
    parser.add_argument('--method', default='pruning', type=str, help='network architecture')
    parser.add_argument('--metric', default='kl', type=str, help='network architecture')

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

        seed_all(seed=args.seed + i)
        sim_length = args.T

        # use_cifar10 = args.dataset == 'CIFAR10'

        train_loader, test_loader, num_classes = load_data(args.dataset, args.batch_size)

        # ann = load_model(args.arch, 10)
        if args.arch == 'resnet18':
            ann  = spiking_resnet18(num_classes=num_classes)
        elif args.arch == 'resnet34':
            ann  = spiking_resnet34(num_classes=num_classes)


        # load_path = 'raw/' + args.dataset + '/' + args.arch + '_wBN_wd5e4_state_dict.pth' if use_bn else \
        #     'raw/' + args.dataset + '/' + args.arch + '_woBN_wd1e4_state_dict.pth'
        # if args.model != '':
        #     load_path = args.model
        load_path = '/home/yuetong/ziqing/SNN_Calibration/logs/' + args.dataset + '_' + args.arch +  '_avg_T16_wd0.06_adamw_lr0.001/checkpoint_max_test_acc1.pth'

        state_dict = torch.load(load_path, map_location=device)['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('base_model.'):
                new_key = k.replace('base_model.', '')
            else:
                new_key = k
            new_state_dict[new_key] = v

        # 将处理后的权重加载到模型中
        ann.load_state_dict(new_state_dict, strict=True)

        # state_dict = {k.replace('base_model.', ''): v for k, v in state_dict.items() if k.startswith('base_model.')}
        
        # ann.load_state_dict(state_dict, strict=True)
        search_fold_and_remove_bn(ann)
        ann.cuda()
        # validate_model(test_loader, ann)

        if args.search:
            args.desired_maxspike = args.maxspike
            args.maxspike = args.initialspike

        snn = SpikeModel(model=ann, sim_length=sim_length, specials=res_spcials, maxspike=args.maxspike)
        snn.cuda()
        # print(snn)

        mse = False if args.calib == 'none' else True
        get_maximum_activation(train_loader, model=snn, momentum=0.9, iters=5, mse=mse, percentile=None, maxspike=args.maxspike,
                            sim_length=sim_length, channel_wise=True)
        # snn.set_spike_state(use_spike=True)
        # results_list.append(validate_model(test_loader, snn))
        if args.search:
            for i in range(args.searchtime):

                optimal_maxspike_list, node_list = sensitivity_anylysis(train_loader, model=snn, maxspike=args.maxspike, maxspike_ratio=args.maxspike_ratio, sim_length=sim_length, disred_maxspike=args.desired_maxspike, minspike=args.minspike, metric=args.metric, method=args.method)
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
                    # m.count = True
                    m.spike_counter = 0
                    if args.search_threshold:
                        m.threshold = m.threshold * optimal_maxspike_list[index]
                    index += 1
            snn.set_spike_state(use_spike=True)
            # results_list.append(validate_model(test_loader, snn))
            acc, spikecount = validate_model(test_loader, snn)

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
                    # m.count = True
                    m.spike_counter = 0
                    if args.search_threshold:
                        m.threshold = m.threshold * optimal_maxspike_list[index]
                    # m.threshold = m.threshold * optimal_maxspike_list[index]
                    index += 1
            snn.set_spike_state(use_spike=True)
            # results_list.append(validate_model(test_loader, snn))
            acc, spikecount = validate_model(test_loader, snn)

    print(acc)
    print(spikecount)

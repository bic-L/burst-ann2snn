import datetime
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import math
from os import listdir
from os.path import join
from torch.cuda import amp
import torchvision.models as models
from models.resnet import spiking_resnet18, spiking_resnet34, spiking_resnet50, spiking_resnet101
from models.vgg import vgg11, vgg16
# import pre_act_model_voxel
import general_utils.utils as utils
from spikingjelly.clock_driven import functional
from timm.models import create_model
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import SoftTargetCrossEntropy
import autoaugment
from augment import EventAugment
_seed_ = 42
import random
random.seed(42)
root_path = os.path.abspath(__file__)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
import wandb
np.random.seed(_seed_)
# writer = SummaryWriter("./")
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--dataset', default='cifar10dvs', help='dataset, cifar10dvs or dvs128gesture')
    parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                        help='number of label classes (default: 1000)')
    parser.add_argument('--data-path', default='/datase/CIFAR10DVS', help='dataset')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--local-rank', default=-1, type=int)

    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./logs', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        # default=True,
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', default=True, action='store_true',
                        help='Use AMP training')


    # distributed training parameters
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', default=True,  action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=16, type=int, help='simulation steps')
    # parser.add_argument('--adam', default=True, action='store_true',
    #                     help='Use Adam')

    # Optimizer Parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--weight-decay', default=0.06, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum for SGD. Adam will not use momentum')

    parser.add_argument('--connect_f', default='ADD', type=str, help='element-wise connect function')
    parser.add_argument('--T_train', default=None, type=int)

    #Learning rate scheduler
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup', type=float, default=0.5,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--no-wandb', action='store_true', help='disable wandb logging')
    parser.add_argument('--if_pretrain', action='store_true', help='disable wandb logging')
    args = parser.parse_args()
    return args

def load_model(name, num_classes, pretrain):
    # 获取对应的预训练模型
    if name == "resnet18":
        resnet = spiking_resnet18(pretrained=True, num_classes=num_classes)
    elif name == "resnet34":
        resnet = spiking_resnet34(pretrained=True, num_classes=num_classes)
    elif name == "resnet50":
        resnet = spiking_resnet50(pretrained=True, num_classes=num_classes)
    elif name == "resnet101":
        resnet = spiking_resnet101(pretrained=True, num_classes=num_classes)
    elif name == "vgg11":
        resnet = vgg11(pretrained=pretrain, num_classes=num_classes)
    elif name == "vgg16":
        resnet = vgg16(pretrained=pretrain, num_classes=num_classes)
    # ... 可以继续添加其他ResNet类型
    else:
        raise ValueError(f"Unknown ResNet type: {name}")
    return resnet



def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None, T_train=None, aug=None, trival_aug=None, mixup_fn=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        image = image.float()  # [N, T, C, H, W]
        N,C,H,W = image.shape
        # print(image.shape)
        # # N,T,C,H,W = image.shape            
        if aug != None:
            image = torch.stack([(aug(image[i])) for i in range(N)])
        if trival_aug != None:
            image = torch.stack([(trival_aug(image[i])) for i in range(N)])

        if mixup_fn is not None:
            image, target = mixup_fn(image, target)
            target_for_compu_acc = target.argmax(dim=-1)


        if T_train:
            sec_list = np.random.choice(image.shape[1], T_train, replace=False)
            sec_list.sort()
            image = image[:, sec_list]

        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)
        if mixup_fn is not None:
            acc1, acc5 = utils.accuracy(output, target_for_compu_acc, topk=(1, 2))
        else:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 2))
        batch_size = image.shape[0]
        loss_s = loss.item()
        # if math.isnan(loss_s):
        #     raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image = image.float()
            output = model(image)
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 2))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5


def load_data(dataset, dataset_dir, distributed, T):
    # Data loading code
    print("Loading data")
    st = time.time()
    if dataset == 'cifar10dvs':
        dataset_train, dataset_test = Cifar10DVS(root="/dataset/CIFAR10DVS", resolution=(128, 128))
        nb_classes = 10
    elif dataset == 'dvs128gesture':
        dataset_train, dataset_test = Dvs128Gesture(root="/dataset/DvsGesture", resolution=(128, 128))
        nb_classes = 11
    elif dataset == 'ncars':
        event_resolution = (100, 120)
        train_dataset = "/home/dataset/N-Cars/train"
        validation_dataset = "/home/dataset/N-Cars/test"
        dataset_train = NCars(train_dataset, True, event_resolution)
        dataset_test = NCars(validation_dataset, False, event_resolution)
        nb_classes = 2
    elif dataset == 'ncaltech101':
        event_resolution = (180, 240)
        train_dataset = "/dataset/dataset/N-Caltech101/training"
        validation_dataset = "/dataset/dataset/N-Caltech101/validation"
        dataset_train = NCaltech101(train_dataset, True, event_resolution)
        dataset_test = NCaltech101(validation_dataset, False, event_resolution)
        nb_classes = 101
    elif dataset == 'asldvs':    
        event_resolution = (180, 240)
        train_dataset = "/home/dataset/ASLDVS/"
        dataset_train, dataset_test = AslDVS(train_dataset)
        nb_classes = 24
    elif dataset == 'actionrecognition':
        event_resolution = (260,346)
        train_dataset = "/home/dataset/falldetection/Action_Recognition/train"
        validation_dataset = "/home/dataset/falldetection/Action_Recognition/test"
        dataset_train = ActionRecognition(train_dataset, False, (260,346))
        dataset_test =  ActionRecognition(validation_dataset, False, (260,346))
        nb_classes = 10
    print("Took", time.time() - st)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler, nb_classes


def main(args):

    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.

    utils.init_distributed_mode(args)
    print(args)

    output_dir = os.path.join(args.output_dir, f'{args.dataset}_{args.model}_avg_T{args.T}_ann')

    if args.T_train:
        output_dir += f'_Ttrain{args.T_train}'

    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'

    if args.opt == 'adamw':
        output_dir += '_adamw'
    else:
        output_dir += '_sgd'

    output_dir += f'_lr{args.lr}'
    
    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    device = torch.device(args.device)
    data_path = args.data_path
    dataset_train, dataset_test, train_sampler, test_sampler, nb_classes = load_data(args.dataset, data_path, args.distributed, args.T)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True)

    args.num_classes = nb_classes
    model = load_model(args.model, args.num_classes, args.if_pretrain)
    model = torch.compile(model)
    print("Creating model")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion_train = SoftTargetCrossEntropy().cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(args, model)
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

    if args.test_only:

        evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        return

    if utils.is_main_process() and not args.no_wandb:
        wandb.init(project="spikepruning", entity="spikingtransformer", config=args, name=f'{args.dataset}_{args.model}_{args.batch_size}_prtrain_{args.if_pretrain}', reinit=True)

    train_snn_aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5)
                    ])
    train_trivalaug = autoaugment.SNNAugmentWide()
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        mixup_fn = Mixup(**mixup_args)
    print("Start training")
    start_time = time.time()
    max_accuracy = 0
    for epoch in range(args.start_epoch, num_epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if epoch >= int(0.75*num_epochs):
            mixup_fn.mixup_enabled = False
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, criterion_train, optimizer, data_loader, device, epoch,
            args.print_freq, scaler, args.T_train,
            train_snn_aug, train_trivalaug, mixup_fn)
        if utils.is_main_process() and not args.no_wandb:
            try:
                wandb.log({"train_loss": train_loss}, step=epoch)
                wandb.log({'lr': float(optimizer.state_dict()['param_groups'][0]['lr'])}, step=epoch)
            except Exception as e:
                print(f"Warning: wandb logging failed: {e}")
        lr_scheduler.step(epoch + 1)

        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        # if te_tb_writer is not None:
        max_accuracy = max(max_accuracy, test_acc1)
        if utils.is_main_process() and not args.no_wandb:
            try:
                wandb.log({"test_loss": test_loss}, step=epoch)
                wandb.log({"test_acc": test_acc1}, step=epoch)
                wandb.log({"max_acc": max_accuracy}, step=epoch)
            except Exception as e:
                print(f"Warning: wandb logging failed: {e}")


        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True


        if output_dir:

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
            }

            if save_max:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
        print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1, 'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)
        print(output_dir)
    if output_dir:
        utils.save_on_master(
            checkpoint,
            os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

    return max_test_acc1

from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.asl_dvs import ASLDVS 
from spikingjelly.datasets.nav_gesture import NAVGestureWalk
from spikingjelly.datasets.nav_gesture import NAVGestureSit

class SpikingjellyDataset:

    def __init__(self, dataset, train, resolution):
        self.dataset = dataset
        if train:
            self.event_augment = EventAugment(resolution)
        else:
            self.event_augment = None
        self.quantization_layer = QuantizationLayerVoxGrid((9,  128, 128))
        self.crop_dimension = (224, 224)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dict_events, label = self.dataset[idx]

        x = dict_events['x'].astype(np.float32)
        y = dict_events['y'].astype(np.float32)
        t = dict_events['t'].astype(np.float32)
        p = dict_events['p'].astype(np.float32)
        
        events = torch.from_numpy(np.concatenate([x[:, np.newaxis], y[:, np.newaxis], t[:, np.newaxis], p[:, np.newaxis]], axis=1))
        # print(events.shape)
        # print(events)
        if self.event_augment is not None and random.random() < 0.5:
            events = self.event_augment(events)
        events =torch.cat([events, torch.zeros(len(events), 1)], dim=1)
        vox = self.quantization_layer.forward(events)
        events = self.resize_to_resolution(vox)
        events = events.squeeze(0)

        return events, label

    def resize_to_resolution(self, x):
        B, C, H, W = x.shape
        if H > W:
            ZeroPad = nn.ZeroPad2d(padding=(int((H - W) / 2), int((H - W) / 2), 0, 0))
        else:
            ZeroPad = nn.ZeroPad2d(padding=(0, 0, int((W - H) / 2), int((W - H) / 2)))
        y = ZeroPad(x)
        y = F.interpolate(y, size=self.crop_dimension)
        return y


def Cifar10DVS(root, resolution=(128, 128)):
    dataset = CIFAR10DVS(root, data_type="event")
    train_set, test_set = split_to_train_test_set(0.9, dataset, num_classes=10)
    return SpikingjellyDataset(train_set, True, resolution=resolution), SpikingjellyDataset(test_set, False, resolution=resolution)

def Dvs128Gesture(root, resolution=(128, 128)): #class 11
    train_set = DVS128Gesture(root, train=True, data_type="event")
    test_set =  DVS128Gesture(root, train=False, data_type="event")
    return SpikingjellyDataset(train_set, True, resolution=resolution), SpikingjellyDataset(test_set, False, resolution=resolution)

def AslDVS(root, saltnoise):
    dataset = ASLDVS(root, data_type="event")
    print(len(dataset))
    train_set, test_set = split_to_train_test_set(0.9, dataset, num_classes=24)
    return SpikingjellyDataset(train_set, True, resolution=(180, 240), saltnoise=saltnoise), SpikingjellyDataset(test_set, False, resolution=(180, 240), saltnoise=saltnoise)

def NavGestureWalk(root, saltnoise):
    dataset = NAVGestureWalk(root, data_type="event")
    train_set, test_set = split_to_train_test_set(0.7, dataset, num_classes=6)
    return SpikingjellyDataset(train_set, True, resolution=(240, 304), saltnoise=saltnoise), SpikingjellyDataset(test_set, False, resolution=(304, 240), saltnoise=saltnoise)

def NavGestureSit(root, saltnoise):
    dataset = NAVGestureSit(root, data_type="event")
    train_set, test_set = split_to_train_test_set(0.9, dataset, num_classes=6)
    return SpikingjellyDataset(train_set, True, resolution=(240, 304), saltnoise=saltnoise), SpikingjellyDataset(test_set, False, resolution=(304, 240), saltnoise=saltnoise)


class QuantizationLayerVoxGrid(nn.Module):
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim

    def forward(self, events):
        epsilon = 10e-3
        B = int(1+events[-1, -1].item())
        # tqdm.write(str(B))
        num_voxels = int(2 * np.prod(self.dim) * B)
        C, H, W = self.dim
        vox = events[0].new_full([num_voxels, ], fill_value=0)
        # get values for each channel
        x, y, t, p, b = events.T

        for bi in range(B):
            # tqdm.write(str(t[events[:, -1] == bi].shape))
            t[events[:, -1] == bi] /= t[events[:, -1] == bi].max()

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b
        for i_bin in range(C):
            values = torch.zeros_like(t)
            values[(t > i_bin/C) & (t <= (i_bin+1)/C)] = 1

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1) # (B, 2, H, W)
        return vox
    


class NCaltech101:
    def __init__(self, root, train, resolution):
        self.classes = listdir(root)
        self.classes.sort()
        self.files = []
        self.labels = []
        if train:
            self.event_augment = EventAugment(resolution)
        else:
            self.event_augment = None
        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)
        self.np_labels = np.array(self.labels)
        self.quantization_layer = QuantizationLayerVoxGrid((9,  180,  240))
        self.crop_dimension = (224,224)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)
        events[:, 3] = (events[:, 3] + 1) / 2
        events = torch.from_numpy(events)
        if self.event_augment is not None and random.random() < 0.5:
            events = self.event_augment(events)
        events=torch.cat([events, torch.zeros(len(events), 1)], dim=1)
        vox = self.quantization_layer.forward(events)
        events = self.resize_to_resolution(vox)
        events = events.squeeze(0)
        return events, label
    
    def resize_to_resolution(self, x):
        B, C, H, W = x.shape
        if H > W:
            ZeroPad = nn.ZeroPad2d(padding=(int((H - W) / 2), int((H - W) / 2), 0, 0))
        else:
            ZeroPad = nn.ZeroPad2d(padding=(0, 0, int((W - H) / 2), int((W - H) / 2)))
        y = ZeroPad(x)
        y = F.interpolate(y, size=self.crop_dimension)
        return y


class NCars:
    def __init__(self, root, train, resolution):
        self.classes = listdir(root)
        self.classes.sort()
        self.files = []
        self.labels = []
        if train:
            self.event_augment = EventAugment(resolution)
        else:
            self.event_augment = None

        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)
        self.np_labels = np.array(self.labels)
        self.quantization_layer = QuantizationLayerVoxGrid((9,  100,  120))
        self.crop_dimension = (224, 224)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)
        events[:, 3] = (events[:, 3] + 1) / 2
        events = torch.from_numpy(events)
        if self.event_augment is not None and random.random() < 0.5:
            events = self.event_augment(events)
        events=torch.cat([events, torch.zeros(len(events), 1)], dim=1)
        vox = self.quantization_layer.forward(events)
        events = self.resize_to_resolution(vox)
        events = events.squeeze(0)
        return events, label
    
    def resize_to_resolution(self, x):
        B, C, H, W = x.shape
        if H > W:
            ZeroPad = nn.ZeroPad2d(padding=(int((H - W) / 2), int((H - W) / 2), 0, 0))
        else:
            ZeroPad = nn.ZeroPad2d(padding=(0, 0, int((W - H) / 2), int((W - H) / 2)))
        y = ZeroPad(x)
        y = F.interpolate(y, size=self.crop_dimension)
        return y
    
class ActionRecognition:
    def __init__(self, root, train, resolution):
        self.classes = listdir(root)
        self.classes.sort()
        self.files = []
        self.labels = []
        if train:
            self.event_augment = EventAugment(resolution)
        else:
            self.event_augment = None

        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)
        self.np_labels = np.array(self.labels)
        self.quantization_layer = QuantizationLayerVoxGrid((9, *resolution))
        self.crop_dimension = (224, 224)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)
        # print(events.shape)
        # events[3, :] = (events[3, :] + 1) / 2
        # events = torch.from_numpy(events).transpose(0,1)

        events[:, 3] = (events[:, 3] + 1) / 2
        events = torch.from_numpy(events)
        if self.event_augment is not None and random.random() < 0.5:
            events = self.event_augment(events)
        events=torch.cat([events, torch.zeros(len(events), 1)], dim=1)
        vox = self.quantization_layer.forward(events)
        events = self.resize_to_resolution(vox)
        events = events.squeeze(0)
        # print(events.shape)
        return events, label
    
    def resize_to_resolution(self, x):
        B, C, H, W = x.shape
        if H > W:
            ZeroPad = nn.ZeroPad2d(padding=(int((H - W) / 2), int((H - W) / 2), 0, 0))
        else:
            ZeroPad = nn.ZeroPad2d(padding=(0, 0, int((W - H) / 2), int((W - H) / 2)))
        y = ZeroPad(x)
        y = F.interpolate(y, size=self.crop_dimension)
        return y

if __name__ == "__main__":
    args = parse_args()
    main(args)

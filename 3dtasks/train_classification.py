from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from models.dataset import ShapeNetDataset, ModelNetDataset
from models.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import _LRScheduler
import math

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.01, T_up=0, gamma=0.5, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_up = T_up
        self.gamma = gamma
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_0 - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.T_cur = self.last_epoch % self.T_0
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=64, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lr', default=0.001, type=float, help='default learning rate')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default="/home/yuetong/ziqing/pointnet.pytorch/Modelnet40/ModelNet40", help="dataset path")
parser.add_argument('--dataset_type', type=str, default='modelnet40', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', type=bool, default=True, help="use feature transform")

opt = parser.parse_args()
print(opt)

if opt.dataset_type == 'shapenet':
    opt.dataset = "/home/yuetong/ziqing/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0"
elif opt.dataset_type == 'modelnet40':
    opt.dataset = "/home/yuetong/ziqing/pointnet.pytorch/Modelnet40/ModelNet40"

wandb.init(project="3dclassification", entity="spikingtransformer", name=f'{opt.dataset_type}_lr{opt.lr}', config=opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = 42
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

opt.outf = f'{opt.dataset_type}_lr{opt.lr}'

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))



optimizer = optim.AdamW(classifier.parameters(), lr=opt.lr)
scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=opt.nepoch, T_up=20, eta_max=0.01, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize
best_accuracy = 0.0
interval = 1 # interval for performance test during training 


for epoch in tqdm(range(opt.nepoch)):
    classifier.train()
    total_loss = 0
    correct = 0
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points, target = points.transpose(2, 1).cuda(), target.cuda()
        optimizer.zero_grad()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target[:, 0])
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred_choice = pred.data.max(1)[1]
        correct += pred_choice.eq(target[:, 0].data).cpu().sum()

    scheduler.step()
    training_accuracy = 100. * correct / len(dataset)

    if epoch % interval == 0: 
        classifier.eval()
        total_correct = 0
        for i, data in enumerate(testdataloader, 0):
            with torch.no_grad():
                points, target = data
                points, target = points.transpose(2, 1).cuda(), target.cuda()
                pred, _, _ = classifier(points)
                pred_choice = pred.data.max(1)[1]
                total_correct += pred_choice.eq(target[:, 0].data).cpu().sum()

        test_accuracy = 100. * total_correct / len(test_dataset)
        print(f'Epoch: {epoch+1}, Train Loss: {total_loss/len(dataloader)}, Train Acc: {training_accuracy}%, Test Acc: {test_accuracy}%')

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(classifier.state_dict(), f'{opt.outf}/best_cls_model.pth')
            print(f'New best model saved at epoch {epoch+1} with accuracy: {best_accuracy}%')

        wandb.log({"Train Loss": total_loss / len(dataloader),
                   "Train Accuracy": training_accuracy,
                   "Test Accuracy": test_accuracy})
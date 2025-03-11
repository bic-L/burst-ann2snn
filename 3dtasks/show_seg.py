from __future__ import print_function
from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from models.dataset import ShapeNetDataset
from models.model import PointNetDenseCls
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('--dataset', type=str, default='/home/yuetong/ziqing/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0', help='dataset path')
parser.add_argument('--class_choice', type=str, default='Chair', help='class choice')

opt = parser.parse_args()
print(opt)

dataset = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)

state_dict = torch.load(opt.model)
classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0], feature_transform=True)
classifier.load_state_dict(state_dict)
classifier.eval()

for idx in tqdm(range(len(dataset))):
    point, seg = dataset[idx]
    point_np = point.numpy()
    point = point.transpose(1, 0).contiguous()
    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    pred, _, _ = classifier(point)
    pred_choice = pred.data.max(2)[1]

    cmap = plt.cm.get_cmap("hsv", 10)
    cmap = np.array([cmap(i) for i in range(10)])[:, :3]
    gt = cmap[seg.numpy() - 1, :]
    pred_color = cmap[pred_choice.numpy()[0], :]
    filename = f"ann/{opt.class_choice}_{idx}.png"
    showpoints(point_np, gt, pred_color, filename=filename)
    # break

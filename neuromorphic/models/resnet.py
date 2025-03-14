import torch
import torch.nn as nn

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from general_utils.spiking_layer import SpikeModule
from general_utils.utils import append_to_csv
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from copy import deepcopy

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url

__all__ = ['ResNet', 'spiking_resnet18', 'spiking_resnet34', 'spiking_resnet50', 'spiking_resnet101',
           'spiking_resnet152', 'spiking_resnext50_32x4d', 'spiking_resnext101_32x8d',
           'spiking_wide_resnet50_2', 'spiking_wide_resnet101_2']

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.reduction = nn.Conv2d(18, 3, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.reduction(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class SpikeResModule(SpikeModule):
    """
    Spike-based Module that can handle spatial-temporal information.
    threshold :param that decides the maximum value
    conv :param is the original normal conv2d module
    """

    def __init__(self, sim_length: int, maxspike: int, conv: Union[nn.Conv2d, nn.Linear], enable_shift: bool = True):
        super(SpikeResModule, self).__init__(sim_length, maxspike, conv, enable_shift)

    def forward(self, input: torch.Tensor, residual: torch.Tensor):
        if self.analyze:
            x = self.fwd_func(input, self.weight, self.bias, **self.fwd_kwargs) + residual
            return self.clip_floor(x, self.sim_length, self.threshold)
        if self.use_spike:
            x = (self.fwd_func(input, self.weight, self.bias, **self.fwd_kwargs)) + residual
            x = x +  0.5 / self.sim_length * self.threshold
            self.mem_pot = self.mem_pot + x
            spike = (self.mem_pot / self.threshold).floor()
            temp = spike.clamp(min=0, max=self.maxspike)
            self.spike_counter += temp.sum().item()
            self.cur_t += 1
            if self.count:
                if self.cur_t == self.sim_length:
                    append_to_csv('spikecount.csv', [self.spike_counter])
            spike = temp * self.threshold
            self.mem_pot -= spike
            return spike
        else:
            return self.relu(self.fwd_func(input, self.org_weight, self.org_bias, **self.fwd_kwargs) + residual)



class SpikeBasicBlock(nn.Module):
    def __init__(self, basic_block: BasicBlock, **spike_params):
        super().__init__()
        self.conv1 = SpikeModule(conv=basic_block.conv1, **spike_params)
        self.conv1.add_module('relu', basic_block.relu1)
        self.conv2 = SpikeResModule(conv=basic_block.conv2, **spike_params)
        self.conv2.add_module('relu', basic_block.relu2)
        self.downsample = basic_block.downsample
        self.stride = basic_block.stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv2(out, residual)
        return out


class SpikeBottleneck(nn.Module):
    def __init__(self, bottleneck: Bottleneck, **spike_params):
        super().__init__()
        self.conv1 = SpikeModule(conv=bottleneck.conv1, **spike_params)
        self.conv1.add_module('relu', bottleneck.relu1)
        self.conv2 = SpikeModule(conv=bottleneck.conv2, **spike_params)
        self.conv2.add_module('relu', bottleneck.relu2)
        self.conv3 = SpikeResModule(conv=bottleneck.conv3, **spike_params)
        self.conv3.add_module('relu', bottleneck.relu3)
        self.downsample = bottleneck.downsample
        self.stride = bottleneck.stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv3(out, residual)
        return out


def _spiking_resnet(arch, block, layers, pretrained, progress, num_classes=1000, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # Create a new state_dict where keys match the model's state_dict
        new_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        
        model.load_state_dict(new_state_dict, strict=False)
    
    if num_classes != 1000:
        model.fc = nn.Linear(512 * block.expansion, num_classes)

    return model


def spiking_resnet18(pretrained=False, progress=True, num_classes=1000, **kwargs):
    return _spiking_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes, **kwargs)

def spiking_resnet34(pretrained=False, progress=True, num_classes=1000, **kwargs):
    return _spiking_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, num_classes, **kwargs)

def spiking_resnet50(pretrained=False, progress=True, num_classes=1000, **kwargs):
    return _spiking_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, num_classes, **kwargs)

def spiking_resnet101(pretrained=False, progress=True, num_classes=1000, **kwargs):
    return _spiking_resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, num_classes, **kwargs)

res_spcials = {BasicBlock: SpikeBasicBlock,
               Bottleneck: SpikeBottleneck}
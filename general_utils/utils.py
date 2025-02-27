import torch.distributed as dist
import numpy as np
import torch.nn as nn

def allaverage(tensor):
    """
    Average the tensor across all machines in distributed mode
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def reduce_tensor(tensor):
    """
    Reduces the tensor data across all machines in DDP mode.
    
    Args:
        tensor: data to be reduced
    Returns:
        reduced tensor
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    world_size = dist.get_world_size()
    rt /= world_size
    return rt


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


class AvgPoolConv(nn.Conv2d):
    """
    Converting the AvgPool layers to a convolution-wrapped module,
    so that this module can be identified in Spiking-refactor.
    """

    def __init__(self, kernel_size=2, stride=2, input_channel=64, padding=0, freeze_avg=True):
        super().__init__(input_channel, input_channel, kernel_size, padding=padding, stride=stride,
                         groups=input_channel, bias=False)
        # init the weight to make them equal to 1/k/k
        self.set_weight_to_avg()
        self.freeze = freeze_avg
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *inputs):
        self.set_weight_to_avg()
        x = super().forward(*inputs)
        return self.relu(x)

    def set_weight_to_avg(self):
        self.weight.data.fill_(1).div_(self.kernel_size[0] * self.kernel_size[1])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update(self, tensor, num=1):
        reduced_tensor = reduce_tensor(tensor)
        self.update(reduced_tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count
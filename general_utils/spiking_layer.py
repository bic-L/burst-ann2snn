import copy
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import AverageMeter, allaverage, StraightThrough
import random
from tqdm import tqdm

# ------------------------- New Version ---------------------------

class SpikeModule(nn.Module):
    """
    Spike-based Module that can handle spatial-temporal information.
    threshold :param that decides the maximum value
    conv :param is the original normal conv2d module
    """

    def __init__(self, sim_length: int, maxspike: int, conv: Union[nn.Conv2d, nn.Linear], enable_shift: bool = True,
                 safe_int: bool = True):
        super(SpikeModule, self).__init__()
        if isinstance(conv, nn.Conv2d):
            self.fwd_kwargs = {"stride": conv.stride, "padding": conv.padding,
                               "dilation": conv.dilation, "groups": conv.groups}
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = {}
            self.fwd_func = F.linear
        self.threshold = None
        self.max = None
        self.bkp = None
        self.mem_pot = 0
        self.mem_pot_init = 0
        self.weight = conv.weight
        self.org_weight = copy.deepcopy(conv.weight.data)
        if conv.bias is not None:
            self.bias = conv.bias
            self.org_bias = copy.deepcopy(conv.bias.data)
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the spike forward for default setup
        self.use_spike = False
        self.enable_shift = enable_shift
        self.sim_length = sim_length
        self.cur_t = 0
        self.relu = StraightThrough()
        self.buffer = 0
        self.firing_rate = 0
        self.maxspike = maxspike
        self.active_elements_count = 0
        self.analyze = False
        self.spike_counter = 0.
        self.count = False

    def forward(self, input: torch.Tensor):
        if self.analyze:
            x = self.fwd_func(input, self.weight, self.bias, **self.fwd_kwargs)
            return self.clip_floor(x, self.sim_length, self.threshold)
        if self.use_spike and not isinstance(self.relu, StraightThrough):
            global layer
            layer += 1
            self.cur_t += 1
            x = self.fwd_func(input, self.weight, self.bias, **self.fwd_kwargs)
            x = x + 0.5 / self.sim_length * self.threshold
            self.mem_pot = self.mem_pot + x
            spike = (self.mem_pot / self.threshold).floor()
            temp = spike.clamp_(min=0, max=self.maxspike)
            self.spike_counter += temp.sum().item()
            spike = temp * self.threshold
            self.mem_pot -= spike
            return spike
        else:
            return self.relu(self.fwd_func(input, self.org_weight, self.org_bias, **self.fwd_kwargs))

    def init_membrane_potential(self):
        self.neuron.reset()
        self.mem_pot = self.mem_pot_init if isinstance(self.mem_pot_init, int) else self.mem_pot_init.clone()
        self.cur_t = 0

    def clip_floor(self, tensor: torch.Tensor, T: int, Vth: Union[float, torch.Tensor]):
        snn_out = torch.clamp(tensor / Vth, min=0, max=self.maxspike) * T
        return snn_out.floor() * Vth / T


class SpikeModel(nn.Module):

    def __init__(self, model: nn.Module, sim_length: int, maxspike: int, specials: dict = {}):
        super().__init__()
        self.model = model
        self.specials = specials
        self.spike_module_layers = []
        self.maxspike = maxspike
        self.spike_module_refactor(self.model, sim_length, maxspike)
        self.use_spike = False

        assert sim_length > 0, "SNN does not accept negative simulation length"
        self.T = sim_length

    def spike_module_refactor(self, module: nn.Module, sim_length: int, maxspike: int, prev_module=None):
        """
        Recursively replace the normal conv2d to SpikeConv2d
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        """
        prev_module = prev_module
        for name, immediate_child_module in module.named_children():
            if type(immediate_child_module) in self.specials:
                setattr(module, name, self.specials[type(immediate_child_module)]
                        (immediate_child_module, sim_length=sim_length, maxspike=maxspike))
                self.spike_module_layers.append(module)
            elif isinstance(immediate_child_module, nn.Conv2d):
                setattr(module, name, SpikeModule(sim_length=sim_length, maxspike=maxspike, conv=immediate_child_module))
                self.spike_module_layers.append(module)
                prev_module = getattr(module, name)
            elif isinstance(immediate_child_module, (nn.ReLU, nn.ReLU6)):
                if prev_module is not None:
                    prev_module.add_module('relu', immediate_child_module)
                    setattr(module, name, StraightThrough())
                else:
                    continue
            elif isinstance(immediate_child_module, nn.Linear):
                self.classifier = copy.deepcopy(immediate_child_module)
                # setattr(module, name, StraightThrough())
            else:
                prev_module = self.spike_module_refactor(
                    immediate_child_module, sim_length=sim_length, maxspike=maxspike, prev_module=prev_module)

        return prev_module

    def set_spike_state(self, use_spike: bool = True):
        self.use_spike = use_spike
        for m in self.model.modules():
            if isinstance(m, SpikeModule):
                m.use_spike = use_spike

    def init_membrane_potential(self):
        for m in self.model.modules():
            if isinstance(m, SpikeModule):
                m.init_membrane_potential()

    def forward(self, input):
        if self.use_spike:
            self.init_membrane_potential()
            out = 0
            for sim in range(self.T):
                out += self.model(input)
            out /= self.T
        else:
            self.init_membrane_potential()
            out = self.model(input)
        return out

# ------------------------- Max Activation ---------------------------
class DataSaverHook:
    def __init__(self, momentum: Union[float, None] = 0.9, sim_length: int = 8, maxspike: int = 1,
                 mse: bool = True, percentile: Union[float, None] = None, channel_wise: bool = False,
                 dist_avg: bool = False):
        self.momentum = momentum
        self.max_act = None
        self.bkp_act = None 
        self.T = sim_length
        self.maxspike = maxspike
        self.mse = mse
        self.percentile = percentile
        self.channel_wise = channel_wise
        self.dist_avg = dist_avg

    def __call__(self, module, input_batch, output_batch):
        def get_act_thresh(tensor):
            if self.mse:
                act_thresh = find_threshold_mse(output_batch, T=self.T, maxspike=module.maxspike, channel_wise=self.channel_wise)
            elif self.percentile is not None:
                assert 0. <= self.percentile <= 1.0
                if self.channel_wise:
                    num_channel = output_batch.shape[1]
                    act_thresh = torch.ones(num_channel).type_as(output_batch)
                    for i in range(num_channel):
                        act_thresh[i] = quantile(output_batch[:, i], self.percentile)
                    act_thresh = act_thresh.reshape(1, num_channel, 1, 1)
                else:
                    act_thresh = quantile(output_batch, self.percentile)
            return act_thresh

        if self.max_act is None:
            self.max_act = get_act_thresh(output_batch)
        else:
            cur_max = get_act_thresh(output_batch)
            if self.momentum is None:
                self.max_act = self.max_act if self.max_act > cur_max else cur_max
            else:
                self.max_act = self.momentum * self.max_act + (1 - self.momentum) * cur_max
        if self.dist_avg:
            allaverage(self.max_act)
        module.max = self.max_act
        module.threshold = self.max_act
        module.active_elements_count = output_batch[0].numel()


def quantile(tensor: torch.Tensor, p: float):
    try:
        return torch.quantile(tensor, p)
    except:
        tensor_np = tensor.cpu().detach().numpy()
        return torch.tensor(np.percentile(tensor_np, q=p * 100)).type_as(tensor)


def find_threshold_mse(tensor: torch.Tensor, T: int = 8, maxspike: int = 1, channel_wise: bool = True):
    """
    This function use grid search to find the best suitable
    threshold value for snn.
    :param tensor: the output batch tensor,
    :param T: simulation length
    :param channel_wise: set threshold channel-wise
    :return: threshold with MMSE
    """
    def clip_floor(tensor: torch.Tensor, T: int, maxspike: int, Vth: Union[float, torch.Tensor]):
        snn_out = torch.clamp(tensor / Vth, min=0, max=maxspike) * T
        return snn_out.floor() * Vth / T

    if channel_wise and len(tensor.shape) == 4:
        num_channel = tensor.shape[1]
        max_act = torch.ones(num_channel).type_as(tensor)
        for i in range(num_channel):
            max_act[i] = tensor[:, i].max()
        max_act = max_act.reshape(1, num_channel, 1, 1)
        best_score = torch.ones_like(max_act).mul(1e10)
        best_Vth = torch.clone(max_act)
        for i in range(95):
            new_Vth = max_act * (1.0 - (i * 0.01))
            mse = lp_loss(tensor, clip_floor(tensor, T, maxspike, new_Vth), p=2.0, reduction='channel_split')
            mse = mse.reshape(1, num_channel, 1, 1)
            mask = mse < best_score
            best_score[mask] = mse[mask]
            best_Vth[mask] = new_Vth[mask]
    else:
        max_act = tensor.max()
        best_score = 1e5
        best_Vth = 0
        for i in range(95):
            new_Vth = max_act * (1.0 - (i * 0.01))
            mse = lp_loss(tensor, clip_floor(tensor, T, maxspike, new_Vth), p=2.0, reduction='other')
            if mse < best_score:
                best_Vth = new_Vth
                best_score = mse
    return best_Vth

@torch.no_grad()
def get_maximum_activation(train_loader: torch.utils.data.DataLoader,
                           model: SpikeModel,
                           momentum: Union[float, None] = 0.9,
                           iters: int = 20,
                           sim_length: int = 8,
                           maxspike: int = 1,
                           mse: bool = True, percentile: Union[float, None] = None,
                           channel_wise: bool = False,
                           dist_avg: bool = False):
    """
    This function store the maximum activation in each convolutional or FC layer.
    :param train_loader: Data loader of the training set
    :param model: target model
    :param momentum: if use momentum, the max activation will be EMA updated
    :param iters: number of iterations to calculate the max act
    :param sim_length: sim_length when computing the mse of SNN output
    :param mse: if Ture, use MMSE to find the V_th
    :param percentile: if mse = False and percentile is in [0,1], use percentile to find the V_th
    :param channel_wise: use channel-wise mse
    :param dist_avg: if True, then compute mean between distributed nodes
    :return: model with stored max activation buffer
    """
    # do not use train mode here (avoid bn update)
    model.set_spike_state(use_spike=False)
    model.eval()
    device = next(model.parameters()).device
    hook_list = []
    for m in model.modules():
        if isinstance(m, SpikeModule):
            hook_list += [m.register_forward_hook(DataSaverHook(momentum, sim_length, maxspike, mse, percentile, channel_wise,
                                                                dist_avg))]
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device=device)
        _ = model(input)
        if i > iters:
            break
    for h in hook_list:
        h.remove()


def compute_accuracy(output, target):
    """Compute the accuracy of the predictions."""
    _, predicted = output.max(1)
    total = target.size(0)
    correct = (predicted == target).sum().item()
    return correct / total


@torch.no_grad()
def sensitivity_anylysis(train_loader: torch.utils.data.DataLoader,
                           model: SpikeModel,
                           sim_length: int = 8,
                           maxspike: int = 5,
                           maxspike_ratio: float = 0.95,
                           dist_avg: bool = False,
                           disred_maxspike: int = 1,
                           minspike: int = 1,
                           metric: str = 'accuracy',
                           method: str = 'dp'):
    # do not use train mode here (avoid bn update)
    model.set_spike_state(use_spike=False)
    model.eval()
    device = next(model.parameters()).device
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device=device)
        target = target.to(device=device)
        gt_output = model(input)
        base_acc = compute_accuracy(gt_output, target)
        gt_output = F.softmax(gt_output, dim=1)

        break
    spike_module_layers = []
    for m in model.modules():
        if isinstance(m, SpikeModule):
            spike_module_layers.append(m)

    maxspike_candidates = torch.arange(minspike, maxspike + 1, 1).tolist()
    sen_result = [[0 for _ in range(len(spike_module_layers))] for _ in range(len(maxspike_candidates))]
    energy_result = [[0 for _ in range(len(spike_module_layers))] for _ in range(len(maxspike_candidates))]

    for i in range(len(spike_module_layers)):
        for j, spike in enumerate(maxspike_candidates):
            spike_module_layers[i].spike_counter = 0
            model.set_spike_state(use_spike=True)
            spike_module_layers[i].maxspike = spike
            temp_output = model(input)
            temp_acc = compute_accuracy(temp_output, target)
            temp_output = F.softmax(temp_output, dim=1)
            kl_div = symmetric_kl(temp_output, gt_output)
            sen_result[j][i] = kl_div.item()
            spike_module_layers[i].maxspike = maxspike
            energy_result[j][i] = spike_module_layers[i].spike_counter
            spike_module_layers[i].spike_counter = 0.

    if method == 'pruning':
        node_list = get_FrontierFrontier(sen_result=sen_result, BOP_result=energy_result, timestep_candidates=maxspike_candidates, constraint=maxspike_ratio * sum(energy_result[disred_maxspike-1]))
        best_node = min(node_list, key=lambda node: node.profit)

        bs = []
        current_node = best_node
        while current_node.parent is not None:
            bs.append(current_node.timestep)
            current_node = current_node.parent

        # Reverse the bs list to match the layer order
        bs.reverse()
    return bs, []


def get_FrontierFrontier(sen_result, BOP_result, timestep_candidates, constraint=9e7):
    root = Node(cost=0, profit=0, parent=None)  # Initialize the root node
    current_list = [root]

    for layer_id in tqdm(range(len(sen_result[0]))):
        next_list = []
        for n in current_list:
            for i, timestep in enumerate(timestep_candidates):
                new_cost = n.cost + BOP_result[i][layer_id]
                new_profit = n.profit + sen_result[i][layer_id]
                new_node = Node(new_cost, new_profit, timestep=timestep, parent=n)
                next_list.append(new_node)
        
        # Sort by profit
        next_list.sort(key=lambda x: x.cost, reverse=False)

        # Prune based on profit and constraint
        pruned_list = []
        # print('constraint', constraint)
        for node in next_list:
            # print(f"Checking node with cost: {node.cost}, profit: {node.profit}")  # Debugging print
            if (len(pruned_list) == 0 or pruned_list[-1].profit > node.profit) and node.cost <= constraint:
                    pruned_list.append(node)
        current_list = pruned_list
    return current_list

class Node:
    def __init__(self, cost=0, profit=0, timestep=None, parent=None):
        self.parent = parent
        self.timestep = timestep
        self.cost = cost
        self.profit = profit

    def __str__(self):
        return 'cost: {:.2f} profit: {:.2f} timestep: {}'.format(self.cost, self.profit, self.timestep)

    def __repr__(self):
        return self.__str__()


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    elif reduction == 'channel_split':
        return (pred - tgt).abs().pow(p).sum((0, 2, 3))
    else:
        return (pred - tgt).abs().pow(p).mean()
    
def kl_divergence(P, Q):
    return (P * (P / Q).log()).sum() / P.size(0) # batch size
    # F.kl_div(Q.log(), P, None, None, 'sum')

def symmetric_kl(P, Q):
    return (kl_divergence(P, Q) + kl_divergence(Q, P)) / 2


def random_sample(sen_result, BOP_result, timestep_candidates):
    random_code = [random.randint(0,len(timestep_candidates)-1) for _ in range(len(sen_result[0]))]
    sen = 0
    size = 0
    for i, t in enumerate(random_code):
        sen += sen_result[t][i]
        size += BOP_result[t][i]
    return size, sen



@torch.no_grad()
def sensitivity_anylysis_threshold(train_loader: torch.utils.data.DataLoader,
                           model: SpikeModel,
                           sim_length: int = 8,
                           maxspike: int = 5,
                           threshold_ratio: float = 0.95,
                           dist_avg: bool = False,
                           method: str = 'pruning',
                           metric: str = 'accuracy'):
    # do not use train mode here (avoid bn update)
    model.set_spike_state(use_spike=False)
    model.eval()
    device = next(model.parameters()).device
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device=device)
        target = target.to(device=device)
        gt_output = model(input)
        base_acc = compute_accuracy(gt_output, target)
        gt_output = F.softmax(gt_output, dim=1)
        break
    spike_module_layers = []
    for m in model.modules():
        if isinstance(m, SpikeModule):
            spike_module_layers.append(m)

    #adopted threshold ranges
    threshold_ratio_candidates = torch.arange(1, 10.1, 0.5).tolist()
    sen_result = [[0 for _ in range(len(spike_module_layers))] for _ in range(len(threshold_ratio_candidates))]
    Energy_result = [[0 for _ in range(len(spike_module_layers))] for _ in range(len(threshold_ratio_candidates))]
    for i in range(len(spike_module_layers)):
        original_threshold = spike_module_layers[i].threshold
        for j, ratio in enumerate(threshold_ratio_candidates):
            model.set_spike_state(use_spike=True)
            spike_module_layers[i].spike_counter = 0
            spike_module_layers[i].threshold = original_threshold * ratio
            temp_output = model(input)
            temp_acc = compute_accuracy(temp_output, target)
            temp_output = F.softmax(temp_output, dim=1)
            kl_div = symmetric_kl(temp_output, gt_output)
            acc = base_acc - temp_acc
            if metric == 'accuracy':
                sen_result[j][i] = acc
            else:
                sen_result[j][i] = kl_div.item()
            Energy_result[j][i] = spike_module_layers[i].spike_counter
            spike_module_layers[i].threshold = original_threshold
            spike_module_layers[i].spike_counter = 0

    if method == 'pruning':
        node_list = get_FrontierFrontier(sen_result=Energy_result, BOP_result=sen_result, timestep_candidates=threshold_ratio_candidates, constraint=threshold_ratio * sum(sen_result[0]))
        best_node = min(node_list, key=lambda node: node.profit)

        ssr = []
        current_node = best_node
        while current_node.parent is not None:
            ssr.append(current_node.timestep)
            current_node = current_node.parent
    return ssr, []


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
        link.allreduce(tensor)
        self.update(tensor.item(), num=num)

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

class Energy:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.energy = torch.tensor(0.0, device=device)
        self.energy_meter = AverageMeter()
        self.flops = 0
        self.samples_processed = 0
        self.sparsity_meters = {}
        self.first_conv_encountered = False  # 标记是否遇到第一个Conv2d层
        self.current_threshold = None

    def calculate_flops(self, layer, inputs, outputs):
        if isinstance(layer.module, nn.Conv2d):
            in_channels = layer.module.in_channels
            out_channels = layer.module.out_channels
            kernel_size = torch.prod(torch.tensor(layer.module.kernel_size)).item()
            output_size = torch.prod(torch.tensor(outputs.size()[-2:])).item()
            flops = in_channels * out_channels * kernel_size * output_size
        elif isinstance(layer.module, nn.Linear):
            in_features = layer.module.in_features
            out_features = layer.module.out_features
            flops = in_features * out_features
        else:
            flops = 0
        self.flops += flops
        return flops

    def update_sparsity(self, layer_name, density):
        if layer_name not in self.sparsity_meters:
            self.sparsity_meters[layer_name] = AverageMeter()
        self.sparsity_meters[layer_name].update(density)

    def get_artificial_energy(self, layer, inputs, outputs):
        FLOPs = self.calculate_flops(layer, inputs, outputs)
        energy = 4.6 * FLOPs  
        self.energy += energy / (10 ** 9)
        self.energy_meter.update(energy / (10 ** 9))

    def get_spike_energy(self, threshold):
        def hook(layer, inputs, outputs):
            FLOPs = self.calculate_flops(layer, inputs, outputs)
            density = torch.count_nonzero(inputs[0]).item() / inputs[0].numel()
            energy = 0.9 * FLOPs * density
            self.energy += energy / (10 ** 9)
            self.energy_meter.update(energy / (10 ** 9))
            layer_name = layer.__class__.__name__
            self.update_sparsity(layer_name, density)
        return hook

    def register_hooks(self):
        previous_layer_threshold = None
        for name, module in self.model.named_modules():
            if isinstance(module, SpikeModule):
                # print(self.first_conv_encountered)
                if not self.first_conv_encountered:
                    self.first_conv_encountered = True
                    current_threshold = module.threshold
                    print(f"{name} registered for artificial energy calculation!")
                    module.register_forward_hook(self.get_artificial_energy)

                else:
                    current_threshold = module.threshold
                    hook = self.get_spike_energy(previous_layer_threshold)
                    module.register_forward_hook(hook)
                    # module.register_forward_hook(self.get_spike_energy)
                    print(f"{name} registered for spike energy calculation!")
                previous_layer_threshold = current_threshold

    def print_sparsity(self):
        for name, meter in self.sparsity_meters.items():
            print(f"Layer {name}: Average Sparsity = {meter.avg}")
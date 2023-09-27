import torch
from distributed_utils.dist_helper import allaverage, allreduce

from models.spiking_layer import SpikeModel, SpikeModule, lp_loss


def bias_corr_model(model: SpikeModel, train_loader: torch.utils.data.DataLoader, correct_mempot: bool = False,
                    dist_avg: bool = False):
    """
    This function corrects the bias in SNN, by matching the
    activation expectation in some training set samples.
    Here we only sample one batch of the training set。

    :param model: SpikeModel that need to be corrected with bias
    :param train_loader: Training images
    :param correct_mempot: if True, the correct the initial membrane potential
    :param dist_avg: if True, then average the tensor between distributed GPUs
    :return: SpikeModel with corrected bias
    """
    device = next(model.parameters()).device
    for (input, target) in train_loader:
        input = input.to(device=device)
        # begin bias correction layer-by-layer
        for name, module in model.named_modules():
            if isinstance(module, SpikeModule):
                emp_bias_corr(model, module, input, correct_mempot, dist_avg)
        break



def emp_bias_corr(model: SpikeModel, module: SpikeModule, train_data: torch.Tensor, correct_mempot: bool = False,
                  dist_avg: bool = False):
    """
    Empirical Bias Correction for a single layer.
    Note that the original output must be clipped at 0 to stay non-negative.

    :param model:
    :param module:
    :param train_data:
    """
    # compute the original output
    model.set_spike_state(use_spike=False)
    get_out = GetLayerInputOutput(model, module)
    org_out = get_out(train_data)[1]
    # # clip output here
    # org_out = org_out
    # compute the SNN output
    model.set_spike_state(use_spike=True)
    get_out.data_saver.reset()
    snn_out = get_out(train_data)[1]
    # divide the snn output by T
    snn_out = snn_out / model.T
    if not correct_mempot:
        # calculate the bias
        org_mean = org_out.mean(3).mean(2).mean(0).detach() if len(org_out.shape) == 4 else org_out.mean(0).detach()
        snn_mean = snn_out.mean(3).mean(2).mean(0).detach() if len(snn_out.shape) == 4 else snn_out.mean(0).detach()
        bias = (snn_mean - org_mean).data.detach()

        if dist_avg:
            allaverage(bias)
        if module.bias is None:
            module.bias = - bias
        else:
            module.bias.data = module.bias.data - bias
    else:
        # calculate the mean along the batch dimension
        org_mean, snn_mean = org_out.mean(0, keepdim=True), snn_out.mean(0, keepdim=True)
        pot_init_temp = ((org_mean - snn_mean) * model.T).data.detach()
        if dist_avg:
            allaverage(pot_init_temp)
        module.mem_pot_init = pot_init_temp


class ActivationSaverHook:
    """
    This hook can save output of a layer.
    Note that we have to accumulate T times of the output
    if the model spike state is TRUE.
    """

    def __init__(self):
        self.stored_output = None
        self.stored_input = None
        self.stored_residual = None

    def __call__(self, module, input_batch, output_batch):
        if self.stored_output is None:
            self.stored_output = output_batch
        else:
            self.stored_output = output_batch + self.stored_output
        if self.stored_input is None:
            self.stored_input = input_batch[0]
        else:
            self.stored_input = input_batch[0] + self.stored_input
        if len(input_batch) == 2:
            if self.stored_residual is None:
                self.stored_residual = input_batch[1].detach()
            else:
                self.stored_residual = input_batch[1].detach() + self.stored_residual
        else:
            if self.stored_residual is None:
                self.stored_residual = 0

    def reset(self):
        self.stored_output = None
        self.stored_input = None
        self.stored_residual = None


class GetLayerInputOutput:
    def __init__(self, model: SpikeModel, target_module: SpikeModule):
        self.model = model
        self.module = target_module
        self.data_saver = ActivationSaverHook()

    @torch.no_grad()
    def __call__(self, input):
        # do not use train mode here (avoid bn update)
        self.model.eval()
        h = self.module.register_forward_hook(self.data_saver)
        # note that we do not have to check model spike state here,
        # because the SpikeModel forward function can already do this
        _ = self.model(input)
        h.remove()
        return self.data_saver.stored_input.detach(), self.data_saver.stored_output.detach(), \
            self.data_saver.stored_residual
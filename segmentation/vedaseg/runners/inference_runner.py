import torch
torch.backends.cudnn.enabled = False
from ..models import build_model
from ..utils import load_checkpoint
from .base import Common
from ..models.encoders.backbones.fold_bn import search_fold_and_remove_bn
from ..models.encoders.backbones.calibration import bias_corr_model
from ..models.encoders.backbones.spiking_layer import SpikeModule
from ..models.encoders.backbones.fold_bn import search_fold_and_remove_bn
from ..models.encoders.backbones.resnet import res_spcials
from ..models.encoders.backbones.spiking_layer import SpikeModel, get_maximum_activation, sensitivity_anylysis
from tqdm import tqdm

class InferenceRunner(Common):
    def __init__(self, train_cfg, test_cfg, inference_cfg, base_cfg=None):
        inference_cfg = inference_cfg.copy()

        base_cfg = {} if base_cfg is None else base_cfg.copy()

        super().__init__(base_cfg)
        self.test_dataloader = self._build_dataloader(test_cfg['data'])
        self.train_dataloader = self._build_dataloader(train_cfg['data']['train'])

        self.multi_label = inference_cfg.get('multi_label', False)

        # build inference transform
        self.transform = self._build_transform(inference_cfg['transforms'])

        # build model
        self.model = self._build_model(inference_cfg['model'])

    def load_checkpoint(self, filename, map_location='default', strict=True):
        self.logger.info('Load checkpoint from {}'.format(filename))

        if map_location == 'default':
            if self.use_gpu:
                device_id = torch.cuda.current_device()
                map_location = lambda storage, loc: storage.cuda(device_id)
            else:
                map_location = 'cpu'

        load_checkpoint(self.model, filename, map_location, strict)
        self.model.eval()

        if self.snn:
            self.desired_maxspike = self.maxspike
            self.maxspike = self.maxspike * 2
            search_fold_and_remove_bn(self.model)
            snn = SpikeModel(model=self.model, sim_length=self.timestep,
                        specials=res_spcials, maxspike=self.maxspike)
            # print(snn)
            get_maximum_activation(self.train_dataloader, model=snn, momentum=0.9, iters=5, mse=True, percentile=None, maxspike=self.maxspike,
                            sim_length=self.timestep, channel_wise=True, dist_avg=False)
            bias_corr_model(model=snn, train_loader=self.train_dataloader, correct_mempot=False)

            if self.search:
                optimal_maxspike_list, node_list = sensitivity_anylysis(self.train_dataloader, model=snn, maxspike=self.maxspike, sim_length=self.timestep, disred_maxspike=self.desired_maxspike)
                print(f"Timesteps per layer: {optimal_maxspike_list}")
                for m in snn.modules():
                    if isinstance(m, SpikeModule):
                        m.maxspike = optimal_maxspike_list[index]
                        # m.maxspike = 2
                        index += 1
                get_maximum_activation(self.train_dataloader, model=snn, momentum=0.9, iters=5, mse=True, percentile=None, maxspike=self.maxspike,
                                sim_length=self.timestep, channel_wise=True, dist_avg=False)
                bias_corr_model(model=snn, train_loader=self.train_dataloader, correct_mempot=False)
            snn.set_spike_state(use_spike=True)
            self.model = snn

    def _build_model(self, cfg):
        self.logger.info('Build model')

        model = build_model(cfg)
        # print(model)



        if torch.cuda.is_available():
            if self.distribute:
                model = torch.nn.parallel.DistributedDataParallel(
                    model.cuda(),
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=True,
                )
                self.logger.info('Using distributed training')
            else:
                if torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model)
                model.cuda()
        return model

    def compute(self, output):
        if self.multi_label:
            output = output.sigmoid()
            output = torch.where(output >= 0.5,
                                 torch.full_like(output, 1),
                                 torch.full_like(output, 0)).long()

        else:
            output = output.softmax(dim=1)
            _, output = torch.max(output, dim=1)
        return output

    def __call__(self, image, masks):
        with torch.no_grad():
            image = self.transform(image=image, masks=masks)['image']
            image = image.unsqueeze(0)

            if self.use_gpu:
                image = image.cuda()

            output = self.model(image)
            output = self.compute(output)

            output = output.squeeze().cpu().numpy()

        return output

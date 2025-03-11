from __future__ import division

import os
import random
import argparse
import time
import cv2
import numpy as np
# from copy import deepcopy
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.voc0712 import VOCDetection
from data.coco2017 import COCODataset
from data import config
from data import BaseTransform, detection_collate

import tools

from utils import distributed_utils
# from utils.com_paras_flops import FLOPs_and_Params
from utils.augmentations import SSDAugmentation, ColorAugmentation
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.vocapi_evaluator import VOCAPIEvaluator
from utils.modules import ModelEMA
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('-bs', '--batch_size', default=16, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='visualize target.')
    parser.add_argument('-v', '--version', default='yolov2_tiny',
                            help='yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_tiny')
    
    # dataset
    parser.add_argument('-root', '--data_root', default='/home/dataset/',
                        help='dataset root')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    
    # train trick
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup') 
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')      
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='use mosaic augmentation')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema training trick')

    # DDP train
    parser.add_argument("--local_rank", default=0, type=int)    
    
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    #quantization training
    parser.add_argument('--init', help='initialize form pre-trained floating point model', type=str, default='')
    return parser.parse_args()

def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    
    # set distributed
    args.distributed = True if 'WORLD_SIZE' in os.environ else False
    
    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))
    
    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model_name = args.version
    print('Model: ', model_name)
    
    if args.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        args.world_size = int(os.environ['WORLD_SIZE'])
        print('Local rank: {}, World size: {}'.format(local_rank, args.world_size))
        
    if local_rank == 0:
        wandb.init(project='your_proj', entity='your_entity', name='yolov2_r50_coco', config=args)

    # load model and config file
    if model_name == 'yolov2_d19':
        from models.yolov2_d19 import YOLOv2D19 as yolo_net
        cfg = config.yolov2_d19_cfg
    elif model_name == 'yolov2_tiny':
        from models.yolov2_tiny import YOLOv2tiny as yolo_net
        cfg = config.yolov2_tiny_cfg
        
    elif model_name == 'yolov2_r34':
        from models.yolov2_r34 import YOLOv2R34 as yolo_net
        cfg = config.yolov2_r50_cfg
        
    elif model_name == 'yolov2_r50':
        from models.yolov2_r50 import YOLOv2R50 as yolo_net
        cfg = config.yolov2_r50_cfg

    elif model_name == 'yolov3':
        from models.yolov3 import YOLOv3 as yolo_net
        cfg = config.yolov3_d53_cfg

    elif model_name == 'yolov3_spp':
        from models.yolov3_spp import YOLOv3Spp as yolo_net
        cfg = config.yolov3_d53_cfg
        
    else:
        print('Unknown model name...')
        exit(0)

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)
    
    # multi-scale
    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = cfg['train_size']
        val_size = cfg['val_size']
    else:
        train_size = val_size = cfg['train_size']

    # Model ENA
    if args.ema:
        print('use EMA trick ...')

    # dataset and evaluator
    if args.dataset == 'voc':
        data_dir = os.path.join(args.data_root, 'VOCdevkit')
        num_classes = 20
        dataset = VOCDetection(data_dir=data_dir, 
                                transform=SSDAugmentation(train_size))

        evaluator = VOCAPIEvaluator(data_root=data_dir,
                                    img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size))

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.data_root, 'MSCOCO2017')
        num_classes = 80
        dataset = COCODataset(
                    data_dir=data_dir,
                    transform=SSDAugmentation(train_size))

        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=BaseTransform(val_size))
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)
    
    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # build model
    anchor_size = cfg['anchor_size_voc'] if args.dataset == 'voc' else cfg['anchor_size_coco']
    if model_name == 'yolov2_tiny' or model_name == 'yolov2_r34' or model_name == 'yolov2_r50':
         net = yolo_net(device=device, 
                       input_size=train_size, 
                       num_classes=num_classes, 
                       trainable=True, 
                       anchor_size=anchor_size)       
    else:
        net = yolo_net(device=device, 
                       input_size=train_size, 
                       num_classes=num_classes, 
                       trainable=True, 
                       anchor_size=anchor_size)


    if args.init:
        state_dict = torch.load(args.init, map_location=device)
        # print(state_dict.keys())
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'backbone' not in k:
                name = 'backbone.' + k  # add 'backbone.' prefix
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        # print(new_state_dict.keys())
        net.load_state_dict(new_state_dict, strict=True)

    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        net.load_state_dict(torch.load(args.resume, map_location=device))

    model = net
    model = model.to(device).train()

    # SyncBatchNorm
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP
    # model_without_ddp = model
    if args.distributed:
        model = DDP(model)
        # model_without_ddp = model.module
        
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()
        
    # dataloader
    # batch_size = args.batch_size * args.world_size
    batch_size = args.batch_size
    
    if args.distributed and args.world_size > 1:
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        batch_size=batch_size, 
                        collate_fn=detection_collate,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True,
                        sampler=torch.utils.data.distributed.DistributedSampler(dataset)
                        )

    else:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        shuffle=True,
                        batch_size=batch_size, 
                        collate_fn=detection_collate,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True
                        )

    # EMA
    ema = ModelEMA(model) if args.ema else None

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)
    
    # optimizer setup
    # base_lr = (args.lr / 16) * batch_size
    
    base_lr = args.lr
    
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), 
                            lr=base_lr, 
                            momentum=args.momentum,
                            weight_decay=args.weight_decay
                            )
    
    max_epoch = cfg['max_epoch']
    epoch_size = len(dataloader)
    best_map = -1.
    warmup = not args.no_warmup

    t0 = time.time()
    # start training loop
    for epoch in range(args.start_epoch, max_epoch):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)        

        # use step lr
        if epoch in cfg['lr_epoch']:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)
    
        for iter_i, (images, targets) in enumerate(dataloader):
            # WarmUp strategy for learning rate
            ni = iter_i + epoch * epoch_size
            # warmup
            if epoch < args.wp_epoch and warmup:
                nw = args.wp_epoch * epoch_size
                tmp_lr = base_lr * pow(ni / nw, 4)
                set_lr(optimizer, tmp_lr)

            elif epoch == args.wp_epoch and iter_i == 0 and warmup:
                # warmup is over
                warmup = False
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                r = cfg['random_size_range']
                train_size = random.randint(r[0], r[1]) * 32
                model.module.set_grid(train_size) if args.distributed else model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            targets = [label.tolist() for label in targets]
            # visualize labels
            if args.vis:
                vis_data(images, targets, train_size)
                continue

            # label assignment
            if model_name in ['yolov2_d19', 'yolov2_r50', 'yolov2_r34','yolov2_tiny']:
                targets = tools.gt_creator(input_size=train_size, 
                                           stride=net.stride, 
                                           label_lists=targets, 
                                           anchor_size=anchor_size
                                           )
            else:
                targets = tools.multi_gt_creator(input_size=train_size, 
                                                 strides=net.stride, 
                                                 label_lists=targets, 
                                                 anchor_size=anchor_size
                                                 )

            # to device
            images = images.float().to(device)
            targets = torch.tensor(targets).float().to(device)

            # forward
            conf_loss, cls_loss, box_loss, iou_loss = model(images, target=targets)

            # compute loss
            total_loss = conf_loss + cls_loss + box_loss + iou_loss

            loss_dict = dict(conf_loss=conf_loss,
                             cls_loss=cls_loss,
                             box_loss=box_loss,
                             iou_loss=iou_loss,
                             total_loss=total_loss
                            )

            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # check NAN for loss
            if torch.isnan(total_loss):
                print('loss is nan !!')
                continue

            # backprop
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

            # ema
            if args.ema:
                ema.update(model)

            # display
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    tblogger.add_scalar('conf loss',  loss_dict_reduced['conf_loss'].item(),  iter_i + epoch * epoch_size)
                    tblogger.add_scalar('cls loss',  loss_dict_reduced['cls_loss'].item(),  iter_i + epoch * epoch_size)
                    tblogger.add_scalar('box loss',  loss_dict_reduced['box_loss'].item(),  iter_i + epoch * epoch_size)
                    tblogger.add_scalar('iou loss',  loss_dict_reduced['iou_loss'].item(),  iter_i + epoch * epoch_size)
                
                t1 = time.time()
                cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
                
                # basic infor
                log =  '[Epoch: {}/{}]'.format(epoch+1, max_epoch)
                log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
                log += '[lr: {:.6f}]'.format(cur_lr[0])
                # loss infor
                for k in loss_dict_reduced.keys():
                    log += '[{}: {:.2f}]'.format(k, loss_dict[k])

                # other infor
                log += '[time: {:.2f}]'.format(t1 - t0)
                log += '[size: {}]'.format(train_size)
                
                print(log, flush=True)
                
                t0 = time.time()


        # evaluation
        if epoch > 0 and (epoch + 1) % args.eval_epoch == 0 or (epoch + 1) == max_epoch:
        # if (epoch  % args.eval_epoch) == 0 or (epoch == max_epoch - 1):
            if args.ema:
                model_eval = ema.ema.module if args.distributed else ema.ema
            else:
                model_eval = model.module if args.distributed else model

            # check evaluator
            if evaluator is None:
                print('No evaluator ... save model and go on training.')
                print('Saving state, epoch: {}'.format(epoch + 1))
                weight_name = '{}_epoch_{}.pth'.format(args.version, epoch + 1)
                checkpoint_path = os.path.join(path_to_save, weight_name)
                torch.save(model_eval.state_dict(), checkpoint_path)                      
        
            else:
                print('eval ...')
                # set eval mode
                model_eval.trainable = False
                model_eval.set_grid(val_size)
                model_eval.eval()
                
                if local_rank == 0:
                    # evaluate
                    evaluator.evaluate(model_eval)
    
                    cur_map = evaluator.map
                    if cur_map > best_map:
                        # update best-map
                        best_map = cur_map
                        # save model
                        print('Saving state, epoch:', epoch + 1)
                        weight_name = '{}_epoch_{}_{:.2f}.pth'.format(args.version, epoch + 1, best_map*100)
    
                        checkpoint_path = os.path.join(path_to_save, weight_name)
                        torch.save(model_eval.state_dict(), checkpoint_path)  
    
                    if args.tfboard:
                        if args.dataset == 'voc':
                            tblogger.add_scalar('07test/mAP', evaluator.map, epoch)
                        elif args.dataset == 'coco':
                            tblogger.add_scalar('val/AP50_95', evaluator.ap50_95, epoch)
                            tblogger.add_scalar('val/AP50', evaluator.ap50, epoch)
                    if args.dataset == 'voc':
                        wandb.log({'07test/mAP': evaluator.map}, step=epoch)
                    elif args.dataset == 'coco':
                        wandb.log({'val/AP50_95': evaluator.ap50_95}, step=epoch)
                        wandb.log({'val/AP50': evaluator.ap50}, step=epoch)
                # set train mode.
                model_eval.trainable = True
                model_eval.set_grid(train_size)
                model_eval.train()

            # wait for all processes to synchronize
            if args.distributed:
                dist.barrier()

    if args.tfboard:
        tblogger.close()


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def vis_data(images, targets, input_size):
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img = images[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    img = ((img * std + mean)*255).astype(np.uint8)
    img = img.copy()

    for box in targets[0]:
        xmin, ymin, xmax, ymax = box[:-1]
        xmin *= input_size
        ymin *= input_size
        xmax *= input_size
        ymax *= input_size
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    train()

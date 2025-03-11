import argparse
import os
import numpy as np
import cv2
import time
import torch
from data.coco2017 import coco_class_index, coco_class_labels
from data import config, BaseTransform


# coco_class_labels = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#     'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
#     'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#     'train', 'tvmonitor')

# coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  


# class_colors = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
#             [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
#             [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
#             [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
#             [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


class_colors = [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
            [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
            [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
            [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
            [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
            [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
            [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
            [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
            [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
            [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
            [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
            [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
            [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
            [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
            [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
            [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
            [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
            [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
            [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
            [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
            [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
            [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
            [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
            [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
            [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
            [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
            [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
            [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
            [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
            [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
            [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
            [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
            [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
            [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
            [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
            [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
            [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
            [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
            [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
            [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
            [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
            [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
            [64, 192, 96], [64, 160, 64], [64, 64, 0]]



def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Demo Detection')
    # basic
    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from image, video or camera')
    parser.add_argument('-size', '--input_size', default=416, type=int,
                        help='input_size')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    parser.add_argument('--path_to_img', default='images/',
                        type=str, help='The path to image files')
    parser.add_argument('--path_to_vid', default='data/demo/videos/',
                        type=str, help='The path to video files')
    parser.add_argument('--path_to_save', default='det_results/',
                        type=str, help='The path to save the detection results')
    parser.add_argument('-vs', '--visual_threshold', default=0.3,
                        type=float, help='visual threshold')
    # model
    parser.add_argument('-v', '--version', default='yolo_v2',
                        help='yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_tiny')
    parser.add_argument('--conf_thresh', default=0.1, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.45, type=float,
                        help='NMS threshold')
    parser.add_argument('--init', default='weights/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--bit', default=32, type=int, help='the bit-width of the quantized network')
    parser.add_argument('--spike', action='store_true', default=False,
                        help='Use SNN')    
    parser.add_argument('--voc', action='store_true', default=False, help='Use VOC Classes')
    return parser.parse_args()
                    

def plot_bbox_labels(img, bbox, label, cls_color, test_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    # plot title bbox
    cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * test_scale), y1), cls_color, -1)
    # put the test on the title bbox
    cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, test_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, bboxes, scores, cls_inds, class_colors, vis_thresh=0.3):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_color = class_colors[int(cls_inds[i])]
            cls_id = coco_class_index[int(cls_inds[i])]
            mess = '%s: %.2f' % (coco_class_labels[cls_id], scores[i])
            # print(mess)
            # print(cls_color)
            img = plot_bbox_labels(img, bbox, mess, cls_color, test_scale=ts)

    return img


def detect(net, 
           device, 
           transform, 
           vis_thresh, 
           mode='image', 
           path_to_img=None, 
           path_to_vid=None, 
           path_to_save=None):
    # class color
    # class_colors = [(np.random.randint(255),
    #                  np.random.randint(255),
    #                  np.random.randint(255)) for _ in range(80)]
    save_path = os.path.join(path_to_save, mode)
    os.makedirs(save_path, exist_ok=True)

    # ------------------------- Camera ----------------------------
    if mode == 'camera':
        print('use camera !!!')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if ret:
                if cv2.waitKey(1) == ord('q'):
                    break
                img_h, img_w = frame.shape[:2]
                scale = np.array([[img_w, img_h, img_w, img_h]])

                # prepare
                x = torch.from_numpy(transform(frame)[0][:, :, ::-1]).permute(2, 0, 1)
                x = x.unsqueeze(0).to(device)
                # inference
                t0 = time.time()
                bboxes, scores, cls_inds = net(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")

                # rescale
                bboxes *= scale

                frame_processed = visualize(img=frame, 
                                            bboxes=bboxes,
                                            scores=scores, 
                                            cls_inds=cls_inds,
                                            class_colors=class_colors,
                                            vis_thresh=vis_thresh)
                cv2.imshow('detection result', frame_processed)
                cv2.waitKey(1)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    # ------------------------- Image ----------------------------
    elif mode == 'image':
        for i, img_id in enumerate(os.listdir(path_to_img)):
            img = cv2.imread(path_to_img + '/' + img_id, cv2.IMREAD_COLOR)
            img_h, img_w = img.shape[:2]
            scale = np.array([[img_w, img_h, img_w, img_h]])
            
            # prepare
            x = torch.from_numpy(transform(img)[0][:, :, ::-1].copy()).permute(2, 0, 1)

            x = x.unsqueeze(0).to(device)
            # inference
            t0 = time.time()
            bboxes, scores, cls_inds = net(x)
            t1 = time.time()
            print("detection time used ", t1-t0, "s")

            # rescale
            bboxes *= scale

            img_processed = visualize(img=img, 
                                    bboxes=bboxes,
                                    scores=scores, 
                                    cls_inds=cls_inds,
                                    class_colors=class_colors,
                                    vis_thresh=vis_thresh)

            # cv2.imshow('detection', img_processed)
            cv2.imwrite(os.path.join(save_path, img_id), img_processed)
            # cv2.imwrite(os.path.join(save_path, str(i).zfill(6)+'.jpg'), img_processed)
            # cv2.waitKey(0)

    # ------------------------- Video ---------------------------
    elif mode == 'video':
        video = cv2.VideoCapture(path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        save_path = os.path.join(save_path, 'det.avi')
        fps = 15.0
        out = cv2.VideoWriter(save_path, fourcc, fps, save_size)

        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------
                img_h, img_w = frame.shape[:2]
                scale = np.array([[img_w, img_h, img_w, img_h]])
                # prepare
                x = torch.from_numpy(transform(frame)[0][:, :, ::-1]).permute(2, 0, 1)
                x = x.unsqueeze(0).to(device)
                # inference
                t0 = time.time()
                bboxes, scores, cls_inds = net(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")

                # rescale
                bboxes *= scale
                
                frame_processed = visualize(img=frame, 
                                            bboxes=bboxes,
                                            scores=scores, 
                                            cls_inds=cls_inds,
                                            class_colors=class_colors,
                                            vis_thresh=vis_thresh)

                frame_processed_resize = cv2.resize(frame_processed, save_size)
                out.write(frame_processed_resize)
                cv2.imshow('detection', frame_processed)
                cv2.waitKey(1)
            else:
                break
        video.release()
        out.release()
        cv2.destroyAllWindows()


def run():
    args = parse_args()

    # use cuda
    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # model
    model_name = args.version
    print('Model: ', model_name)

    # load model and config file
    if model_name == 'yolov2_d19':
        from models.yolov2_d19 import YOLOv2D19 as yolo_net
        cfg = config.yolov2_d19_cfg

    elif model_name == 'yolov2_r50':
        from models.yolov2_r50 import YOLOv2R50 as yolo_net
        cfg = config.yolov2_r50_cfg

    elif model_name == 'yolov2_slim':
        from models.yolov2_slim import YOLOv2Slim as yolo_net
        cfg = config.yolov2_slim_cfg

    elif model_name == 'yolov2_r34':
        from models.yolov2_r34 import YOLOv2R34 as yolo_net
        cfg = config.yolov2_r50_cfg

    elif model_name == 'yolov2_tiny':
        from models.yolov2_tiny import YOLOv2tiny as yolo_net
        cfg = config.yolov2_tiny_cfg


    elif model_name == 'yolov3':
        from models.yolov3 import YOLOv3 as yolo_net
        cfg = config.yolov3_d53_cfg

    elif model_name == 'yolov3_spp':
        from models.yolov3_spp import YOLOv3Spp as yolo_net
        cfg = config.yolov3_d53_cfg

    elif model_name == 'yolov3_tiny':
        from models.yolov3_tiny import YOLOv3tiny as yolo_net
        cfg = config.yolov3_tiny_cfg
    else:
        print('Unknown model name...')
        exit(0)

    # input size
    input_size = args.input_size
    
    # build model
    if args.voc:
        anchor_size = cfg['anchor_size_voc']
        num_classes = 20
    else:
        anchor_size = cfg['anchor_size_coco']
        num_classes = 80
        
    if model_name == 'yolov2_tiny' or model_name == 'yolov2_r34':
        net = yolo_net(device=device, 
                       input_size=input_size, 
                       num_classes=num_classes, 
                       trainable=False, 
                       conf_thresh=args.conf_thresh,
                       nms_thresh=args.nms_thresh,
                       anchor_size=anchor_size,
                       bit=args.bit,
                       spike=args.spike)
    else:
        net = yolo_net(device=device, 
                       input_size=input_size, 
                       num_classes=80, 
                       trainable=False, 
                       conf_thresh=args.conf_thresh,
                       nms_thresh=args.nms_thresh,
                       anchor_size=anchor_size)

    # load weight
    net.load_state_dict(torch.load(args.init, map_location=device))
    net.to(device).eval()
    print('Finished loading model!')

    # run
    detect(net=net, 
            device=device,
            transform=BaseTransform(input_size),
            mode=args.mode,
            path_to_img=args.path_to_img,
            path_to_vid=args.path_to_vid,
            path_to_save=args.path_to_save,
            vis_thresh=args.visual_threshold
            )


if __name__ == '__main__':
    run()

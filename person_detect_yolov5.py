#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/1/18
# @Author : zengwb

import argparse
import os
import time
import platform
import shutil
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized


def set_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--source', type=str, default='/media/zengwb/PC/Dataset/ReID-dataset/channel1/1.mp4',
    #                     help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    # parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')

    parser.add_argument('--view-img', default=True, help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')


    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')

    return parser.parse_args()


def bbox_r(width, height, *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


class Person_detect():
    def __init__(self, opt, source):

        # Initialize
        self.device = opt.device if torch.cuda.is_available() else 'cpu'
        self.half = self.device != 'cpu'  # half precision only supported on CUDA
        self.augment = opt.augment
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.classes = opt.classes
        self.agnostic_nms = opt.agnostic_nms
        self.webcam = opt.cam
        # Load model
        self.model = attempt_load(opt.weights, map_location=self.device)  # load FP32 model
        print('111111111111111111111111111111111111111', self.model.stride.max())
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def detect(self, path, img, im0s, vid_cap):

        half = self.device != 'cpu'  # half precision only supported on CUDA

        # print('444444444444444444444444444444444')
        # Run inference
        # print('55555555555555555555555555555')
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                   agnostic=self.agnostic_nms)

        # Process detections
        bbox_xywh = []
        confs = []
        clas = []
        xy = []
        for i, det in enumerate(pred):  # detections per image
            # if self.webcam:  # batch_size >= 1
            #     p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            # else:
            #     p, s, im0 = path, '', im0s
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    img_h, img_w, _ = im0s.shape  # get image shape
                    x_c, y_c, bbox_w, bbox_h = bbox_r(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    # if cls == opt.classes:  # detct classes id
                    if not conf.item() > 0.3:
                        continue
                    bbox_xywh.append(obj)
                    confs.append(conf.item())
                    clas.append(cls.item())
                    xy.append(xyxy)
                    # print('jjjjjjjjjjjjjjjjjjjj', confs)
        return np.array(bbox_xywh), confs, clas, xy

    
if __name__ == '__main__':
    person_detect = Person_detect(source='/media/zengwb/PC/Dataset/ReID-dataset/channel1/1.mp4')
    with torch.no_grad():
            person_detect.detect()

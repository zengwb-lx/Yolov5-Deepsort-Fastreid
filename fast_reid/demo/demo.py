# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import glob
import os
import sys
import time
import cv2
import numpy as np
import tqdm
from torch.backends import cudnn
import torch
sys.path.append('..')

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager

from predictor import FeatureExtractionDemo
# import some modules added in project like this below
# from projects.PartialReID.partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        default='../../kd-r34-r101_ibn/config-test.yaml',
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        # nargs="+",
        default='../query/a/0013_c1-2_818.jpg',
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='demo_output',
        help='path to save features'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', '../../kd-r34-r101_ibn/model_final.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser


def reid_feature():
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)
    return demo


class Reid_feature():
    def __init__(self):
        args = get_parser().parse_args()
        cfg = setup_cfg(args)
        self.demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    def __call__(self, img_list):
        t1 = time.time()
        feat = self.demo.run_on_batch(img_list)
        # print('reid time:', time.time() - t1, len(img_list))
        return feat


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    PathManager.mkdirs(args.output)
    # onnx_output = np.load('../tools/deploy/onnx_output/0065_c6s1_009501_02.npy')
    # onnx_output = np.vstack((onnx_output, onnx_output))
    # print(onnx_output.shape)

    if args.input:
        while True:
            img_list = []
        # if PathManager.isdir(args.input[0]):
        #     args.input = glob.glob(os.path.expanduser(args.input[0]))
        #     assert args.input, "The input path(s) was not found"
        # for path in tqdm.tqdm(os.listdir(args.input)):
            t1 = time.time()
            img = cv2.imread(args.input)
            im = img[:, :, ::-1]  # reid 前处理
            im = cv2.resize(im, (128, 256), interpolation=cv2.INTER_CUBIC)
            img_list.append(torch.as_tensor(im.astype("float32").transpose(2, 0, 1))[None])
            img_list.append(torch.as_tensor(im.astype("float32").transpose(2, 0, 1))[None])
            img_list.append(torch.as_tensor(im.astype("float32").transpose(2, 0, 1))[None])
            img_list.append(torch.as_tensor(im.astype("float32").transpose(2, 0, 1))[None])
            img_list.append(torch.as_tensor(im.astype("float32").transpose(2, 0, 1))[None])
            img_list.append(torch.as_tensor(im.astype("float32").transpose(2, 0, 1))[None])
            img_list.append(torch.as_tensor(im.astype("float32").transpose(2, 0, 1))[None])
            # img_list.append(img)


            # feat = demo.run_on_image(img)
            # print(img_list[0].shape)
            feat = demo.run_on_batch(img_list)
            print('pytorch time:', time.time() - t1)
            pytorch_output = feat.numpy()
            # print(pytorch_output.shape)
            # sim = np.dot(pytorch_output, onnx_output.transpose())
            # print('====sim:', sim)

        # np.testing.assert_allclose(pytorch_output, onnx_output, rtol=1e-4, atol=1e-6)
        # print("恭喜你 ^^ ，onnx 和 pytorch 结果一致 ， Exported model has been executed decimal=5 and the result looks good!")

        # np.save(os.path.join(args.output, path.replace('.jpg', '.npy').split('/')[-1]), feat)

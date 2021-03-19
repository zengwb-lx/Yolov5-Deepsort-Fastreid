import argparse
from face_feature_extract import models
from face_cluster.face_cluster_by_infomap import cluster_main
from face_feature_extract.extract_feature import extract_fature
from tools.utils import Timer
import os
import cv2
import numpy as np
import sys
sys.path.append('../demo')
sys.path.append('../')
import torch
from fastreid.evaluation import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader, build_reid_gallery_loader
from predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer
import torchvision.transforms as T


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def gallery_transform(cfg):
    transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
    ])
    return transform


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        default='./sbs_R50-ibn-model/config.yaml',
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        nargs="+",
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
        default=[],
        nargs=argparse.REMAINDER,
    )



# parser = argparse.ArgumentParser(description='Face Cluster')
    parser.add_argument('--is_cuda', default='True', type=str)

    # feature-extract config
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))


    parser.add_argument('--arch',
                        '-a',
                        metavar='ARCH',
                        default='resnet50',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=1, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--input-size', default=112, type=int)
    parser.add_argument('--feature-dim', default=2048, type=int)
    parser.add_argument('--load-path', default='pretrain_models/res50_softmax.pth.tar', type=str)
    parser.add_argument('--strict', dest='strict', action='store_true')
    parser.add_argument('--output-path', default='bin/test.bin', type=str)

    # cluster config
    parser.add_argument('--input_picture_path', default='./provate3', type=str)
    parser.add_argument('--output_picture_path', default='data/output_pictures/data_sample', type=str)
    parser.add_argument('--knn_method', default='faiss-cpu', type=str)
    parser.add_argument('--is_evaluate', default='False', type=str)
    parser.add_argument('--k', default=100, type=int)
    parser.add_argument('--min_sim', default=0.40, type=float)
    parser.add_argument('--metrics', default=['pairwise', 'bcubed', 'nmi'], type=list)
    parser.add_argument('--label_path', default='data/tmp/test.meta', type=str)
    parser.add_argument('--save_result', default='True', type=str)
    return parser

def detect(opt, logger, cfg):
    demo = FeatureExtractionDemo(cfg, parallel=opt.parallel)
    query_loader = []
    print(len(os.listdir(opt.input_picture_path)))
    for query_image in os.listdir(opt.input_picture_path):
        query_image = cv2.imread(os.path.join(opt.input_picture_path, query_image))
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)  # PIL: (233, 602)
        query_image = cv2.resize(query_image, (128, 256))
        query_image = np.transpose(query_image, (2, 0, 1))
        query_feats = demo.run_on_image(torch.from_numpy(query_image).unsqueeze(0))
        print(query_feats.shape)
        query_loader.append(query_feats)
    extract_features = torch.cat(query_loader, dim=0).data.cpu().numpy()
    # extract_features = torch.nn.functional.normalize(query_feats, dim=1, p=2).data.cpu().numpy()
    print('features:', extract_features.shape)
    with Timer('All Steps'):
        global args
        # args = parser.parse_args()
        label_path = None
        pred_label_path = None
        print('=> Use cuda ?: {}'.format(opt.is_cuda))
        # with Timer('Extract Feature'):
        #     extract_features = extract_fature(args)
        if eval(opt.is_evaluate):
            opt.label_path = 'data/tmp/test.meta'
        if not eval(opt.is_cuda):
            opt.knn_method = 'faiss-cpu'
        with Timer('Face Cluster'):
            cluster_main(opt, extract_features)
        print("=> Face cluster done! The cluster results have been saved in {}".format(opt.output_picture_path))


if __name__ == '__main__':
    # ############ 行人重识别模型初始化 #############
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt = get_parser().parse_args()
    logger = setup_logger()
    cfg = setup_cfg(opt)
    with torch.no_grad():
        detect(opt, logger, cfg)

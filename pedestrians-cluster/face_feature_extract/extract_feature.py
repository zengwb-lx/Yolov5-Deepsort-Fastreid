import argparse
import os
import numpy as np
import time
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from face_feature_extract import models
from tools.utils import AverageMeter, load_ckpt, bin_loader
from face_feature_extract.datasets import GenDataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# parser = argparse.ArgumentParser(description='Feature Extractor')
# parser.add_argument('--arch',
#                     '-a',
#                     metavar='ARCH',
#                     default='resnet50',
#                     choices=model_names,
#                     help='model architecture: ' + ' | '.join(model_names) +
#                     ' (default: resnet18)')
# parser.add_argument('-j', '--workers', default=1, type=int)
# parser.add_argument('-b', '--batch-size', default=128, type=int)
# parser.add_argument('--input-size', default=112, type=int)
# parser.add_argument('--feature-dim', default=256, type=int)
# parser.add_argument('--load-path', default='../pretrain_models/res50_softmax.pth.tar', type=str)
# parser.add_argument('--bin-file', type=str)
# parser.add_argument('--strict', dest='strict', action='store_true')
# parser.add_argument('--output-path', default='../bin/test.bin', type=str)


class IdentityMapping(nn.Module):
    def __init__(self, base):
        super(IdentityMapping, self).__init__()
        self.base = base

    def forward(self, x):
        x = self.base(x)
        return x


def extract_fature(args):
    # global args
    # args = parser.parse_args()

    # assert args.output_path.endswith('.bin')
    print("************Init the face feature extract model***************")
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](feature_dim=args.feature_dim)
    model = IdentityMapping(model)

    if eval(args.is_cuda):
        model.cuda()
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    else:
        model = torch.nn.DataParallel(model)

    if args.load_path:
        if args.strict:
            classifier_keys = ['module.logits.weight', 'module.logits.bias']
            load_ckpt(args.load_path,
                      model,
                      ignores=classifier_keys,
                      strict=True)
        else:
            load_ckpt(args.load_path, model, strict=False)

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.25, 0.25, 0.25])

    test_loader = DataLoader(GenDataset(
        args.input_picture_path,
        args.is_evaluate,
        transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.ToTensor(),
            normalize,
        ])),
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             pin_memory=True)

    print("=> Extracting face features with the input path {}......".format(args.input_picture_path))
    features = extract(test_loader, model)
    print("=> Extract face features done!")
    print("=> The total face：{}".format(str(features.shape[0])))
    assert features.shape[1] == args.feature_dim

    print("=> Save the extracted features to {}".format(args.output_path))
    folder = os.path.dirname(args.output_path)
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder)
    # features.tofile(args.output_path)
    return features
    # np.save(args.output_path, features)


def extract(test_loader, model):
    batch_time = AverageMeter(10)
    model.eval()
    features = []
    with torch.no_grad():
        end = time.time()
        for i, input in enumerate(test_loader):
            # compute output
            output = model(input)
            features.append(output.data.cpu().numpy())
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return np.vstack(features)


if __name__ == '__main__':
    features = extract_fature(input_picture_path='../data/input_pictures/data_sample')
    # get_path()


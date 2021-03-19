import time
import numpy as np
import torch
import torch.distributed as dist
import os
import io
import sys
import pickle
from PIL import Image
import multiprocessing as mp
import json


class TextColors:
    HEADER = '\033[35m'
    OKBLUE = '\033[34m'
    OKGREEN = '\033[32m'
    WARNING = '\033[33m'
    FATAL = '\033[31m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(
                self.name,
                time.time() - self.start))
        return exc_type is None

def init_processes(addr, port, gpu_num, backend):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print(rank, size)
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    torch.cuda.set_device(rank % gpu_num)
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend)
    print('initialize {} successfully (rank {})'.format(backend, rank))
    return rank, size


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length):
        self.length = length
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0

    def update(self, val):
        self.history.append(val)
        if len(self.history) > self.length:
            del self.history[0]

        self.val = self.history[-1]
        self.avg = np.mean(self.history)


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_ckpt(state, ckpt, epoch, is_best):
    folder = os.path.dirname(ckpt)
    fn = '{}_epoch_{}.pth.tar'.format(os.path.basename(ckpt), epoch)
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, fn)
    print('saving to {}'.format(path))
    torch.save(state, '{}'.format(path))
    if is_best:
        best_fn = os.path.join(folder, 'model_best.pth.tar')
        if os.path.exists(best_fn):
            os.unlink(best_fn)
        os.symlink(fn, best_fn)


def load_ckpt(path, model, ignores=[], strict=True, optimizer=None):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        if len(ignores) > 0:
            assert optimizer == None
            keys = set(checkpoint['state_dict'].keys())
            for ignore in ignores:
                if ignore in keys:
                    print('ignoring {}'.format(ignore))
                    del checkpoint['state_dict'][ignore]
                else:
                    raise ValueError(
                        'cannot find {} in load_path'.format(ignore))
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
        if not strict:
            pretrained_keys = set(checkpoint['state_dict'].keys())
            model_keys = set([k for k, _ in model.named_parameters()])
            for k in model_keys - pretrained_keys:
                print('warning: {} not loaded'.format(k))
        if optimizer != None:
            assert len(ignores) == 0
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (step {})".format(
                path, checkpoint['epoch']))
            return checkpoint['epoch'], checkpoint['best_prec1']
    else:
        assert False, "=> no checkpoint found at '{}'".format(path)


def simplify_ckpt(path, opath='', ignores=[]):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        keys = list(checkpoint.keys())
        for key in keys:
            if key == 'state_dict':
                continue
            del checkpoint[key]
        if len(ignores) >= 0:
            keys = set(checkpoint['state_dict'].keys())
            for ignore in ignores:
                if ignore in keys:
                    print('ignoring {}'.format(ignore))
                    del checkpoint['state_dict'][ignore]
                else:
                    for k in keys:
                        if k.find('base') < 0:
                            print(k, checkpoint['state_dict'][k].shape)
                    raise ValueError(
                        'cannot find {} in load_path'.format(ignore))
        if opath == '':
            opath = path + '_simplified'
        print("=> saving simplified checkpoint to '{}'".format(opath))
        torch.save(checkpoint, opath)
    else:
        assert False, "=> no checkpoint found at '{}'".format(path)


def normalize(feat, axis=1):
    if len(feat.shape) == 1:
        return feat / np.linalg.norm(feat)
    if axis == 0:
        return feat / np.linalg.norm(feat, axis=0)
    elif axis == 1:
        return feat / np.linalg.norm(feat, axis=1)[:, np.newaxis]


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
        return img


def bin_loader(path):
    '''load verification img array and label from bin file
    '''
    with open(path, 'rb') as f:
        if sys.version_info[0] == 2:
            data = pickle.load(open(path, 'rb'))
        elif sys.version_info[0] == 3:
            data = pickle.load(open(path, 'rb'), encoding='bytes')
        else:
            raise EnvironmentError('Only support python 2 or 3')
    bins, lbs = data
    assert len(bins) == 2 * len(lbs)
    imgs = [pil_loader(b) for b in bins]
    return imgs, lbs


def get_img(filepath, is_evaluate=False):
    img_list = []
    picture_path = []
    label_list = []
    label = 0
    for root, dirs, files in os.walk(filepath):
        if len(files) > 0:
            for file in files:
                path = os.path.join(root + '/' + file)
                with Image.open(path) as img:
                    img = img.convert('RGB')
                    img = img.resize((112, 112))
                    img_list.append(img)
                    label_list.append(label)
                picture_path.append(path)
            label += 1
    label_arr = np.array(label_list)
    label_path = 'data/tmp/test.meta'
    if is_evaluate:
        np.savetxt(label_path, label_arr, fmt="%d")
    a = {}
    for i in range(len(picture_path)):
        a[i] = picture_path[i]
    pic_path_tmp_path = 'data/tmp/pic_path'
    f = open(pic_path_tmp_path, 'w')
    json.dump(a, f)
    return img_list


def save_imgs(imgs, ofolder):
    '''save pil image array to JPEG image file
    '''
    for i, img in enumerate(imgs):
        opath = os.path.join(ofolder, "{}.jpg".format(i))
        if not os.path.exists(os.path.dirname(opath)):
            print(opath)
            os.makedirs(os.path.dirname(opath))
        img.save(opath, "JPEG")
    else:
        raise TypeError(
            'axis value should be 0 or 1(cannot handel axis {})'.format(axis))


def mkdir_if_no_exist(path, subdirs=['']):
    if path == '':
        return
    for sd in subdirs:
        d = os.path.dirname(os.path.join(path, sd))
        if not os.path.exists(d):
            os.makedirs(d)


def read_feat(path, inst_num, feat_dim, dtype=np.float32, verbose=False):
    assert (inst_num > 0 or inst_num == -1) and feat_dim > 0
    count = -1
    if inst_num > 0:
        count = inst_num * feat_dim
    probs = np.fromfile(path, dtype=dtype, count=count)
    if feat_dim > 1:
        probs = probs.reshape(inst_num, feat_dim)
    if verbose:
        print('[{}] shape: {}'.format(path, probs.shape))
    return probs


def write_feat(ofn, features):
    print('save features to', ofn)
    features.tofile(ofn)


def l2norm(vec):
    """
    归一化
    :param vec: 
    :return: 
    """
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr


def read_meta(fn_meta, start_pos=0, verbose=True):
    """
    idx2lb：每一个顶点对应一个类
    lb2idxs：每个类对应一个id
    """
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


def get_path():
    root_path = '/media/zengwb/PC/Dataset/ReID-dataset/channel1/provate3'
    images = os.listdir(root_path)
    picture_path = [os.path.join(root_path, image) for image in images]
    a = {}
    for i in range(len(picture_path)):
        a[i] = picture_path[i]
    f = open('./pic_path', 'w')
    json.dump(a, f)


if __name__ == '__main__':
    get_path()
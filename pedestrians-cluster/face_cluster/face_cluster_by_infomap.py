
import numpy as np
from tqdm import tqdm
import infomap
import json
import time
from multiprocessing.dummy import Pool as Threadpool
from multiprocessing import Pool
import multiprocessing as mp
import os
import shutil
from tools.utils import Timer, mkdir_if_no_exist, l2norm, intdict2ndarray, read_meta
from evaluation import evaluate, accuracy


class knn():
    def __init__(self, feats, k, index_path='', verbose=True):
        pass

    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]
        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue
            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return th_nbrs, th_dists

    def get_knns(self, th=None):
        if th is None or th <= 0.:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer('filter edges by th {} (CPU={})'.format(th, nproc),
                   self.verbose):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(
                    tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns


class knn_faiss(knn):
    """
    内积暴力循环
    归一化特征的内积等价于余弦相似度
    """
    def __init__(self,
                 feats,
                 k,
                 index_path='',
                 knn_method='faiss-cpu',
                 verbose=True):
        import faiss
        with Timer('[{}] build index {}'.format(knn_method, k), verbose):
            knn_ofn = index_path + '.npz'
            if os.path.exists(knn_ofn):
                print('[{}] read knns from {}'.format(knn_method, knn_ofn))
                self.knns = np.load(knn_ofn)['data']
            else:
                feats = feats.astype('float32')
                size, dim = feats.shape
                if knn_method == 'faiss-gpu':
                    import math
                    i = math.ceil(size/1000000)
                    if i > 1:
                        i = (i-1)*4
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(i * 1024 * 1024 * 1024)
                    index = faiss.GpuIndexFlatIP(res, dim)
                else:
                    index = faiss.IndexFlatIP(dim)
                index.add(feats)
        with Timer('[{}] query topk {}'.format(knn_method, k), verbose):
            knn_ofn = index_path + '.npz'
            if os.path.exists(knn_ofn):
                pass
            else:
                sims, nbrs = index.search(feats, k=k)
                # torch.cuda.empty_cache()
                self.knns = [(np.array(nbr, dtype=np.int32),
                              1 - np.array(sim, dtype=np.float32))
                             for nbr, sim in zip(nbrs, sims)]


def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs


# 构造边
def get_links(single, links, nbrs, dists, args):
    for i in tqdm(range(nbrs.shape[0])):
        count = 0
        for j in range(1, len(nbrs[i])):
            # 排除本身节点
            if i == nbrs[i][j]:
                pass
            elif dists[i][j] <= 1 - args.min_sim:
                count += 1
                links[(i, nbrs[i][j])] = float(1 - dists[i][j])
            else:
                break
        # 统计孤立点
        if count == 0:
            single.append(i)
    return single, links


def cluster_by_infomap(nbrs, dists, args):
    """
    基于infomap的聚类
    :param nbrs: 
    :param dists: 
    :param args: 
    :return: 
    """
    single = []
    links = {}
    with Timer('get links', verbose=True):
        single, links = get_links(single=single, links=links, nbrs=nbrs, dists=dists, args=args)

    infomapWrapper = infomap.Infomap("--two-level --directed")
    for (i, j), sim in tqdm(links.items()):
        _ = infomapWrapper.addLink(int(i), int(j), sim)

    # 聚类运算
    infomapWrapper.run()

    label2idx = {}
    idx2label = {}

    # 聚类结果统计
    for node in infomapWrapper.iterTree():
        # node.physicalId 特征向量的编号
        # node.moduleIndex() 聚类的编号
        idx2label[node.physicalId] = node.moduleIndex()
        if node.moduleIndex() not in label2idx:
            label2idx[node.moduleIndex()] = []
        label2idx[node.moduleIndex()].append(node.physicalId)

    node_count = 0
    for k, v in label2idx.items():
        if k == 0:
            node_count += len(v[2:])
            label2idx[k] = v[2:]
            # print(k, v[0:])
        else:
            node_count += len(v[1:])
            label2idx[k] = v[1:]
            # print(k, v[0:])

    # 孤立点个数
    print("=> Single cluster:{}".format(len(single)))

    keys_len = len(list(label2idx.keys()))
    # print(keys_len)

    # 孤立点放入到结果中
    for single_node in single:
        idx2label[single_node] = keys_len
        label2idx[keys_len] = [single_node]
        keys_len += 1

    print("=> Total clusters:{}".format(keys_len))

    idx_len = len(list(idx2label.keys()))
    print("=> Total nodes:{}".format(idx_len))

    # 保存聚类结果
    if eval(args.save_result):
        with open('pred_label_path.txt', 'w') as of:
            for idx in range(idx_len):
                of.write(str(idx2label[idx]) + '\n')

    # 评价聚类结果
    if eval(args.is_evaluate) and args.label_path is not None:
        pred_labels = intdict2ndarray(idx2label)
        true_lb2idxs, true_idx2lb = read_meta(args.label_path)
        gt_labels = intdict2ndarray(true_idx2lb)
        for metric in args.metrics:
            evaluate(gt_labels, pred_labels, metric)

    # 归档图片
    if args.output_picture_path is not None:
        print("=> Start copy pictures to the output path {} ......".format(args.output_picture_path))
        with open('./tools/pic_path', 'r') as f:
            content = f.read()
            picture_path_dict = json.loads(content)
        mkdir_if_no_exist(args.output_picture_path)
        shutil.rmtree(args.output_picture_path)
        os.mkdir(args.output_picture_path)
        tmp_pth = 'data/input_pictures/alldata'
        for label, idxs in label2idx.items():
            print('---------------------------', label, idxs)
            picture_reuslt_path = args.output_picture_path + '/' + str(label)
            mkdir_if_no_exist(picture_reuslt_path)
            for idx in idxs:
                picture_path = picture_path_dict[str(idx)]
                shutil.copy(picture_path, picture_reuslt_path)
                # shutil.copy(picture_path, tmp_pth)


def get_dist_nbr(features, args):
    # features = np.fromfile(feature_path, dtype=np.float32)
    features = features.reshape(-1, 2048)
    features = l2norm(features)

    index = knn_faiss(feats=features, k=args.k, knn_method=args.knn_method)
    knns = index.get_knns()
    dists, nbrs = knns2ordered_nbrs(knns)
    return dists, nbrs


def cluster_main(args, extract_features):
    dists, nbrs = get_dist_nbr(features=extract_features, args=args)
    print(dists.shape, nbrs.shape)
    cluster_by_infomap(nbrs, dists, args)


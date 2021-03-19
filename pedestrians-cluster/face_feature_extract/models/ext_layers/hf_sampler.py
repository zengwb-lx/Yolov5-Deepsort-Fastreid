import os
import sys
import time
import random
import numpy as np

import torch
from torch.autograd import Function
from torch.nn.modules.module import Module
from annoy import AnnoyIndex
from .paramclient import ParameterClient
from multiprocessing.dummy import Pool


class HFSamplerFunc(Function):
    def __init__(self,
                 client,
                 anns,
                 pool,
                 fdim,
                 sample_num,
                 num_output,
                 is_prob=False,
                 bias=False,
                 midw='0',
                 midb='1'):
        self.fdim = fdim
        self.num_output = num_output
        self.sample_num = sample_num
        self.is_prob = is_prob
        self.client = client
        self.anns = anns
        self.pool = pool
        self.midw = midw
        self.bias = bias

    def forward(self, features, labels):
        labels = labels.cpu().numpy()
        self.n_nbr = int(self.sample_num / labels.size + 1)
        self.rows, labels = self._annoy_share_mask(features, labels,
                                                   self.sample_num,
                                                   self.num_output)
        weights = self.client.get_value_by_rows(self.midw, self.rows)
        if not self.bias:
            bias = np.zeros([self.sample_num], dtype=np.float32)
        else:
            bias = self.client.get_value_by_rows(self.midb, self.rows)

        return torch.from_numpy(weights).cuda(), \
            torch.from_numpy(bias).cuda(), \
            torch.from_numpy(labels).cuda()

    def backward(self, grad_w, grad_b, grad_l):
        """ update return immediately
        """
        self.client.update_by_rows(self.midw, self.rows, grad_w.cpu().numpy())
        if self.bias:
            self.client.update_by_rows(self.midb, self.rows,
                                       grad_b.cpu().numpy())
        return None, None

    """ private functions
    """
    def _gen_idxs(self, labels):
        """ This function constructs the `relative label` inside batch.
            idx represents the index of label in the batch, while
            lbs is the set of absolute label.
        """
        lbs = set(labels)
        lbs_size = len(lbs)
        lbs = list(lbs)
        idxs = np.array([lbs.index(l) for l in labels], dtype=np.int64)
        assert idxs.shape[-1] == len(labels)
        return idxs, lbs, lbs_size

    def _norm(self, lst):
        return lst * 1.0 / lst.sum()

    def _get_nns_by_vector(self, v):
        return self.anns.get_nns_by_vector(v, self.n_nbr)

    def _annoy_thread(self, x):
        # since python is limited by GIL, more threads may lower the speed
        res = self.pool.map_async(self._get_nns_by_vector, x)
        res.wait()
        if res.ready() and res.successful():
            nbrs = [nbr for nbrs in res.get() for nbr in nbrs]
            return nbrs

    def _annoy(self, x):
        nbrs = []
        for v in x:
            nbrs.extend(self._get_nns_by_vector(v))
        return nbrs

    def _annoy_prob(self, x, sample_num):
        nbrs, probs = [], []
        for v in x:
            nbr, prob = self.anns.get_nns_by_vector(v,
                                                    self.n_nbr,
                                                    include_distances=True)
            nbrs.extend(nbr)
            probs.extend(prob)
        probs = self._norm(np.array(probs))
        nbrs = np.random.choice(nbrs, sample_num, replace=False, p=probs)
        return nbrs

    def _annoy_share_mask(self, feat, labels, sample_num, num_output):
        idxs, lbs, lbs_size = self._gen_idxs(labels)
        if not self.is_prob:
            neg_lbs = self._annoy_thread(feat)
            # the output of multi-thread and single-thread should be equal,
            # you can verify by uncommenting below centence
            # assert neg_lbs == self._annoy(feat)
        else:
            neg_lbs = self._annoy_prob(feat, sample_num)
        neg_lbs = list(set(neg_lbs).difference(set(lbs)))
        rnum = sample_num - lbs_size
        if len(neg_lbs) > rnum:
            random.shuffle(neg_lbs)
            neg_lbs = neg_lbs[:rnum]
        else:
            rneg = set(range(num_output)).difference(set(neg_lbs) | set(lbs))
            neg_lbs += random.sample(list(rneg), rnum - len(neg_lbs))
        selected_cls = np.append(np.array(lbs), np.array(neg_lbs))
        assert len(selected_cls) == sample_num, \
                "unmask size vs sample num ({} vs {})".format(len(selected_cls), sample_num)

        return selected_cls, idxs


class HFSampler(Module):
    def __init__(self,
                 rank,
                 fdim,
                 sample_num,
                 num_output,
                 bias=False,
                 ntrees=50,
                 interval=100,
                 start_iter=0,
                 midw='0',
                 midb='1'):
        super(HFSampler, self).__init__()
        self.rank = rank
        self.fdim = fdim
        self.sample_num = sample_num
        self.num_output = num_output
        self.full_cls = np.arange(self.num_output)
        # init param client
        self.client = ParameterClient(rank)
        self.midw = midw
        self.midb = midb
        self.is_bias = bias
        self.client.add_matrix(self.midw, [self.num_output, self.fdim])
        if self.is_bias:
            self.client.add_matrix(self.midb, [self.num_output, 1])
        # init hashing forest
        self.ntrees = ntrees
        self.interval = interval
        self.start_iter = start_iter
        self.iter = start_iter
        self.test_iter = start_iter
        self.anns = AnnoyIndex(self.fdim)
        self.pool = Pool(processes=2)

    def __repr__(self):
        return ('{name}({rank}, fdim={fdim}, sample_num={sample_num},'
                ' num_output={num_output})'.format(
                    name=self.__class__.__name__,
                    rank=self.rank,
                    fdim=self.fdim,
                    sample_num=self.sample_num,
                    num_output=self.num_output))

    def _update_hf(self):
        if not self.iter % self.interval == 0 and \
            not self.iter == self.start_iter:
            return
        w = self.client.get_value_by_rows(self.midw, self.full_cls)
        self.anns = AnnoyIndex(self.fdim)
        for i, v in enumerate(w):
            self.anns.add_item(i, v)
        self.anns.build(self.ntrees)

    def forward(self, features, labels):
        if self.training:
            self._update_hf()
            self.iter += 1
            return HFSamplerFunc(self.client,
                                 self.anns,
                                 self.pool,
                                 self.fdim,
                                 self.sample_num,
                                 self.num_output,
                                 bias=self.is_bias)(features, labels)
        else:
            if self.iter > self.test_iter:
                self.test_iter = self.iter
                self.weights = self.client.get_value_by_rows(
                    self.midw, self.full_cls)
                if not self.is_bias:
                    self.bias = np.zeros([self.num_output], dtype=np.float32)
                else:
                    self.bias = self.client.get_value_by_rows(
                        self.midb, self.full_cls)
            return torch.from_numpy(self.weights).cuda(), \
                   torch.from_numpy(self.bias).cuda(), \
                   labels

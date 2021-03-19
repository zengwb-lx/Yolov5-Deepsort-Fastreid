import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .ext_layers import HFSampler, HNSWSampler, Ident

__all__ = ['Classifier', 'CosFaceClassifier', 'HFClassifier', 'HNSWClassifier']


class Classifier(nn.Module):
    def __init__(self, base, feature_dim, num_classes, **kwargs):
        super(Classifier, self).__init__()
        self.base = base
        self.dropout = nn.Dropout(p=0.5)
        self.logits = nn.Linear(feature_dim, num_classes)

    def forward(self, x, label):
        # input label here maintain the same api,
        # which is actually useless
        x = self.base(x)
        x = self.dropout(x)
        x = self.logits(x)
        return x


class CosFaceClassifier(nn.Module):
    def __init__(self, base, feature_dim, num_classes, s=64, m=0.2, **kwargs):
        super(CosFaceClassifier, self).__init__()
        self.base = base
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        assert s > 0
        assert 0 < m <= 1
        self.weight = Parameter(torch.Tensor(num_classes, feature_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def __repr__(self):
        return ('feature_dim={}, num_classes={}, s={}, m={}'.format(
            self.feature_dim, self.num_classes, self.s, self.m))

    def forward(self, x, label):
        embed = self.base(x)
        n_weight = F.normalize(self.weight, p=2, dim=1)
        n_embed = F.normalize(embed, p=2, dim=1)
        out = F.linear(n_embed, n_weight)
        zero = torch.cuda.FloatTensor(out.shape).fill_(0)
        out -= zero.scatter_(1, label.view(-1, 1), self.m)
        out *= self.s
        return out


def var_hook(grad):
    # hook can be used to send grad to ParameterServer
    return grad


class HFClassifier(nn.Module):
    def __init__(self, base, rank, feature_dim, sampler_num, num_classes):
        super(HFClassifier, self).__init__()
        self.base = base
        self.dropout = nn.Dropout(p=0.5)
        self.hf_sampler = HFSampler(rank, feature_dim, sampler_num,
                                    num_classes)

    def forward(self, x, labels):
        x = self.base(x)
        x = self.dropout(x)
        w, b, labels = self.hf_sampler(x, labels)
        labels = labels.detach()
        # w.register_hook(var_hook)
        # asssert w.requires_grad == True
        x = torch.mm(x, w.t()) + b
        return x, labels


class HNSWClassifier(nn.Module):
    def __init__(self, base, rank, feature_dim, sampler_num, num_classes):
        super(HNSWClassifier, self).__init__()
        self.base = base
        self.dropout = nn.Dropout(p=0.5)
        self.hnsw_sampler = HNSWSampler(rank, feature_dim, sampler_num,
                                        num_classes)

    def forward(self, x, labels):
        x = self.base(x)
        x = self.dropout(x)
        w, b, labels = self.hnsw_sampler(x, labels)
        labels = labels.detach()
        x = torch.mm(x, w.t()) + b
        return x, labels

import math
import numpy as np
import torch
from torch.utils.data.sampler import Sampler


__all__ = ["DistSequentialSampler"]


class DistSequentialSampler(Sampler):
    def __init__(self, dataset, world_size, rank):
        assert rank >= 0
        assert dataset.num >= world_size, '{} vs {}'.format(dataset.size, world_size)
        sub_num = int(math.ceil(1. * dataset.num / world_size))
        # add extra samples to make it evenly divisible
        tot_num = sub_num * world_size
        self.dsize = dataset.num
        self.beg = sub_num * rank
        self.end = min(self.beg + sub_num, tot_num)

    def __iter__(self):
        indices = [i % self.dsize for i in range(self.beg, self.end)]
        return iter(indices)

    def __len__(self):
        return self.end - self.beg

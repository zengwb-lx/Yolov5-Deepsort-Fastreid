import torch
from torch.autograd import Function
from torch.nn.modules.module import Module


class IdentFunc(Function):
    def __init__(self):
        pass

    def forward(self, features):
        return features

    def backward(self, grad):
        return grad


class Ident(Module):
    def __init__(self):
        super(Ident, self).__init__()
        self.ident = IdentFunc()

    def __repr__(self):
        return ('{name}'.format(name=self.__class__.__name__))

    def forward(self, features):
        return IdentFunc()(features)

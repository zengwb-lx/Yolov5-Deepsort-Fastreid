from .resnet import *
from .hynet import *
from .ir import *
from .classifier import *
from .ext_layers import ParameterClient

__factory_classifier__ = {
    'linear': Classifier,
    'cosface': CosFaceClassifier,
    'hf': HFClassifier,
    'hnsw': HNSWClassifier,
}


def build_classifier(name, model, **kwargs):
    if name not in __factory_classifier__:
        raise KeyError("Unknown classifier:", name)
    return __factory_classifier__[name](model, **kwargs)

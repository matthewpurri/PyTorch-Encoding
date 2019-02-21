from .model_zoo import get_model
from .model_store import get_model_file
from .resnet import *
from .cifarresnet import *
from .base import *
from .fcn import *
from .fcn_ms import *
from .psp import *
from .encnet import *
from .deeplab import *
from .mvcnet import *


def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    # models = {
    #     'fcn': get_fcn,
    #     'psp': get_psp,
    #     'atten': get_atten,
    #     'encnet': get_encnet,
    #     'encnetv2': get_encnetv2,
    #     'deeplab': get_deeplab,
    # }
    models = {
        'fcn': get_fcn,
        'fcn_ms': get_fcn_ms,
        'psp': get_psp,
        'encnet': get_encnet,
        'deeplab': get_deeplab,
        'mvcnet': get_mvcnet,
    }
    return models[name.lower()](**kwargs)

###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from ..nn import ConcurrentModule, SyncBatchNorm

from .base import BaseNet

__all__ = ['FCN', 'get_fcn_ms']


class FCN(BaseNet):
    """Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    Examples
    --------
    >>> model = FCN(nclass=21, backbone='resnet50')
    >>> print(model)
    """

    def __init__(self, nclass, backbone, aux=True, se_loss=False, with_global=False,
                 norm_layer=SyncBatchNorm, **kwargs):
        super(FCN, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = FCNHead(2048, nclass, norm_layer, self._up_kwargs, with_global)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = interpolate(x, imsize, **self._up_kwargs)
        outputs = [x]
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = interpolate(auxout, imsize, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GlobalPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(GlobalPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return interpolate(pool, (h, w), **self._up_kwargs)


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs={}, with_global=False):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self._up_kwargs = up_kwargs
        if with_global:
            self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                       norm_layer(inter_channels),
                                       nn.ReLU(),
                                       ConcurrentModule([
                                           Identity(),
                                           GlobalPooling(inter_channels, inter_channels,
                                                         norm_layer, self._up_kwargs),
                                       ]),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(2*inter_channels, out_channels, 1))
        else:
            self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                       norm_layer(inter_channels),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


def get_fcn_ms(dataset='pascal_voc', backbone='resnet50', pretrained=False,
               root='~/.encoding/models', **kwargs):
    r"""FCN model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    # infer number of classes
    from ..datasets import datasets, acronyms
    model = FCN(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_ms_%s_%s' % (backbone, acronyms[dataset]), root=root)))
    return model


# def get_fcn_resnet50_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
#     r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
#     <https://arxiv.org/pdf/1803.08904.pdf>`_
#
#     Parameters
#     ----------
#     pretrained : bool, default False
#         Whether to load the pretrained weights for model.
#     root : str, default '~/.encoding/models'
#         Location for keeping the model parameters.
#
#
#     Examples
#     --------
#     >>> model = get_fcn_resnet50_pcontext(pretrained=True)
#     >>> print(model)
#     """
#     return get_fcn('pcontext', 'resnet50', pretrained, root=root, aux=False, **kwargs)
#
#
# def get_fcn_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
#     r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
#     <https://arxiv.org/pdf/1803.08904.pdf>`_
#
#     Parameters
#     ----------
#     pretrained : bool, default False
#         Whether to load the pretrained weights for model.
#     root : str, default '~/.encoding/models'
#         Location for keeping the model parameters.
#
#
#     Examples
#     --------
#     >>> model = get_fcn_resnet50_ade(pretrained=True)
#     >>> print(model)
#     """
#     return get_fcn('ade20k', 'resnet50', pretrained, root=root, **kwargs)
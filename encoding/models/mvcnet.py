import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn import SyncBatchNorm
from .base import BaseNet


class MVCNet(BaseNet):
    def __init__(self, nclass, backbone, aux=False, se_loss=False, with_global=False,
                 norm_layer=SyncBatchNorm, **kwargs):
        super(MVCNet, self).__init__(nclass, backbone, aux, se_loss, with_global,
                                     norm_layer, **kwargs)

        nn.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.center = _DecoderBlock(2048, 1024, 800)
        self.classC = _ClassifierBlock(800, nclass)
        self.dec4 = _DecoderBlock(800, 512, 256)
        self.class4 = _ClassifierBlock(256, nclass)
        self.dec3 = _DecoderBlock(256, 180, 128)
        self.class3 = _ClassifierBlock(128, nclass)
        self.dec2 = _DecoderBlock(128, 100, 100)
        self.class2 = _ClassifierBlock(100, nclass)
        self.dec1 = nn.Sequential(
            nn.Conv2d(100, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.final = nn.Conv2d(64, nclass, kernel_size=1)
        self.final = nn.Conv2d(800, nclass, kernel_size=3, padding=1)

    def forward(self, x):
        _, _, _, c4 = self.base_forward(x)

        # Save each upsampling of the output feature maps
        # c4 = nn.upsample(c4)
        final = self.center(c4)
        # center = self.center(c4)
        # out_C = self.classC(center)
        # dec4 = self.dec4(center)
        # out_4 = self.class4(dec4)
        # dec3 = self.dec3(dec4)
        # out_3 = self.class3(dec3)
        # dec2 = self.dec2(dec3)
        # out_2 = self.class2(dec2)
        # dec1 = self.dec1(dec2)
        final = self.final(final)
        final = F.interpolate(final, x.size()[2:],
                              mode='bilinear', align_corners=True)
        output = [final]
        if self.multi_res_loss is True:
            output.append(out_C)
            output.append(out_4)
            output.append(out_3)
            output.append(out_2)
            return tuple(output)
        else:
            return tuple(output)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(
                mid_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.decode(x)


class _ClassifierBlock(nn.Module):
    def __init__(self, in_channels, nclasses):
        super(_ClassifierBlock, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=3, bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(in_channels//4, nclasses, kernel_size=3, bias=False)
        )

    def forward(self, x):
        return self.classifier(x)


def get_mvcnet(dataset='spacenet3', backbone='resnet50', pretrained=False,
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
    model = MVCNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('mvcnet_%s_%s' % (backbone, acronyms[dataset]), root=root)))
    return model

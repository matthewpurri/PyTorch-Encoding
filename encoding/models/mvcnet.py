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

        self.head = MVCNetHead()

    def forward(self, x):
        c1, c2, c3, c4 = self.base_forward(x)

        # Save each upsampling of the output feature maps
        if self.multi_res_loss is True:
            final, out4, out3, out2 = self.head(x, c4, c3, c2, c1,
                                                self.multi_res_loss)
            final = F.interpolate(final, x.size()[2:],
                                  mode='bilinear', align_corners=True)
            output = [final]
            output.append(out4)
            output.append(out3)
            output.append(out2)
            return tuple(output)
        else:
            final = self.head(x, c4, c3, c2, c1, self.multi_res_loss)
            final = F.interpolate(final, x.size()[2:],
                                  mode='bilinear', align_corners=True)
            output = [final]
            return tuple(output)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
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


class MVCNetHead(nn.Module):
    def __init__(self):
        super(MVCNetHead, self).__init__()
        self.center = _DecoderBlock(2048, 1024, 512)
        self.dec4 = _DecoderBlock(512, 512, 256)
        self.dec3 = _DecoderBlock(256+1024, 256, 128)
        self.dec2 = _DecoderBlock(128+512, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256+64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.class4 = _ClassifierBlock(256, 12)
        self.class3 = _ClassifierBlock(128, 12)
        self.class2 = _ClassifierBlock(64, 12)
        self.final = nn.Conv2d(64, 12, kernel_size=1)

    def forward(self, x, c4, c3, c2, c1, loss):
        center = self.center(c4)
        dec4 = self.dec4(center)
        dec3 = self.dec3(torch.cat(
            [dec4, F.interpolate(c3, dec4.size()[2:],
                                 mode='bilinear',
                                 align_corners=True)], 1))
        dec2 = self.dec2(torch.cat(
            [dec3, F.interpolate(c2, dec3.size()[2:],
                                 mode='bilinear',
                                 align_corners=True)], 1))
        dec1 = self.dec1(torch.cat(
            [dec2, F.interpolate(c1, dec2.size()[2:],
                                 mode='bilinear',
                                 align_corners=True)], 1))

        final = self.final(dec1)
        if loss is True:
            out4 = self.class4(dec4)
            out3 = self.class3(dec3)
            out2 = self.class2(dec2)
            return final, out4, out3, out2
        else:
            return final


def make_layers_from_size(sizes):
    layers = []
    for size in sizes:
        layers += [nn.Conv2d(size[0], size[1], kernel_size=3, padding=1),
                   nn.BatchNorm2d(size[1], momentum=0.1),
                   nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class MVCNetHead_old(nn.Module):
    def __init__(self):
        super(MVCNetHead, self).__init__()
        # Layer 5
        self.deconv5 = nn.ConvTranspose2d(2048, 1028, kernel_size=2,
                                          stride=2, padding=1)
        self.CBR5_RGB_DEC = make_layers_from_size([[1028, 512],
                                                   [512, 512],
                                                   [512, 512]])
        self.dropout5 = nn.Dropout2d(p=0.4)

        # Layer 4
        self.deconv4 = nn.ConvTranspose2d(512, 512, kernel_size=2,
                                          stride=2, padding=1)
        self.CBR4_RGB_DEC = make_layers_from_size([[512, 512],
                                                   [512, 512],
                                                   [512, 256]])
        self.dropout4 = nn.Dropout2d(p=0.4)

        # Layer 3
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2,
                                          stride=2, padding=1)
        self.CBR3_RGB_DEC = make_layers_from_size([[256, 256],
                                                   [256, 256],
                                                   [256, 128]])
        self.dropout3 = nn.Dropout2d(p=0.4)

        # Layer 2
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2,
                                          stride=2, padding=1)
        self.CBR2_RGB_DEC = make_layers_from_size([[128, 128],
                                                   [128, 128],
                                                   [128, 64]])

        # Layer 1
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2,
                                          stride=2, padding=1)
        self.CBR1_RGB_DEC = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(64, 12, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.deconv5(x)
        x = self.CBR5_RGB_DEC(x)
        x = self.dropout5(x)

        # Stage 4 dec
        x = self.deconv4(x)
        x = self.CBR4_RGB_DEC(x)
        x = self.dropout4(x)

        # Stage 3 dec
        x = self.deconv3(x)
        x = self.CBR3_RGB_DEC(x)
        x = self.dropout3(x)

        # Stage 2 dec
        x = self.deconv2(x)
        x = self.CBR2_RGB_DEC(x)

        # Stage 1 dec
        x = self.deconv1(x)
        x = self.CBR1_RGB_DEC(x)

        return x


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

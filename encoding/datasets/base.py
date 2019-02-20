###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import torch.utils.data as data

__all__ = ['BaseDataset', 'test_batchify_fn']


class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None,
                 target_transform=None, base_size=520, crop_size=480,
                 vAOI=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.vAOI = vAOI
        if self.mode == 'train':
            print('BaseDataset: base_size {}, crop_size {}'.
                  format(base_size, crop_size))

    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def make_pred(self, x):
        return x + self.pred_offset

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        # final transform
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        w, h = img.size
        long_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # final transform
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()

    def _val_sync_transform_multi(self, img, mask):
        """
        The C channel version of the _val_sync_transform function
        where C > 3.

        img: Numpy image [HxWxC]
        mask: PIL image [HxW]
        """
        outsize = self.crop_size
        short_size = outsize
        w, h = img.shape[:2]
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)

        mask = mask.resize((ow, oh), Image.NEAREST)
        w, h = mask.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))

        trans_img = np.zeros((self.crop_size, self.crop_size, img.shape[2]))
        for c in range(img.shape[2]):
            ch_img = img[:, :, c]
            pil_img = Image.fromarray(ch_img)
            pil_img = pil_img.resize((ow, oh), Image.BILINEAR)
            # center crop
            pil_img = pil_img.crop((x1, y1, x1+outsize, y1+outsize))
            trans_img[:, :, c] = np.array(pil_img)
        # final transform
        return trans_img, self._mask_transform(mask)

    def _sync_transform_multi(self, img, mask):
        """
        The C channel version of the _sync_transform function
        where C > 3.

        img: Numpy image [HxWxC]
        mask: PIL image [HxW]
        """
        crop_size = self.crop_size
        w, h = img.shape[:2]

        long_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh

        # random mirror
        mirror_p = random.random()

        # Alter mask
        if mirror_p < 0.5:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.resize((ow, oh), Image.NEAREST)

        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = mask.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))

        # Channel-wise transformation
        trans_img = np.zeros((self.crop_size, self.crop_size, img.shape[2]))
        for c in range(img.shape[2]):
            ch_img = img[:, :, c]
            pil_img = Image.fromarray(ch_img)
            if mirror_p < 0.5:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            pil_img = pil_img.resize((ow, oh), Image.BILINEAR)

            # pad crop
            if short_size < crop_size:
                pil_img = ImageOps.expand(pil_img, border=(0, 0, padw, padh), fill=0)
            # random crop crop_size
            pil_img = pil_img.crop((x1, y1, x1+crop_size, y1+crop_size))
            trans_img[:, :, c] = np.array(pil_img)

        # final transform
        return trans_img, self._mask_transform(mask)


def test_batchify_fn(data):
    error_msg = "batch must contain tensors, tuples or lists; found {}"
    if isinstance(data[0], (str, torch.Tensor)):
        return list(data)
    elif isinstance(data[0], (tuple, list)):
        data = zip(*data)
        return [test_batchify_fn(i) for i in data]
    raise TypeError((error_msg.format(type(batch[0]))))

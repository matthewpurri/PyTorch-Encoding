import os
import gdal
import torch
from PIL import Image
from glob import glob

from .base import BaseDataset


class SpaceNet3Segmentation(BaseDataset):
    NUM_CLASS = 12
    NUM_CHANNELS = 3

    def __init__(self, root='/media/Data_Drive/SpaceNet/3_Band/', split='train',
                 mode=None, transform=None, target_transform=None,
                 **kwargs):
        super(SpaceNet3Segmentation, self).__init__(root, split, **kwargs)
        self.transform = transform

        self.image_paths, self.mask_paths = _get_spacenet3_pairs(self.root, split, **kwargs)

        if len(self.image_paths) == 0:
            raise RuntimeError('No images found for dataset.')

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        mask = gdal.Open(self.mask_paths[index]).ReadAsArray()

        if self.split == 'train':
            mask = Image.fromarray(mask)
            img, mask = self._sync_transform(img, mask)
        elif self.split == 'val':
            mask = Image.fromarray(mask)
            img, mask = self._val_sync_transform(img, mask)
        elif self.split == 'test':
            mask = torch.from_numpy(mask)
        else:
            raise RuntimeError('Incorrect split value of {}'.format(self.split))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.image_paths)


def _get_spacenet3_pairs(root, split, **kwargs):
    gt_root = os.path.split(root[:-1])[0]
    gt_root = os.path.join(gt_root, 'Ground_Truth/')
    img_paths = []
    mask_paths = []
    if split == 'train':
        AOI_dirs = glob(root+'*/')
        AOI_names = [os.path.basename(os.path.normpath(AOI)) for AOI in AOI_dirs]
        AOI_names.remove(kwargs['vAOI'])
        train_AOIs = AOI_names
        for AOI in train_AOIs:
            AOI_dir = os.path.join(root, AOI)
            AOI_img_paths = sorted(glob(AOI_dir+'/*.png'))
            for img_path in AOI_img_paths:
                # Find the corresponding ground truth mask
                img_name = os.path.splitext(os.path.split(img_path)[1])[0]
                gt_name = 'gt_' + str(img_name[-3:]) + '.tif'
                gt_path = os.path.join(gt_root, AOI, gt_name)
                # Add to image and mask path lists
                img_paths.append(img_path)
                mask_paths.append(gt_path)
    elif split == 'val' or split == 'test':
        val_AOIs = []
        val_AOIs.append(kwargs['vAOI'])
        for AOI in val_AOIs:
            AOI_dir = os.path.join(root, AOI)
            AOI_img_paths = sorted(glob(AOI_dir+'/*.png'))
            for img_path in AOI_img_paths:
                # Find the corresponding ground truth mask
                img_name = os.path.splitext(os.path.split(img_path)[1])[0]
                gt_name = 'gt_' + str(img_name[-3:]) + '.tif'
                gt_path = os.path.join(gt_root, AOI, gt_name)
                # Add to image and mask path lists
                img_paths.append(img_path)
                mask_paths.append(gt_path)
    return img_paths, mask_paths

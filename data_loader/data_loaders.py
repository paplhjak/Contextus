import os
import random
from PIL import Image
import numpy as np
import os.path as osp
import random
from base import BaseDataLoader
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.utils.data

import pickle


class KittiDataLoader(BaseDataLoader):
    """
    Kitti data loading using BaseDataLoader
    """

    def __init__(self, isval=False, sparsity=100, crop_width=1216, crop_height=352, rotation_max_angle=0.0,
                 horizontal_flip_p=0.5, batch_size=8,
                 shuffle=True, validation_split=0.0, num_workers=1, istest=False, use_coarse=False):
        self.dataset = KittiDepthCompletionDataset(isval, crop_width, crop_height, rotation_max_angle,
                                                   horizontal_flip_p, sparsity)
        super().__init__(self.dataset, isval, sparsity, crop_width, crop_height, rotation_max_angle,
                         horizontal_flip_p, batch_size, shuffle, validation_split, num_workers, istest, use_coarse)




class KittiDepthCompletionDataset(torch.utils.data.Dataset):
    def __init__(self, isval, crop_width, crop_height, rotation_max_angle, horizontal_flip_p, sparsity):
        self.isVal = isval
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.rotation_max_angle = rotation_max_angle
        self.horizontal_flip_p = horizontal_flip_p
        self.sparsity = sparsity
        self.totensor = T.ToTensor()
        super().__init__()

        self.depth = []
        self.labels = []
        self.rgbs = []

        if (self.isVal):
            if self.sparsity == 100:
                with open('/mnt/data/vras/datasets/kitti/path_lists/KITTI_val_rgb.data', "rb") as f:
                    self.rgbs = pickle.load(f)
                with open('/mnt/data/vras/datasets/kitti/path_lists/KITTI_val_depth.data', "rb") as f:
                    self.depth = pickle.load(f)
                with open('/mnt/data/vras/datasets/kitti/path_lists/KITTI_val_depth_gt.data', "rb") as f:
                    self.labels = pickle.load(f)
            else:
                with open('/mnt/data/vras/datasets/kitti/path_lists/sparse_' + str(
                        self.sparsity) + '_KITTI_val_rgb.data', "rb") as f:
                    self.rgbs = pickle.load(f)
                with open('/mnt/data/vras/datasets/kitti/path_lists/sparse_' + str(
                        self.sparsity) + '_KITTI_val_depth.data', "rb") as f:
                    self.depth = pickle.load(f)
                with open('/mnt/data/vras/datasets/kitti/path_lists/sparse_' + str(
                        self.sparsity) + '_KITTI_val_depth_gt.data', "rb") as f:
                    self.labels = pickle.load(f)
        else:
            with open('/mnt/data/vras/datasets/kitti/path_lists/KITTI_train_rgb.data', "rb") as f:
                self.rgbs = pickle.load(f)
            with open('/mnt/data/vras/datasets/kitti/path_lists/KITTI_train_depth.data', "rb") as f:
                self.depth = pickle.load(f)
            with open('/mnt/data/vras/datasets/kitti/path_lists/KITTI_train_depth_gt.data', "rb") as f:
                self.labels = pickle.load(f)


    def __len__(self):
        return len(self.depth)

    def depth_transform(self, pil_img):
        # cast pil image to np.array for manipulation
        depth_png = np.array(pil_img, dtype=int)[:, :, np.newaxis]
        # kitti should be 16bit
        assert (np.max(depth_png) > 255)
        # depth for a pixel can be computed
        # in meters by converting the uint16 value to float and dividing it by 256.0
        depth = depth_png.astype(np.float) / 256.
        return depth

    def transform(self, raw, gt, rgb):
        # Crop data
        i, j, h, w = T.RandomCrop.get_params(raw, output_size=(self.crop_height, self.crop_width))
        raw = TF.crop(raw, 352-self.crop_height, j, self.crop_height, self.crop_width)
        gt = TF.crop(gt, 352-self.crop_height, j, self.crop_height, self.crop_width)
        rgb = TF.crop(rgb, 352-self.crop_height, j, self.crop_height, self.crop_width)

        # Random rotation
        angle = np.random.uniform(-self.rotation_max_angle, self.rotation_max_angle)
        raw = TF.rotate(raw, angle, resample=Image.NEAREST)
        gt = TF.rotate(gt, angle, resample=Image.NEAREST)
        rgb = TF.rotate(rgb, angle, resample=Image.NEAREST)

        # Random horizontal flip
        if random.random() < self.horizontal_flip_p:
            raw = TF.hflip(raw)
            gt = TF.hflip(gt)
            rgb = TF.hflip(rgb)
        return raw, gt, rgb

    def __getitem__(self, index):
        raw_path = self.depth[index]
        gt_path = self.labels[index]
        rgb_path = self.rgbs[index]

        raw_pil = Image.open(raw_path)
        gt_pil = Image.open(gt_path)
        rgb_pil = Image.open(rgb_path)
        
        """
        raw_pil = self.depth_pils[index]
        gt_pil = self.label_pils[index]
        rgb_pil = self.rgb_pils[index]
        """

        # rotation, crop, horizontal flip
        raw_pil, gt_pil, rgb_pil = self.transform(raw_pil, gt_pil, rgb_pil)

        raw = self.depth_transform(raw_pil)
        gt = self.depth_transform(gt_pil)
        rgb = np.array(rgb_pil, dtype=int)[:, :, np.newaxis]

        raw = self.totensor(raw).float()
        gt = self.totensor(gt).float()

        #rgb_new = rgb.reshape((3, self.crop_height, self.crop_width))
        rgb_new = rgb[:,:,0,:]
        #print(rgb_new.shape)
        rgb_new = torch.from_numpy(rgb_new).permute(2,0,1).float()
        # depth, rgb, gt depth
        return raw, rgb_new, gt

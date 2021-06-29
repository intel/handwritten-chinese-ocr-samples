"""
Apache v2 license
Copyright (C) <2018-2021> Intel Corporation
SPDX-License-Identifier: Apache-2.0
"""


import os
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, data_path: str, img_shape: tuple, phase: str, batch_size=1):
        super(ImageDataset, self).__init__()
        if not (phase in ['train', 'val', 'test']):
            raise AssertionError(phase)

        self.data_list = []
        self.img_c = img_shape[0]
        self.img_h = img_shape[1]
        #self.img_w = img_shape[2]
        self.phase = phase
        self.batch_size = batch_size

        img_id_gt_txt = os.path.join(data_path, phase + '_img_id_gt.txt')
        with open(img_id_gt_txt, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',', 1)
                img_path = os.path.join(data_path, phase, line[0])
                if os.path.exists(img_path) and os.stat(img_path).st_size > 0 and line[1]:
                    self.data_list.append((img_path, line[1]))

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        img = self.pil_loader(img_path)
        return (img, label)

    def __len__(self):
        return self.batch_size * (len(self.data_list) // self.batch_size)

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            # TODO: convertion input from other format
            # assume all the input are 8-bit grayscale images
            img = Image.open(f)
            img = np.array(img)

            height, width = img.shape[:2]
            ratio = self.img_h/height
            new_width = int(width * ratio)
            img_resize = cv2.resize(img, (new_width, self.img_h),
                                    interpolation=cv2.INTER_AREA)
            img_resize = img_resize if self.img_c == 3 else img_resize[:, :, np.newaxis]
            return img_resize


# to be removed
class ZerosPAD(object):
    def __init__(self, max_size):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size

    def __call__(self, img):
        img = self.toTensor(img)
        c, h, w = img.shape
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad

        return Pad_img


class NormalizePAD(object):
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = \
                img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):
    def __init__(self, imgH=48, PAD='ZerosPAD'):
        self.imgH = imgH
        self.PAD = PAD

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        maxW = 0
        for image in images:
            h, w, c = image.shape
            if w > maxW:
                maxW = w

        if self.PAD == 'ZerosPAD':
            trans = ZerosPAD((1, self.imgH, maxW))
        elif self.PAD == 'NormalizePAD':
            trans = NormalizePAD((1, self.imgH, maxW))
        else:
            raise ValueError("not expected padding.")

        padded_images = []
        for image in images:
            h, w, c = image.shape
            padded_images.append(trans(image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in padded_images], 0)

        return image_tensors, labels

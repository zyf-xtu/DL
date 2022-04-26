#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 下午8:15
# @Author  : zhangyunfei
# @File    : mnist_dataset.py
# @Software: PyCharm
from torch.utils.data import Dataset
from PIL import Image


# 自定义数据读取
class MnistData(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
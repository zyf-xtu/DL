#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 下午7:51
# @Author  : zhangyunfei
# @File    : mnist_save.py
# @Software: PyCharm
import torch
import torchvision
import os
from torchvision import transforms
from torchvision.datasets import mnist
import torch.utils.data as data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import itertools


# 将mnist保存为图像
def save():
    os.makedirs('mnist/train', exist_ok=True)
    os.makedirs('mnist/test', exist_ok=True)
    for i in range(10):
        os.makedirs('mnist/train/' + str(i), exist_ok=True)
        os.makedirs('mnist/test/' + str(i), exist_ok=True)
    # 保存训练集
    for i, item in enumerate(train_loader):
        img, label = item
        img = img[0].cpu().numpy()
        array = (img.reshape((28, 28)) * 255).astype(np.uint8)
        img = Image.fromarray(array, 'L')
        label = label.cpu().numpy()[0]
        img_path = 'mnist/train/' + str(label) + '/' + str(i) + '.jpg'
        print(img_path)
        img.save(img_path)

    # 保存测试集
    for i, item in enumerate(test_loader):
        img, label = item
        img = img[0].cpu().numpy()
        array = (img.reshape((28, 28)) * 255).astype(np.uint8)
        img = Image.fromarray(array, 'L')
        label = label.cpu().numpy()[0]
        img_path = 'mnist/test/' + str(label) + '/' + str(i) + '.jpg'
        print(img_path)
        img.save(img_path)


# 查看部分mnist图像
def show():
    plt.figure(figsize=(16, 9))
    for i, item in enumerate(itertools.islice(train_loader,2,12)):
        plt.subplot(2, 5, i+1)
        img,label= item
        img = img[0].cpu().numpy()
        array = (img.reshape((28, 28)) * 255).astype(np.uint8)
        img = Image.fromarray(array, 'L')
        label = label.cpu().numpy()[0]
        plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()


if __name__ == '__main__':
    train_data = mnist.MNIST('mnist', train=True, transform=transforms.ToTensor(), download=True)
    test_data = mnist.MNIST('mnist', train=False, transform=transforms.ToTensor(), download=True)
    train_loader = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    test_loader = data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)
    train_total = train_loader.__len__()
    test_total = test_loader.__len__()
    labels = train_data.targets
    print(train_data.targets)
    print(train_total, test_total)
    dataiter = iter(train_data)
    print(train_data)
    images, labs = dataiter.__next__()
    print(type(images), type(labs))
    print(images.shape, labs)
    save()
    show()
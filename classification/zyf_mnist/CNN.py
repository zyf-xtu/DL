#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 下午8:16
# @Author  : zhangyunfei
# @File    : CNN.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary

'''
    自定义建立一个3层的卷积神经网络
'''


# CNN分类网络
class CnnNet(nn.Module):
    def __init__(self, classes=10):
        super(CnnNet, self).__init__()
        # 分类数
        self.classes = classes
        # 第一层卷积，输入：bs*3*28*28 输出：bs*16*14*14
        self.conv1 = nn.Sequential(
            # 卷积 输入：bs*3*28*28  输出：bs*16*28*28
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            # 归一化
            nn.BatchNorm2d(16),
            # 激活函数
            nn.ReLU(),
            # 最大池化：输入:bs*16*28*28  输出：bs*16*14*14
            nn.MaxPool2d(2)
        )
        # 第二层卷积，输入：bs*16*14*14 输出：bs*64*7*7
        self.conv2 = nn.Sequential(
            # 卷积 输入：bs*16*14*14  输出：bs*32*14*14
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            # 归一化
            nn.BatchNorm2d(32),
            # 激活函数
            nn.ReLU(),
            # 最大池化：输入:bs*16*14*14  输出：bs*32*7*7
            nn.MaxPool2d(2)
        )
        # 第三层卷积，输入：bs*32*7*7 输出：bs*64*3*3
        self.conv3 = nn.Sequential(
            # 卷积 输入：bs*32*7*7  输出：bs*64*3*3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # 归一化
            nn.BatchNorm2d(64),
            # 激活函数
            nn.ReLU(),
            # 最大池化：输入：bs*32*7*7 输出：bs*64*3*3
            nn.MaxPool2d(2)
        )
        # 自适应池化，将bs*64*3*3映射为bs*64*1*1
        self.advpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Linear(64, self.classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.advpool(x)
        # 需要将多维度的值展平为一维，送入linear中，但是需要保持batchsize的维度
        # 例如2*64*1*1 变成2*64
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    # 测试数据
    x = torch.rand((2, 3, 28, 28))
    cnn = CnnNet(classes=10)
    print(cnn)
    out = cnn(x)
    print(out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn = cnn.to(device)
    # 网络模型的数据流程及参数信息
    summary(cnn, (3, 28, 28))

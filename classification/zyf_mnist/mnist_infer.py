#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 下午8:20
# @Author  : zhangyunfei
# @File    : mnist_infer.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import json

'''
    手写数字预测+特征提取

'''


# 预测类
class Pred:
    def __init__(self):
        self.labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        model_path = 'checkpoints/best_model.pth'
        # 加载模型
        self.model = torch.load(model_path)
        self.device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    # 预测
    def predict(self, img_path):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = transform(img)
        img = img.view(1, 3, 28, 28).to(self.device)
        output = self.model(img)
        output = torch.softmax(output, dim=1)
        # 每个预测值的概率
        probability = output.cpu().detach().numpy()[0]
        # 找出最大概率值的索引
        output = torch.argmax(output, dim=1)
        index = output.cpu().numpy()[0]
        # 预测结果
        pred = self.labels[index]
        print(pred, probability[index])
        return pred


if __name__ == '__main__':
    img_path = 'mnist/test/1/12.jpg'
    pred = Pred()
    res = pred.predict(img_path)
    print(res)
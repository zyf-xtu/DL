#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 下午8:16
# @Author  : zhangyunfei
# @File    : train.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import os
from mnist_dataset import MnistData
from CNN import CnnNet
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 将数据集划分训练集和验证集
def split_data(files):
    """
    :param files:
    :return:
    """
    random.shuffle(files)
    # 计算比例系数，分割数据训练集和验证集
    ratio = 0.9
    offset = int(len(files) * ratio)
    train_data = files[:offset]
    val_data = files[offset:]
    return train_data, val_data


# 训练
def train(model, loss_func, optimizer, checkpoints, epoch):
    print('Train......................')
    # 记录每个epoch的loss和acc
    best_acc = 0
    best_epoch = 0
    # 训练过程
    for epoch in range(1, epoch):
        # 设置计时器，计算每个epoch的用时
        start_time = time.time()
        model.train()  # 保证每一个batch都能进入model.train()的模式
        # 记录每个epoch的loss和acc
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
        for i, (inputs, labels) in enumerate(train_data):
            # print(batch_size)
            # print(i, inputs, labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 预测输出
            outputs = model(inputs)
            # 计算损失
            loss = loss_func(outputs, labels)
            # print(outputs)
            # 因为梯度是累加的，需要清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 优化器
            optimizer.step()
            # 计算准确率
            output = nn.functional.softmax(outputs, dim=1)
            pred = torch.argmax(output, dim=1)
            acc = torch.sum(pred == labels)
            train_loss += loss.item()
            train_acc += acc.item()
        # 验证集进行验证
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 预测输出
                outputs = model(inputs)
                # 计算损失
                loss = loss_func(outputs, labels)
                # 计算准确率
                output = nn.functional.softmax(outputs, dim=1)
                pred = torch.argmax(output, dim=1)
                # print(pred,'================')
                # print(pred==labels,'=====----------======')
                acc = torch.sum(pred == labels)
                # acc = calculat_acc(outputs, labels)
                val_loss += loss.item()
                val_acc += acc.item()

        # 计算每个epoch的训练损失和精度
        train_loss_epoch = train_loss / train_data_size
        train_acc_epoch = train_acc / train_data_size
        # 计算每个epoch的验证集损失和精度
        val_loss_epoch = val_loss / val_data_size
        val_acc_epoch = val_acc / val_data_size
        end_time = time.time()
        print(
            'epoch:{} | time:{:.4f} | train_loss:{:.4f} | train_acc:{:.4f} | eval_loss:{:.4f} | val_acc:{:.4f}'.format(
                epoch,
                end_time - start_time,
                train_loss_epoch,
                train_acc_epoch,
                val_loss_epoch,
                val_acc_epoch))

        # 记录验证集上准确率最高的模型
        best_model_path = checkpoints + "/" + 'best_model' + '.pth'
        if val_acc_epoch >= best_acc:
            best_acc = val_acc_epoch
            best_epoch = epoch
            torch.save(model, best_model_path)
        print('Best Accuracy for Validation :{:.4f} at epoch {:d}'.format(best_acc, best_epoch))
        # 每迭代50次保存一次模型
        # if epoch % 50 == 0:
        #     model_name = '/epoch_' + str(epoch) + '.pt'
        #     torch.save(model, checkpoints + model_name)
    # 保存最后的模型
    torch.save(model, checkpoints + '/last.pt')


if __name__ == '__main__':
    # batchsize
    bs = 32
    # learning rate
    lr = 0.01
    # epoch
    epoch = 100
    # checkpoints,模型保存路径
    checkpoints = 'checkpoints'
    os.makedirs(checkpoints, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 训练0-9的数据
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    base_dir = 'mnist/train'
    imgs = []
    # 获取数据
    for label in labels:
        label_dir = os.path.join(base_dir, str(label))
        images = os.listdir(label_dir)
        for img in images:
            img_path = os.path.join(label_dir, img)
            imgs.append((img_path, label))
    print(len(imgs))
    # 将训练数据拆分成训练集和验证集
    trains, vals = split_data(imgs)
    # 加载训练集数据
    train_dataset = MnistData(trains, transform=transform)
    train_data = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
    # 加载验证集数据
    val_dataset = MnistData(vals, transform=transform)
    val_data = DataLoader(val_dataset, batch_size=bs, shuffle=True, num_workers=0)
    train_data_size = train_dataset.__len__()
    val_data_size = val_dataset.__len__()
    print(train_dataset.__len__())
    print(val_dataset.__len__())

    # 加载模型
    model = CnnNet(classes=10)
    # GPU是否可用，如果可用，则使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 损失函数
    loss_func = nn.CrossEntropyLoss()
    # 优化器，使用SGD,可换Adam
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-3)
    # 训练
    train(model, loss_func, optimizer, checkpoints, epoch)
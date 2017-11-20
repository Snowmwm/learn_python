#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from read_mnist import load_mnist

#超参数设置
EPOCH = 4
BATCH_SIZE = 6
LR = 0.001
MOMENTUM = 0.9
SEED = 999

#随机种子设置
use_gpu = torch.cuda.is_available()
torch.manual_seed(SEED)
if use_gpu:
    torch.cuda.manual_seed(SEED)

#读取数据集
path = 'c:/PTW/learn_python/mnist'
trX, teX, trY, teY = load_mnist(path)
#train_set, test_set = read_mnist(path)


#数据预处理
#PyTorch的卷积神经网络的inputs是一个四维的Variable,所以需要增加表示层数的一维
'''
trX = trX.reshape(-1,1,28,28)
teX = teX.reshape(-1,1,28,28)
'''

#使用Dataloader封装数据
def dataloader(trX, trY, teX, teY):
    trX /= 255.
    teX /= 255.

    trX = torch.from_numpy(trX).float()
    trY = torch.from_numpy(trY).long()
    teX = torch.from_numpy(teX).float()
    teY = torch.from_numpy(teY).long()

    train_dataset = Data.TensorDataset(data_tensor=trX, target_tensor=trY)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = Data.TensorDataset(data_tensor=teX, target_tensor=teY)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader

train_loader, test_loader = dataloader(trX, trY, teX, teY)

#建立卷积神经网络
'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

cnn = CNN()

#优化器和损失函数：
#使用SGD梯度下降
optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=MOMENTUM)
#使用交叉熵损失函数
loss_func = nn.CrossEntropyLoss()

'''
#建立卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input shape (1, 28, 28)
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # output shape (16, 14, 14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # output shape (32, 7, 7)
        self.out = nn.Linear(in_features=32*7*7, out_features=10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output
cnn = CNN()

#优化器和损失函数
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

#训练网络

if use_gpu:
    cnn.cuda() #是否启用GPU加速

def train(epoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        #获取输入和标签
        inputs, labels = data
        #转成Variable类型，否则无法进行梯度计算
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        #参数梯度清零
        optimizer.zero_grad()
        #前向传播，损失计算，后向传播
        outputs = cnn(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        #参数更新
        optimizer.step()
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

def test(epoch):
    correct = 0
    total = 0
    for data in test_loader:
        test_x, test_y = data
        if use_gpu:
            test_out = cnn(Variable(test_x.cuda()))
        else:
            test_out = cnn(Variable(test_x))

        predicted = torch.max(test_out.data, 1)[1]
        #get the index of the max log-probability
        total += test_y.size(0)
        correct += (predicted == test_y.cuda()).sum()
    print('[%d] Accuracy: %.2f %%' %
    (epoch + 1, 100 * correct / total))
    running_loss = 0.0

for epoch in range(EPOCH):
    train(epoch)
    test(epoch)
print('Finished Training')
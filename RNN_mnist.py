#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from read_mnist import load_mnist

#超参数设置
EPOCH = 4
BATCH_SIZE = 12
LR = 0.01
SEED = 666
TIME_STEP = 28 #rnn时间步数（图片高度）
INPUT_SIZE = 28 #rnn每步输入值（图片每行像素）

#随机种子设置
torch.manual_seed(SEED)
    
#读取数据集
path = 'c:/PTW/learn_python/mnist'
trX, teX, trY, teY = load_mnist(path)

#使用Dataloader封装数据
def dataloader(trX, trY, teX, teY):
    trX /= 255.
    teX /= 255.
    
    trX = torch.from_numpy(trX).float()
    trY = torch.from_numpy(trY).long()
    teX = torch.from_numpy(teX).float()
    teY = torch.from_numpy(teY).long()
    
    #(length, 1, row, col) -> (batch, time_step, input_size)
    trX = torch.squeeze(trX, 1)
    teX = torch.squeeze(teX, 1)

    train_data = Data.TensorDataset(data_tensor=trX, target_tensor=trY)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_data = Data.TensorDataset(data_tensor=teX, target_tensor=teY)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader

train_loader, test_loader = dataloader(trX, trY, teX, teY)


#建立循环神经网络
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True, #batch放在第一个维度(batch, time_step, input_size)
        )
        
        self.out = nn.Linear(64, 10)
        
    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])  #(batch, time_step, input_size)
        return out

rnn = RNN()

#优化器和损失函数
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

#训练
def train(epoch):
    running_loss = 0.0
    for step, (x, y) in enumerate(train_loader): #获取输入和标签(x, y)
#        x = x.view(-1, 28, 28)
        #转成Variable类型
        b_x, b_y = Variable(x), Variable(y)
        
        output = rnn(b_x) #前向传播
        loss = loss_func(output, b_y) #损失计算
        optimizer.zero_grad() #参数梯度清零
        loss.backward() #后向传播
        optimizer.step() #参数更新
        
        running_loss += loss.data[0]
        
        if (step + 1) % 1000 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch+1, step+1, running_loss/1000))
            running_loss = 0.0

#测试
def test(epoch):
    correct = 0
    total = 0
    for (test_x, test_y) in test_loader:
#        test_x = test_x.view(-1, 28, 28)
        test_out = rnn(Variable(test_x))

        predicted = torch.max(test_out.data, 1)[1]

        total += test_y.size(0)
        correct += (predicted == test_y).sum()
    print('[%d] Accuracy: %.2f %%' % (epoch+1, 100 * correct / total))
                
for epoch in range(EPOCH):
    train(epoch)
    test(epoch)
print('Finished Training')

"""
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader): #获取输入和标签(x, y)
        b_x, b_y = Variable(x.view(-1, 28, 28)), Variable(y)

        output = rnn(b_x) #前向传播
        loss = loss_func(output, b_y) #损失计算
        optimizer.zero_grad() #参数梯度清零
        loss.backward() #后向传播
        optimizer.step() #参数更新
        
        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size
            print('Epoch: ', epoch, '| train loss: %.4f' %loss.data[0], 
                '| test accuracy: %.3f' %accuracy)
"""                













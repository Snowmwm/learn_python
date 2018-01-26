#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os

from read_mnist import load_mnist

if not os.path.exists('./img'):
    os.mkdir('./img')

def to_img(x):
    return x.view(-1, 1, 28, 28)

#超参数设置
EPOCH = 100
BATCH_SIZE = 25
ZDIM = 100 #噪声维度
SEED = 999

#随机种子设置
use_gpu = torch.cuda.is_available()
torch.manual_seed(SEED)
if use_gpu:
    torch.cuda.manual_seed(SEED)


#读取数据集
path = 'c:/PTW/learn_python/mnist'
trX, teX, trY, teY = load_mnist(path)

#使用Dataloader封装数据
def dataloader(trX, trY):
    trX /= 255.
    trX = torch.from_numpy(trX).float()
    trY = torch.from_numpy(trY).long()
    trX = torch.squeeze(trX, 1)
    train_data = Data.TensorDataset(data_tensor=trX, target_tensor=trY)
    train_loader = Data.DataLoader(dataset=train_data,
                                    batch_size=BATCH_SIZE, shuffle=True)
    return train_loader

train_loader = dataloader(trX, trY)

#全连接网络
'''
#判别网络
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.dis(x)
        return x

#生成网络
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()
            )

    def forward(self, x):
        x = self.gen(x)
        return x

D = discriminator()
G = generator()
'''

#卷积网络

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),  # batch, 32, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 14, 14
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 7, 7
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # batch, 3136=1x56x56
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),  # batch, 50, 56, 56
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 56, 56
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2),  # batch, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x

D = discriminator()
G = generator(ZDIM, 3136)

if use_gpu:
    D = D.cuda()
    G = G.cuda()

loss_func = nn.BCELoss() #二分类的交叉熵
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

for epoch in range(EPOCH):
    for i, (img, _) in enumerate(train_loader):
        num_img = img.size(0) # batch_size
        #img = img.view(num_img, -1) #将图片展开成28x28=784
        img = img.unsqueeze(1) # batch, 1, 28, 28
        real_img = Variable(img) #将真实图片包装成Variable
        real_label = Variable(torch.ones(num_img)) #定义真实图片label为1
        fake_label = Variable(torch.zeros(num_img)) #定义生成图片label为0
        if use_gpu:
            real_img, real_label, fake_label = real_img.cuda(), \
                                real_label.cuda(), fake_label.cuda()

        # 训练判别网络D
        real_out = D(real_img) # 将真实的图片放入判别器
        d_loss_real = loss_func(real_out, real_label) #真实图片的loss
        real_scores = real_out

        z = Variable(torch.randn(num_img, ZDIM)) #随机生成噪声
        if use_gpu:
            z = z.cuda()

        fake_img = G(z) #将噪声放入生成网络生成假的图片
        fake_out = D(fake_img) # 将生成的图片放入判别器
        d_loss_fake = loss_func(fake_out, fake_label) #生成图片的loss
        fake_scores = fake_out

        d_loss = d_loss_real + d_loss_fake #判别网络D的loss
        d_optimizer.zero_grad() #梯度归零
        d_loss.backward() #反向传播
        d_optimizer.step() #参数更新

        # 训练生成网络G
        z = Variable(torch.randn(num_img, ZDIM)) #随机生成噪声
        if use_gpu:
            z = z.cuda()

        fake_img = G(z) #生成假的图片
        output = D(fake_img) #经过判别器得到结果
        g_loss = loss_func(output, real_label) #生成网络G的loss

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 400 == 0:
            print('Epoch: {},Batch: {}, d_loss: {:.3f}, g_loss: {:.3f} '
                  'D real: {:.3f}, D fake: {:.3f}'.format(
                      epoch+1, i+1, d_loss.data[0], g_loss.data[0],
                      real_scores.data.mean(), fake_scores.data.mean()))
    if epoch == 0:
        real_images = to_img(real_img.cpu().data)
        save_image(real_images, './img/real_images.png')

    if (epoch < 10) or ((epoch + 1) % 10 == 0):
        fake_images = to_img(fake_img.cpu().data)
        save_image(fake_images, './img/cnn_fake_images_{}.png'.format(epoch + 1))

torch.save(G.state_dict(), './generator.pkl')
torch.save(D.state_dict(), './discriminator.pkl')
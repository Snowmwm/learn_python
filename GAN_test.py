#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


#超参数设置
SEED = 666
BATCH_SIZE = 32
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 5              
ART_COMPONENTS = 15

PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS)for _ in range(BATCH_SIZE)])

torch.manual_seed(SEED)
np.random.seed(SEED)

"""
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
plt.legend(loc='upper right')
plt.show()
"""

def artist_works():     
    #生成目标数据
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    paintings = torch.from_numpy(paintings).float()
    return Variable(paintings)
    
#生成网络
G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),
)

#判别网络
D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128,1),
    nn.Sigmoid(),
)

#优化器
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()

#训练
for step in range(10000):
    artist_paintings = artist_works()  #目标数据
    G_ideas = Variable(torch.randn(BATCH_SIZE, N_IDEAS))
    G_paintings = G(G_ideas)  #生成数据
    
    prob_artist0 = D(artist_paintings) #D希望将其识别为1
    prob_artist1 = D(G_paintings) #D希望将其识别为0
    
    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1 - prob_artist1))
    #最大化D猜测的正确率(最大化D将目标数据识别为1，将生成数据识别为0的概率)
    
    G_loss = torch.mean(torch.log(1 - prob_artist1))
    #最小化D猜测G生成数据的正确率(最小化D将生成数据识别为0的概率)
    
    #反向传播
    opt_D.zero_grad()
    D_loss.backward(retain_variables=True) #保留参数用于生成网络的反向传播
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
    
    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 15})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 15})
        plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=12);plt.draw();plt.pause(0.01)

plt.ioff()
plt.show()
    
    
    
    

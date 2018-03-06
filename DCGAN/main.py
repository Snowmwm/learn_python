#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import Discriminator, Generator, Generator_2

#超参数设置
EPOCH = 200
BATCH_SIZE = 32
ZDIM = 100 #噪声维度
SEED = 666
LR = 0.0003
D_EVERY = 1
G_EVERY = 3 
SAVE_EVERY = 10 

N_EPOCH = 91
MODEL_PATH_D = './checkpoints/D_uc_90.pkl' 
MODEL_PATH_G = './checkpoints/G_uc_90.pkl'

#随机种子设置
use_gpu = torch.cuda.is_available()
torch.manual_seed(SEED)
if use_gpu:
    torch.cuda.manual_seed(SEED)

#加载数据集
path = './data'
transforms = transforms.Compose([
                transforms.Scale(96),
                transforms.CenterCrop(96),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
dataset = torchvision.datasets.ImageFolder(path,transform=transforms)
dataloader = Data.DataLoader(dataset, batch_size = BATCH_SIZE,
                             shuffle = True, num_workers = 0, drop_last=True)

#搭建神经网络
D = Discriminator(64)
G = Generator_2(ZDIM, 64)

#加载模型参数
if MODEL_PATH_D:
    D.load_state_dict(torch.load(MODEL_PATH_D))
    
if MODEL_PATH_G:
    G.load_state_dict(torch.load(MODEL_PATH_G))

loss_func = nn.BCELoss() #二分类的交叉熵

d_optimizer = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5,0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5,0.999))

real_label = Variable(torch.ones(BATCH_SIZE)) #定义真实图片label为1
fake_label = Variable(torch.zeros(BATCH_SIZE)) #定义生成图片label为0
z = Variable(torch.randn(BATCH_SIZE, ZDIM, 1, 1)) #随机生成噪声

if use_gpu: #将模型和数据移至GPU
    D.cuda()
    G.cuda()
    loss_func.cuda()
    real_label, fake_label = real_label.cuda(), fake_label.cuda()
    z = z.cuda()
    z1 = z.cuda()

for epoch in range(EPOCH):
    running_d_loss = 0.0
    running_g_loss = 0.0
    for i, (img, _) in enumerate(dataloader):
        real_img = Variable(img) #将真实图片包装成Variable
        if use_gpu:
            real_img = real_img.cuda()

        # 训练判别网络D
        if (i + 1) % D_EVERY == 0:
            d_optimizer.zero_grad() #梯度清零
            
            real_out = D(real_img) # 将真实的图片放入判别器
            d_loss_real = loss_func(real_out, real_label) #真实图片的loss
            d_loss_real.backward() #反向传播
            
            z.data.copy_(torch.randn(BATCH_SIZE,ZDIM,1,1))
            fake_img = G(z).detach() #将噪声放入生成网络生成假的图片
            fake_out = D(fake_img) # 将生成的图片放入判别器
            d_loss_fake = loss_func(fake_out, fake_label) #生成图片的loss
            d_loss_fake.backward() #反向传播
            
            d_optimizer.step() #参数更新
            
            d_loss = d_loss_real + d_loss_fake
            running_d_loss += d_loss.data[0]

        # 训练生成网络G
        if (i + 1) % G_EVERY == 0:
            z.data.copy_(torch.randn(BATCH_SIZE,ZDIM,1,1))
            fake_img = G(z) #生成假的图片
            output = D(fake_img) #经过判别器得到结果
            g_loss = loss_func(output, real_label) #生成网络G的loss
            
            running_g_loss += g_loss.data[0]

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        if (i + 1) % 400 == 0:
            print('Epoch:', epoch+N_EPOCH, ' Batch:', i+1, 
                ' d_loss:%.5f' %(running_d_loss/(BATCH_SIZE*i)), 
                ' g_loss:%.5f' %(running_g_loss/(BATCH_SIZE*i)))

    fake_images = G(z1).cpu().data #用相同的噪声z1生成假的图片,便于对比
    save_image(fake_images, './img_uc/fake_images_%s.png' %(epoch+N_EPOCH),
               normalize=True, range=(-1,1))
    
    if (epoch + 1) % SAVE_EVERY == 0:
        torch.save(G.state_dict(), './checkpoints/G_uc_%s.pkl' %(epoch+N_EPOCH))
        torch.save(D.state_dict(), './checkpoints/D_uc_%s.pkl' %(epoch+N_EPOCH))
      
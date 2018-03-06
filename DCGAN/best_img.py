#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from model import Discriminator, Generator, Generator_2

ZDIM = 100
GEN_NUM = 32
GEN_SEARCH = 512
MODEL_PATH_D = './checkpoints/D_200.pkl' 
MODEL_PATH_G = './checkpoints/G_uc_90.pkl'
SAVE_PATH = './best_img/G2_90_b.png'


G = Generator_2(ZDIM, 64).eval()
D = Discriminator(64).eval()

z = torch.randn(GEN_SEARCH, ZDIM, 1, 1).normal_(0,1) #随机生成噪声
z = Variable(z, volatile=True)

#加载预训练模型
D.load_state_dict(torch.load(MODEL_PATH_D))
G.load_state_dict(torch.load(MODEL_PATH_G))

#将模型和数据移至GPU
if torch.cuda.is_available(): 
    D.cuda()
    G.cuda()
    z = z.cuda()
    
#生成图片，并用判别网络计算图片的评分
fake_img = G(z)
scores = D(fake_img).data

#挑选最好的几张图
index = scores.topk(GEN_NUM)[1]
result = []
for i in index:
    result.append(fake_img.data[i])
    
#保存图片
save_image(torch.stack(result), SAVE_PATH, normalize=True, range=(-1,1))
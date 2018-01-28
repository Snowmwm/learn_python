#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

import utils
from VGG_net import Vgg16 
from Transfor_net import TransformerNet


#超参数设置
EPOCH = 2
BATCH_SIZE = 32
SEED = 666
LR = 0.001
IMAGE_SIZE = 256
CONTENT_WEIGHT = 1e5
STYLE_WEIGHT = 1e10

DATA_PATH = 'C:/PTW/learn_python/DCGAN/data' #训练数据文件夹
MODEL_PATH = './checkpoints/transformer.pth' #模型存放路径
STYLE_PATH = './img/style1.jpg' #风格图片存放路径
CONTENT_PATH = './img/input2.jpg' #内容图片存放路径
OUTPUT_PATH = './img/output1.png'


#随机种子设置
use_gpu = torch.cuda.is_available()
torch.manual_seed(SEED)
if use_gpu:
    torch.cuda.manual_seed(SEED)

#用风格图片训练一个风格迁移网络
def train(**kwargs):
    #加载数据集
    transfroms = transforms.Compose([
                    transforms.Scale(IMAGE_SIZE),
                    transforms.CenterCrop(IMAGE_SIZE),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x:x*255)
                    ])
    dataset = torchvision.datasets.ImageFolder(DATA_PATH, transfroms)
    dataloader = Data.DataLoader(dataset, BATCH_SIZE,
                                shuffle = True, num_workers = 0)

    #搭建神经网络
    transformer = TransformerNet() # 风格转换网络
    #transformer.load_state_dict(torch.load(MODEL_PATH,map_location=lambda _s,_: _s))

    vgg = Vgg16().eval() # vgg16损失网络

    #优化器
    optimizer = t.optim.Adam(transformer.parameters(),LR) 

    #加载风格图片
    style = utils.get_style_data(STYLE_PATH)

    #将模型和数据移至GPU
    if use_gpu:
        transformer.cuda()
        vgg.cuda()
        style = style.cuda()

    # 风格图片的gram矩阵
    style_v = Variable(style, volatile=True)
    features_style = vgg(style_v) #vgg网络返回的是一个列表
    gram_style = [Variable(utils.gram_matrix(y.data)) for y in features_style]

    # 训练
    for epoch in range(EPOCH):
        running_style_loss = 0.0
        running_content_loss = 0.0
        for i,(x,_) in enumerate(dataloader):
            x = Variable(x)
            if use_gpu:
                x = x.cuda()
            y = transformer(x)
            x, y = utils.normalize_batch(x), utils.normalize_batch(y)
            features_x = vgg(x)
            features_y = vgg(y)
            
            # content loss
            content_loss = CONTENT_WEIGHT*F.mse_loss(features_y.relu2_2,
                                                        features_x.relu2_2)

            # style loss
            style_loss = 0.
            for ft_y,gm_s in zip(features_y, gram_style):
                gram_y = utils.gram_matrix(ft_y)
                style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
            style_loss *= STYLE_WEIGHT
           
            #反向传播
            total_loss = content_loss + style_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_content_loss += content_loss.data[0]
            running_style_loss += style_loss.data[0]
            if (i + 1) % 100 == 0:
                print('Epoch:', epoch+1, ' Batch:', i+1, 
                ' content_loss:%.5f' %(running_content_loss/(BATCH_SIZE*i)), 
                ' style_loss:%.5f' %(running_style_loss/(BATCH_SIZE*i)))
                    
        torch.save(transformer.state_dict(),'.checkpoints/%s_style.pth' %epoch)

        
#使用训练好的风格迁移网络进行风格迁移        
def stylize(**kwargs): 
    # 图片处理
    content_image = torchvision.datasets.folder.default_loader(CONTENT_PATH)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)
    content_image = Variable(content_image, volatile=True)

    # 模型
    style_model = TransformerNet().eval()
    style_model.load_state_dict(torch.load(MODEL_PATH,
                                map_location=lambda _s,_: _s))
    if use_gpu:
        content_image = content_image.cuda()
        style_model.cuda()
    
    # 风格迁移与保存
    output = style_model(content_image)
    output_data = output.cpu().data[0]
    save_image(((output_data/255)).clamp(min=0,max=1),OUTPUT_PATH)
            
if __name__ == '__main__':
    train()
    stylize()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
from torch.autograd import Variable
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def gram_matrix(y):
    '''
    输入 b,c,h,w
    输出 b,c,c
    '''
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w) #bmm：batch的矩阵乘法
    return gram


def get_style_data(path):
    '''
    加载风格图片，
    输入： path， 文件路径
    返回： 形状 1*c*h*w， 分布 -2~2    tensor
    '''
    style_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean = IMAGENET_MEAN,std = IMAGENET_STD),
    ])

    style_image = torchvision.datasets.folder.default_loader(path)
    style_tensor = style_transform(style_image)
    return style_tensor.unsqueeze(0)

def normalize_batch(batch):
    '''
    将图片标准化
    输入: b,ch,h,w  0~255    Variable
    输出: b,ch,h,w  -2~2    Variable
    '''
    mean = batch.data.new(IMAGENET_MEAN).view(1,-1,1,1)
    std = batch.data.new(IMAGENET_STD).view(1,-1,1,1)
    mean = Variable(mean.expand_as(batch.data))
    std = Variable(std.expand_as(batch.data))
    return (batch/255.0 - mean) / std

















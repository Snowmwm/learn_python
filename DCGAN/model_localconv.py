#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

#反卷积层
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                stride, upsample=2):
        super(UpsampleConvLayer, self).__init__()
        #先上采样，然后再卷积
        self.upsample_layer = nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
        
        
#使用Localconv的反卷积层
class LocalUpsampleConvLayer(nn.Module):
    def __init__(self, in_height, in_width, in_channels, out_channels, 
                kernel_size, stride, upsample=2):
        super(LocalUpsampleConvLayer, self).__init__()
        self.upsample_layer = nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        
        #计算localconv的输入的高度和宽度
        input_height = upsample * in_height + 2 * reflection_padding
        input_width = upsample * in_width + 2 * reflection_padding
        self.conv2dlocal = nn.Conv2dLocal(
            input_height, input_width, in_channels, out_channels, 
            kernel_size, stride)

    def forward(self, x):
        x_in = x
        x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2dlocal(out)
        return out


#判别器
class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        
        #(batch x 3 x 96 x 96) -> (batch x ndf x 32 x 32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, True),
            ) 
        
        #(batch x ndf x 32 x 32) -> (batch x ndf*2 x 16 x 16)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),  
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),
            )
            
        #(batch x ndf*2 x 16 x 16) -> (batch x ndf*4 x 8 x 8)    
        #使用不共享参数的local conv来识别人脸不同区域的局部特征
        self.conv3 = nn.Sequential(
            nn.Conv2dLocal(16, 16, ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),
            )
            
        #(batch x ndf*4 x 8 x 8) -> (batch x ndf*8 x 4 x 4) 
        self.conv4 = nn.Sequential(
            nn.Conv2dLocal(8, 8, ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),
            )
            
        #(batch x ndf*8 x 4 x 4) -> (batch x 1 x 1 x 1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf*8, 1 ,4, 1, 0, bias=False),
            nn.Sigmoid(),
            )
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)        
        x = self.conv5(x).view(-1)
        return x

#生成器 
class Generator(nn.Module):
    def __init__(self, input_size, ngf):
        super(Generator, self).__init__()
        
        #输入nz维度的噪声
        #(batch x nz x 1 x 1) -> (batch x ngf*8 x 4 x 4) 
        self.luc1 = nn.Sequential(
            LocalUpsampleConvLayer(1, 1, input_size, ngf*8, kernel_size=3, 
                                    stride=1, upsample=4),
            nn.InstanceNorm2d(ngf*4, affine=True),
            nn.ReLU(True),
            )
            
        #(batch x ngf*8 x 4 x 4)  -> (batch x ngf*4 x 8 x 8) 
        self.luc2 = nn.Sequential(
            LocalUpsampleConvLayer(4, 4, ngf*8, ngf*4, kernel_size=3, 
                                    stride=1, upsample=2),
            nn.InstanceNorm2d(ngf*4, affine=True),
            nn.ReLU(True),
            )
            
        #(batch x ngf*4 x 8 x 8)  -> (batch x ngf*2 x 16 x 16)   
        self.luc3 = nn.Sequential(
            LocalUpsampleConvLayer(8, 8, ngf*4, ngf*2, kernel_size=3, 
                                    stride=1, upsample=2),
            nn.InstanceNorm2d(ngf*2, affine=True),
            nn.ReLU(True),
            )
            
        #(batch x ngf*2 x 16 x 16)  -> (batch x ngf x 32 x 32)
        self.uc1 = nn.Sequential(
            UpsampleConvLayer(ngf*2, ngf, kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(64,affine=True),
            nn.ReLU(True),
            )
            
        #(batch x ngf x 32 x 32)  -> (batch x 3 x 96 x 96)
        self.uc2 = nn.Sequential(
            UpsampleConvLayer(ngf, 3, kernel_size=5, stride=1, upsample=3),
            nn.Tanh(),
            )
    def forward(self, x):
        x = self.luc1(x)
        x = self.luc2(x)
        x = self.luc3(x)
        x = self.uc1(x)
        x = self.uc2(x)
        return x
        
        


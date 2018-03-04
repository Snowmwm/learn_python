#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import numpy as np

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
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),
            )
            
        #(batch x ndf*4 x 8 x 8) -> (batch x ndf*8 x 4 x 4)    
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
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

#生成器 1
class Generator(nn.Module):
    def __init__(self, input_size, ngf):
        super(Generator, self).__init__()
        
        #输入nz维度的噪声
        #(batch x nz x 1 x 1) -> (batch x ngf*8 x 4 x 4) 
        self.ct1 = nn.Sequential(
            nn.ConvTranspose2d(input_size, ngf*8 ,4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            )
            
        #(batch x ngf*8 x 4 x 4)  -> (batch x ngf*4 x 8 x 8) 
        self.ct2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4 ,4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            )
            
        #(batch x ngf*4 x 8 x 8)  -> (batch x ngf*2 x 16 x 16) 
        self.ct3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2 ,4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            )
            
        #(batch x ngf*2 x 16 x 16)  -> (batch x ngf x 32 x 32) 
        self.ct4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf ,4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            )

        #(batch x ngf x 32 x 32)  -> (batch x 3 x 96 x 96) 
        self.ct5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3 ,5, 3, 1, bias=False),
            nn.Tanh(),
            )

    def forward(self, x):
        x = self.ct1(x)
        x = self.ct2(x)
        x = self.ct3(x)
        x = self.ct4(x)
        x = self.ct5(x)
        return x
        
        
#生成器 2
class Generator_2(nn.Module):
    def __init__(self, input_size, ngf):
        super(Generator_2, self).__init__()
        
        #输入nz维度的噪声
        #(batch x nz x 1 x 1) -> (batch x ngf*8 x 4 x 4) 
        self.ct1 = nn.Sequential(
            nn.ConvTranspose2d(input_size, ngf*8 ,4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            )
            
        #(batch x ngf*8 x 4 x 4)  -> (batch x ngf*4 x 8 x 8) 
        self.ct2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4 ,4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            )
            
        #(batch x ngf*4 x 8 x 8)  -> (batch x ngf*2 x 16 x 16)   
        self.uc3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2 ,4, 2, 1, bias=False),
            nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            )
            
        #(batch x ngf*2 x 16 x 16)  -> (batch x ngf x 32 x 32)
        self.uc4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf ,4, 2, 1, bias=False),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            )
            
        #(batch x ngf x 32 x 32)  -> (batch x 3 x 96 x 96)
        self.uc5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, ngf, 5, 3, 1, bias=False),
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            )
    def forward(self, x):
        x = self.ct1(x)
        x = self.ct2(x)
        x = self.uc3(x)
        x = self.uc4(x)
        x = self.uc5(x)
        return x
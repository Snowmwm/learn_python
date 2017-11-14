#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import os
import numpy as np


def get_imlist(path):
    '''返回目录中所有JPG图像的文件名列表'''
    
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')] #or f.endswith('.png')

    
def imresize(im, sz):
    '''使用PIL对象重新定义图像数组的大小'''
    
    pil_im = Image.fromarray(np.uint8(im))
    return np.array(pil_im.resize(sz))
    
    
def histeq(im, nbr_bins=256):
    '''对一幅灰度图像进行直方图均衡化 
    将图像的灰度直方图变平，使变换后的图像中每个灰度值的分布概率都相同。'''
    
    #计算图像的直方图
    #matplotlib的hist自动绘制直方图,而numpy的histogram只产生数据
    #flatten将array数组变成一维
    #normed=True使得返回的结果在范围内积分为1
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)  

    cdf = np.cumsum(imhist) #累计分布函数
    cdf = 255 * cdf / cdf[-1] #归一化

    #使用累计分布函数的线性插值，计算新的像素值
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

    
def compute_average(imlist):
    '''计算图像列表的平均图像'''
    
    #打开第一幅图像，将其存储在浮点型数组中
    average_im = np.array(Image.open(imlist[0]), 'f')
    
    for imname in imlist[1:]:
        try:
            average_im += np.array(Image.open(imname))
        except:
            print(imname, '...skipped')
    average_im /= len(imlist)
    
    #返回uint8类型的平均图像
    return np.array(average_im, 'uint8')
   
   
def u_int8(im):
    '''将带符号数据缩放为0-255范围的整数'''
    
    return ((im/max(im.max(), abs(im.min())) + 1)*255/2).astype('uint8')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import filters
import os
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pickle
from scipy.misc import imsave

import imtools as it

ImageFile = 'test.png'
SaveFile = 'test1.png'

#利用数组操作图像
'''
im1 = np.array(Image.open(ImageFile).convert('L'))
im2 = 255-im1 #对图像进行反相处理
im3 = (100.0/255) * im1 + 100 #将图像像素值变换到100——200区间
im4 = 255.0 * (im1 / 255.0)**2 #对图像像素值求平方后得到的图像(使暗的更暗)
im5, cdf = it.histeq(im) #直方图均衡化
im6 = it.compute_average(it.get_imlist('C:\\PTW'))#图像平均

out1 = Image.fromarray(np.uint8(im1))
out2 = Image.fromarray(np.uint8(im2))
out3 = Image.fromarray(np.uint8(im3))
out4 = Image.fromarray(np.uint8(im4))
out5 = Image.fromarray(np.uint8(im5))
out6 = Image.fromarray(np.uint8(im6))
out1.save('test1.jpg')
out2.save('test2.jpg')
out3.save('test3.jpg')
out4.save('test4.jpg')
out5.save('test5.jpg')
out6.save('test6.jpg')
'''

#计算图像主成份
'''
import PCA

imlist = it.get_imlist('C:\\PTW\\learn_python')
im = np.array(Image.open(imlist[0]))
m, n = im.shape[0:2] #获取图像大小
imnbr = len(imlist) #获取图像数目

#创建矩阵，保存所有压平后的图像数据
im_matrix = np.array([np.array(Image.open(im)).flatten() for im in imlist], 'f')

#PCA
V, S, im_mean = pca.pca(im_matrix)

#显示均值图像和前6个模式
fig = plt.figure()
plt.gray()
plt.subplot(2, 4, 1)
plt.imshow(im_mean.reshape(m, n))
for i in range(imnbr):
    plt.subplot(2, 4, 2+i)
    plt.imshow(V[i].reshape(m,n))
    
plt.show()

# 保存均值和主成分数据
with open('pca_models.pkl', 'wb') as f:
    pickle.dump(im_mean, f)
    pickle.dump(V, f)
    

'''
'''
# 载入均值和主成分数据
with open('pca_models.pkl', 'rb') as f:
    im_mean = pickle.load(f) # 载入对象顺序必须和保存顺序一样
    V = pickle.load(f)
#显示均值图像和前6个模式
fig = plt.figure()
plt.gray()
plt.subplot(2, 4, 1)
plt.imshow(im_mean.reshape(640, 640))
for i in range(6):
    plt.subplot(2, 4, 2+i)
    plt.imshow(V[i].reshape(640,640))
    
plt.show()
'''


#图像高斯模糊
'''
im1 = np.array(Image.open(ImageFile))
im2 = np.zeros(im1.shape)
for i in range(3):
    im2[:,:,i] = filters.gaussian_filter(im1[:,:,i], 4*i+2)
    
im3 = im1 - im2
im2[:,:,3] = im1[:,:,3]

out2 = Image.fromarray(np.uint8(im2))
out2.save('gaussian_filter.png')
out3 = Image.fromarray(np.uint8(im3))
out3.save('unsharp_masking.png')
'''

#图像导数
'''
sigma = 10 # 标准差

im = np.array(Image.open('jzg.jpg').convert('L'))
imx = np.zeros(im.shape)
imy = np.zeros(im.shape)
img = np.zeros(im.shape)

filters.gaussian_filter(im, (sigma,sigma), (0,1), imx) #计算高斯导数
filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
img = np.sqrt(imx**2 + imy**2)
#\\\\\\\\\\\\\\\\\\\\\\\\
im = np.array(Image.open('jzg.jpg'))
imx = np.zeros(im.shape)
imy = np.zeros(im.shape)
img = np.zeros(im.shape)

for i in range(3):
    filters.sobel(im[:,:,i],1,imx[:,:,i]) #Sobel导数滤波器
    filters.sobel(im[:,:,i],0,imy[:,:,i])
    img[:,:,i] = np.sqrt(imx[:,:,i]**2 + imy[:,:,i]**2)

imx = it.u_int8(imx)
imy = it.u_int8(imy)
img = it.u_int8(img)
outx = Image.fromarray(np.uint8(imx))
outy = Image.fromarray(np.uint8(imy))
outg = Image.fromarray(np.uint8(img))
outx.save('test_x.jpg')
outy.save('test_y.jpg')
outg.save('test_g.jpg')
'''

#ROF去噪
'''
import ROF

# 使用噪声创建合成图像
im = np.zeros((500,500))
im[100:400,100:400] = 128
im[200:300,200:300] = 255
im = im + 30*random.standard_normal((500,500))
U,T = ROF.denoise(im,im)
G = filters.gaussian_filter(im,10)

# 保存生成结果
imsave('synth_im.jpg',im)
imsave('synth_T.jpg',T)
imsave('synth_rof.jpg',U)
imsave('synth_gaussian.jpg',G)
#\\\\\\\\\\\\\\\\\\\\\\\\

#给图片去噪
im = np.array(Image.open('jzg.jpg').convert('L'))
im = im + 20*random.standard_normal((241,241))
U,T = ROF.denoise(im,im)
G = filters.gaussian_filter(im,2)

# 保存生成结果
imsave('synth_im.jpg',im)
imsave('synth_T.jpg',T)
imsave('synth_rof.jpg',U)
imsave('synth_gaussian.jpg',G)
'''
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
    
    
def pca(X):
    """主成份分析
    输入：矩阵X，其中该矩阵中存储训练数据，每一行为一条训练数据
    返回：投影矩阵（按照维度的重要性排序）、方差、均值"""
    
    #X = np.asarray(X)
    #获取样本数n、维度m   
    n, m = X.shape  
    #数据中心化
    mean_X = np.mean(X, axis=0)
    X -= mean_X  #X-E(X)
    
    if m > n:
        #维度大于样本数,使用紧致技巧
        '''对于任意矩阵X,X(T)X的特征值就称为X的奇异值
        XX(T)和X(T)X有完全一致的特征分解
        假设方阵X=Q∑Q(T) 则：X(T)X=(Q∑Q(T))(Q∑Q(T))=Q∑∑Q(T)
        所以X(T)X的特征值就是X的奇异值,恰好为X的特征值的(模长的)平方'''
        
        #协方差矩阵
        M = np.dot(X,X.T) #这是一个nxn的对称矩阵
        
        #利用eigh来求协方差矩阵的特征值E和特征向量EV
        E, EV = np.linalg.eigh(M) 
        tmp = np.dot(X.T, EV).T
        V = tmp[::-1] #按降序排列
        S = np.sqrt(E)[::-1]
        for i in range(V.shape[1]):
            V[:,i] = (V[:,i].T/np.array(S)).T #正交化
        
    else:
        #样本数大于维度，使用奇异值分解
        U, S, V = np.linalg.svd(X)
        V = V[:n] #只返回前n维的数据
        
    return V, S, mean_X
   
   
def u_int8(im):
    '''将带符号数据缩放为0-255范围的整数'''
    
    return ((im/max(im.max(), abs(im.min())) + 1)*255/2).astype('uint8')
    
    
def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    '''Rudin-Osher-Fatemi（ROF）去噪模型
    输入：含有噪声的输入图像(灰度图像)、U的初始值、步长、停业条件、TV正则项权值
    输出：去噪和去除纹理后的图像U、纹理残余'''
    
    m, n = im.shape #噪声图像大小
    
    #初始化
    U = U_init
    Px = im #对偶域的x分量
    Py = im #对偶域的y分量
    error = 1
    
    while error > tolerance:
        Uold = U
        
        #原始变量U的梯度
        GradUx = np.roll(U, -1, axis=1)-U
        GradUy = np.roll(U, -1, axis=0)-U
        
        #更新对偶变量
        PxNew = Px + (tau/tv_weight)*GradUx
        PyNew = Py + (tau/tv_weight)*GradUy
        NormNew = np.maximum(1, np.sqrt(PxNew**2 +PyNew**2))
        
        Px = PxNew/NormNew #更新x分量
        Py = PyNew/NormNew #更新y分量
        
        #更新原始变量
        RxPx = np.roll(Px, 1, axis=1) #对x分量沿x轴向右平移
        RyPy = np.roll(Py, 1, axis=1)
        
        DivP = (Px-RxPx)+(Py-RyPy) #对偶域的散度
        U = im + tv_weight*DivP
        
        #更新误差
        error = np.linalg.norm(U-Uold)/np.sqrt(n*m)
        print(error)
        
    return U, im-U
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
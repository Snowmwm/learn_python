#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np

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
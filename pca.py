#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np

def pca(X):
    '''主成份分析
    输入：矩阵X，其中该矩阵中存储训练数据，每一行为一条训练数据
    返回：投影矩阵（按照维度的重要性排序）、方差、均值'''
    
    X = numpy.asarray(X)
    #获取维度   
    num_data, dim = X.shape
    
    #数据中心化
    mean_X = X.np.mean(axis=0)
    X -= mean_X
    
    if dim > num_data:
        #维度大于样本数,使用紧致技巧
        M = np.dot(X,X.T) 
        e, EV = np.linalg.eigh(M) #特征值和特征向量
        tmp = np.dot(X.T, EV).T
        V = tmp[::-1]
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np

def pca(X):
    """主要成分分析
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
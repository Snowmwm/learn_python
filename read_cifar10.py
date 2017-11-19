#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#from scipy.misc import imsave

#与标签对应的类别
'''
calsses = ('plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')
'''
#用torchvision读取数据    
'''
#数据预处理
transform=transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
#ToTensor把PIL.Image(RGB)或numpy.ndarray(HxWxC)的值从0到255映射到0到1的范围内，并转化成Tensor格式。
#Normalize(mean,std)通过公式:channel=(channel-mean)/std 实现数据归一化 
                    
#利用torchvision加载cifar10数据集 
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=False, transform=transform)

#使用Dataloader封装数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False)


#读取一张图片作为测试
def imshow(im):
    im = im / 2 + 0.5 #反归一化
    npim = im.numpy()
    plt.imshow(np.transpose(npim, (1,2,0))) #把channel那一维放到最后
    plt.show()
    
dataiter = iter(trainloader)  
images, labels = dataiter.next() #用迭代器从dataloader中读取图片和标签
imshow(torchvision.utils.make_grid(images))
'''        
        
#读取文件存入字典
'''
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        return dict
'''
    
#尝试读取100个图片并保存下来
'''
path = './cifar10/cifar-10-batches-py/data_batch_1'    
dict = unpickle(path)
#print(dict.keys())
for i in range(100):
    image = np.frombuffer(dict[b'data'][i], dtype=np.uint8).reshape(3, 32, 32)
    image = np.transpose(image, (1,2,0))
    name = calsses[dict[b'labels'][i]]
    imsave(os.path.join('./cifar10/image_samples', '%s%d.jpg' % (name,i)), image)
'''

#分批读取cifar10数据
def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
        X = dict['data']
        Y = dict['labels']
        X = X.reshape(10000, 3, 32, 32).astype(float)#.transpose(0,2,3,1)
        Y = np.array(Y)
        return X, Y
              
def load_cifar10(path):   
    """
    输入：CIFAR10数据集路径
    输出：拼接好测试集和训练集
    """        
    xs, ys = [], []
    for i in range(1, 6):
        file = os.path.join(path, 'data_batch_%d'% i)
        X, Y = load_cifar_batch(file)
        xs.append(X)
        ys.append(Y)
    trX = np.concatenate(xs) #concatenate函数能够一次完成多个数组的拼接
    trY = np.concatenate(ys)
    del X, Y
    teX, teY = load_cifar_batch(os.path.join(path,'test_batch'))
    return trX, trY, teX, teY
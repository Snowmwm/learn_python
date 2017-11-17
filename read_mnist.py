#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import struct
import numpy as np

#读取MNIST数据集
'''
#先使用二进制方式把文件都读进来
filepath = 'c:/PTW/learn_python/mnist/train-images.idx3-ubyte' 
binfile = open(filepath, 'rb')
buf = binfile.read()

#使用struct模块处理二进制文件
index = 0 #index记录已经读取的到位置
magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
#'>IIII'是说使用大端法读取4个unsinged int32
#大端模式，是指数据的高字节保存在内存的低地址中，而数据的低字节保存在内存的高地址中
index += struct.calcsize('>IIII')
#calcsize函数把格式化字符串作为参数传给该函数，即可返回出应读取内容的长度

#读取一个图片测试是否读取成功
im = struct.unpack_from('>784B' ,buf, index)  #28x28=784
index += struct.calcsize('>784B')

im = np.array(im)
im = im.reshape(28,28)

fig = plt.figure()
plotwindow = fig.add_subplot(111)
plt.imshow(im , cmap='gray')
plt.show()


#读取100个图片保存下来

from scipy.misc import imsave

for i in range(100):
    im = struct.unpack_from('>784B' ,buf, index)  #28x28=784
    index += struct.calcsize('>784B')
    im = np.array(im)
    im = im.reshape(28,28)
    imsave('c:/PTW/learn_python/mnist/image_samples/im_%s.jpg' % i, im)
'''


'''        
def read_image_file(path):
    with open(path, 'rb') as binfile:
        buf = binfile.read() 
        index = 0 
        images = []
        magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
        index += struct.calcsize('>IIII')

        for i in range(numImages):
            im = struct.unpack_from('>784B', buf, index) 
            index += struct.calcsize('>784B')
            images.append(im)
            
        assert len(images) == numImages
        return torch.ByteTensor(images).view(-1, 28, 28)
"""
path = 'c:/PTW/learn_python/mnist/train-images.idx3-ubyte' 
print(read_image_file(path).size())
img = read_image_file(path).numpy()[0,:,:]
fig = plt.figure()
plotwindow = fig.add_subplot(111)
plt.imshow(img , cmap='gray')
plt.show()
"""
       
def read_label_file(path):
    with open(path, 'rb') as binfile:
        buf = binfile.read() 
        index = 0
        labels = []
        magic, numItems = struct.unpack_from('>II', buf, index)
        index += struct.calcsize('>II')
        
        for i in range(numItems):
            it = struct.unpack_from('>B', buf, index) 
            index += struct.calcsize('>B')
            labels.append(it)
            
        assert len(labels) == numItems
        return torch.LongTensor(labels)

        
def read_mnist(path):
    """
    输入：MNIST数据集路径
    输出：拼接好测试集和训练集
    """   
    train_set = (
        read_image_file(os.path.join(path, 'train-images.idx3-ubyte')), 
        read_label_file(os.path.join(path, 'train-labels.idx1-ubyte'))
        )
    test_set = (
        read_image_file(os.path.join(path, 't10k-images.idx3-ubyte')), 
        read_label_file(os.path.join(path, 't10k-labels.idx1-ubyte'))
        )
    """    
    with open(os.path.join(path, 'training.pt'), 'wb') as f:
        torch.save(training_set, f)
    with open(os.path.join(path, 'test.pt'), 'wb') as f:
        torch.save(test_set, f)
    """    
    return train_set, test_set
"""
path = 'c:/PTW/learn_python/mnist'
train_set, test_set = read_mnist(path)
"""
'''

def load_file(path):
    with open(path, 'rb') as binfile:
        buf = binfile.read() 
        magic, length, row, col = struct.unpack_from('>IIII', buf, 0)
        loaded = np.frombuffer(buf, dtype=np.uint8)
        if int(magic) == 2051: 
            return loaded[16:].reshape((length, 1, row, col)).astype(float)
        else:
            return loaded[8:].reshape((length))
            
def load_mnist(path):
    trX = load_file(os.path.join(path, 'train-images.idx3-ubyte'))
    trY = load_file(os.path.join(path, 'train-labels.idx1-ubyte'))
    teX = load_file(os.path.join(path, 't10k-images.idx3-ubyte'))
    teY = load_file(os.path.join(path, 't10k-labels.idx1-ubyte'))
    """
    trX /= 255.
    teX /= 255.
    trY = np.asarray(trY)
    teY = np.asarray(teY)
    """
    return trX, teX, trY, teY
'''
path = 'c:/PTW/learn_python/mnist'
trX, teX, trY, teY = load_mnist(path)
print(trX.shape, trY.shape, teX.shape, teY.shape)
'''





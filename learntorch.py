#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F #激励函数都在这里
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

#GPU加速测试
'''
#cuda test
y = torch.Tensor([1,0])
yy = y.cuda()
print(yy)

#cudnn test
from torch.backends import cudnn
print(cudnn.is_acceptable(yy))
'''

#数据转换
'''
#numpy转torch
np_data = np.array([[1,2,3],[6,7,8]])
th_data = torch.from_numpy(np_data)

#torch转numpy
th_data = torch.FloatTensor(2,3)
np_data = th_data.numpy()
'''

#torch运算
'''
np_data = np.array([[1,2],[6,8]])
th_data = torch.from_numpy(np_data)
print(th_data)
print(torch.mm(th_data,th_data))
print(np.dot(np_data,np_data))
'''

#Variable:一个存放tensor数据的变量
'''
#将tensor放入Variable
#其中requires_grad代表是否参与误差反向传播, 要不要计算梯度
tensor = torch.FloatTensor(2,2)
variable = Variable(tensor, requires_grad=True)
print(variable)

print(variable.data) #获取tensor形式数据
print(variable.data.numpy()) #获取numpy形式数据

#计算梯度
v_out = torch.mean(variable**2)
v_out.backward()  #模拟v_out的误差反向传播
print(variable.grad)  #显示Variable的梯度输出结果
'''

#激励函数
'''
# 做一些假数据
x = torch.linspace(-5, 5, 200)
x = Variable(x)
x_np = x.data.numpy()   # 换成 numpy array
#x是Variable数据，F.relu(x)也是返回Variable数据，然后.data获取 tensor 数据

y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy() #f(x)=ln(1+ex)
#y_softmax = F.softmax(x)  
#softmax比较特殊,不能直接显示, 不过他是关于概率的, 用于分类
'''

#回归问题
'''
#创建假数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  
# x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 
# noisy y data (tensor), shape=(100, 1)

x, y = torch.autograd.Variable(x), Variable(y) #将tensor放入Variable

#建立神经网络
class Net(nn.Module):  # 继承 torch 的 Module

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        # 只有一个隐层
        self.hidden = nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.predict = nn.Linear(n_hidden, n_output)   # 输出层线性输出
        
    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 因为是预测，最后一层不需要激励函数
        return x

#神经网络结构        
net = Net(n_feature=1, n_hidden=10, n_output=1)

#定义优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
#传入net的所有参数, 学习率lr

#定义损失函数
loss_func = nn.MSELoss() # 预测值和真实值的误差计算公式 (均方差)

#开始训练
for t in range(100):
    prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值
    loss = loss_func(prediction, y)     # 计算两者的误差
    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
'''

#分类问题
'''
#创建假数据
#x0和x1分别是两个类别的数据，y0和y1是对应的标签
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, 1)

#拼接数据
#注意x,y数据的数据形式是一定要像下面一样
x = torch.cat((x0, x1), ).type(torch.FloatTensor)  
# FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    
# LongTensor = 64-bit integer

x, y = Variable(x), Variable(y)

#建立神经网络
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() 
        self.hidden = nn.Linear(n_feature, n_hidden) 
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self,x):
        x = F.relu(self.hidden(x))   
        x = self.out(x)       # 不是预测值，预测值还要再另外计算
        return x
        
#神经网络结构 
net = Net(n_feature=2, n_hidden=10, n_output=2)

#优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
#传入net的所有参数, 学习率lr

#定义损失函数
loss_func = nn.CrossEntropyLoss() #交叉熵损失函数默认调用log_softmax函数

#训练
for t in range(100):
    out = net(x)            # 喂给 net 训练数据 x, 输出分析值
    loss = loss_func(out, y)     # 计算两者的误差
    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上

#预测
#过了一道softmax激励函数后的最大概率才是预测值
prediction = torch.max(F.softmax(out), 1)[1]
'''

#快速搭建神经网络
'''
net2 = nn.Sequential(
    nn.Linear(1,10),
    nn.ReLU(),
    nn.Linear(10,1)
    )
'''

#保存/提取
'''
torch.save(net1, 'net.pkl') 
#保存整个网络，net1就是定义的网络
torch.save(net1.state_dict(), 'net_params.pkl') 
#只保存网络中的参数 (速度快, 占内存少)


net2 = torch.load('net.pkl') 
#提取整个网络
prediction = net2(x)

net3.load_state_dict(torch.load('net_params.pkl'))
#只提取网络参数(首先需要定义一样的神经网络)
prediction = net3(x)
'''

'''
#数据加载器DataLoader
#1、将tensor数据转为torch能识别的Dataset
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
#2、把dataset放入DataLoader
loader = Data.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # BATCH_SIZE是我们设定的batch大小
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
        )

#批训练SGD
for epoch in range(3):   # 训练所有数据 3 次
    for step, (batch_x, batch_y) in enumerate(loader):  
        # 每一步 loader 释放一小批数据用来学习
        # 假设这里就是你训练的地方...
        # 打出来一些数据
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())
              
              
        #这里还是tensor数据，真正训练时还要放到Variable中
        b_x = Variable(batch_x)  
        b_y = Variable(batch_y)


#优化器optimizer
opt_SGD      = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
#动量加速，在SGD函数里指定momentum的值即可
opt_RMSprop  = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam     = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
'''








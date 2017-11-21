#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import math 
from PIL import Image

#超参数设置
EPOCH = 100
BATCH_SIZE = 32
LR = 0.001
MOMENTUM = 0.9
SEED = 666
DECAY = 0.04

#随机种子设置
use_gpu = torch.cuda.is_available()
torch.manual_seed(SEED)
if use_gpu:
    torch.cuda.manual_seed(SEED)

#用torchvision读取数据    
#数据预处理
transform=transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
#ToTensor把PIL.Image(RGB)或numpy.ndarray(HxWxC)的值从0到255映射到0到1的范围内，并转化成Tensor格式。
#Normalize(mean,std)通过公式:channel=(channel-mean)/std 实现数据归一化 
                    
#利用torchvision加载cifar10数据集 
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=False, transform=transform)

#使用Dataloader封装数据
train_loader = Data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

#每一epoch中的mini-batch数
batch_num = math.floor(len(trainset) / BATCH_SIZE)  

#W初始化
def weight_init(m):
# 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
# m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()

   
#建立卷积神经网络

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input shape (3, 32, 32)
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # output shape (32, 32, 32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # output shape (64, 32, 32)
        self.maxpool = nn.MaxPool2d(3, 2)
        # output shape (64, 15, 15)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 96, 5, 1, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        # output shape (96, 15, 15)
        self.avgpool = nn.AvgPool2d(3, 2)
        # output shape (96, 7, 7)
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # output shape (128, 7, 7)
        # avgpool
        # output shape (128, 3, 3)
        self.fc = nn.Linear(128*3*3, 128)
        nn.init.normal(self.fc.weight, mean=0, std=0.003)
        
        self.bn = nn.BatchNorm2d(128)
        
        self.out = nn.Linear(128, 10)
        nn.init.normal(self.out.weight, mean=0, std=0.025)
    
    def forward(self, x):
        x = self.pre_layers(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) #展平
        x = self.fc(x)
        x = F.relu(self.bn(x))
        #x = F.dropout(x, training=self.training) #随机将输入张量中的部分元素置0
        output = self.out(x)
        return output      

def train(epoch):
    cnn.train()#把module设成training模式，对Dropout和BatchNorm有影响
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        #获取输入和标签
        inputs, labels = data
        #转成Variable类型，否则无法进行梯度计算
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        #参数梯度清零
        optimizer.zero_grad()
        #前向传播，损失计算，后向传播
        outputs = cnn(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        #参数更新
        optimizer.step()
        running_loss += loss.data[0]
        if i % batch_num == (batch_num - 1):    
            print('epoch: %d, batch: %5d, loss: %.3f' %
                  (epoch+1, i+1, running_loss / batch_num))
            running_loss = 0.0        
        
def test(epoch):
    cnn.eval()#把module设置为评估模式，只对Dropout和BatchNorm模块有影响
    correct = 0
    total = 0
    for data in test_loader:
        test_x, test_y = data
        if use_gpu:
            test_out = cnn(Variable(test_x.cuda()))
        else:
            test_out = cnn(Variable(test_x))    
        predicted = torch.max(test_out.data, 1)[1]              
        total += test_y.size(0)
        correct += (predicted == test_y.cuda()).sum()                
    print('epoch: %d Accuracy: %.2f %%' % (epoch+1, 100 * correct / total))
    running_loss = 0.0        

    
#训练网络
cnn = CNN()
cnn.apply(weight_init)
#cnn = torch.load('./cifar10/cnn_cifar10_epoch_90.pkl')  

if use_gpu:
    cnn.cuda() #是否启用GPU加速

loss_func = nn.CrossEntropyLoss() #损失函数              
 
for epoch in range(EPOCH):
    learn_rate = LR / (1 + DECAY * epoch) #学习率衰减
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learn_rate) #Adam优化器
    train(epoch)
    test(epoch)
    if epoch % 10 == 9:
        torch.save(cnn, './cifar10/cnn_cifar10_epoch_%d.pkl' % (epoch+1)) #保存网络结构和模型参数
print('Finished Training')
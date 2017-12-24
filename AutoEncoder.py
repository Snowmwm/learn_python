#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from read_mnist import load_mnist

#超参数设置
EPOCH = 10
BATCH_SIZE = 32
LR = 0.005
SEED = 666
N_TEST_IMG = 8


torch.manual_seed(SEED)


#读取数据集
path = 'c:/PTW/learn_python/mnist'
trX, teX, trY, teY = load_mnist(path)

#使用Dataloader封装数据
trX /= 255.
trX = torch.from_numpy(trX).float()
trY = torch.from_numpy(trY).long()
trX = torch.squeeze(trX, 1)

train_data = Data.TensorDataset(data_tensor=trX, target_tensor=trY)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


"""
print(train_data.data_tensor.size())
print(train_data.target_tensor.size())
plt.imshow(train_data.data_tensor[2].numpy(), cmap='gray')
plt.title('%i' %train_data.target_tensor[2])
plt.show()
"""

#建立自编码神经网络
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return encoded, decoded
    
autoencoder = AutoEncoder()
    
#优化器和损失函数
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()


# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = Variable(train_data.data_tensor[:N_TEST_IMG].view(-1, 28*28))#.type(torch.FloatTensor)/255.)
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(()) 
    a[0][i].set_yticks(())
    

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28*28))
        b_y = Variable(x.view(-1, 28*28))
        b_label = Variable(y)

        encoded, decoded = autoencoder(b_x)
        loss = loss_func(decoded, b_y)
        optimizer.zero_grad() #参数梯度清零
        loss.backward() #后向传播
        optimizer.step() #参数更新

        if step % 100 == 0:
            print('Epoch:', epoch, '|train loss: %.4f' %loss.data[0])
            
            # plotting decoded image (second row)
            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.05)

plt.ioff()
plt.show()

# visualize in 3D plot
view_data = Variable(train_data.data_tensor[:200].view(-1, 28*28))#.type(torch.FloatTensor)/255.)
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2)
ax = Axes3D(fig)
X = encoded_data.data[:, 0].numpy()
Y = encoded_data.data[:, 1].numpy()
Z = encoded_data.data[:, 2].numpy()
values = train_data.target_tensor[:200].numpy()

for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9))
    ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max()) 
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())

plt.show()

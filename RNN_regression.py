#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

#超参数设置
LR = 0.02
SEED = 666
TIME_STEP = 10 
INPUT_SIZE = 1 

#随机种子设置
torch.manual_seed(SEED)

#生成数据及可视化
"""
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32) #0——2π之间取100个点
x_np = np.sin(steps)
y_np = np.cos(steps)

plt.plot(steps, y_np, 'r-', label='target (cos)')
plt.plot(steps, x_np, 'b-', label='target (sin)')
plt.legend(loc='best')
plt.show()
"""

#建立循环神经网络
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True, #batch放在第一个维度(batch, time_step, input_size)
        )
        
        self.out = nn.Linear(32, 1)
        
    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        
        return torch.stack(outs, dim=1), h_state

rnn = RNN()

#优化器和损失函数
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

#训练
h_state = None
for step in range(120):
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    
    # shape (batch, time_step, input_size)
    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))
    
    prediction, h_state = rnn(x, h_state)
    h_state = Variable(h_state.data) #!!!用Variable重新包装h_state

    loss = loss_func(prediction, y)     # cross entropy loss
    optimizer.zero_grad()               # clear gradients for this training step
    loss.backward()                     # backpropagation, compute gradients
    optimizer.step()                    # apply gradients

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()


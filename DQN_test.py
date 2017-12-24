#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym


#超参数设置
BATCH_SIZE = 32
LR = 0.01

EPSILON = 0.9 #选择最优动作的百分比
GAMMA = 0.9 #奖励递减参数
TARGET_REPLACE_ITER = 100 #Q现实网络的更新频率
MEMORY_CAPACITY = 2000 #记忆库大小

env = gym.make('CartPole-v0') #立杆子游戏
env = env.unwrapped
N_ACTIONS = env.action_space.n #杆子能做的动作
N_STATES = env.observation_space.shape[0] #杆子能获取的环境信息数

ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  


#建立神经网络框架
class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)   # 权重初始化
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

#建立DQN学习体系
class DQN(object):
    def __init__(self):
        #通过神经网络框架建立target net和eval net
        self.eval_net = NET()
        self.target_net = NET()

        self.learn_step_counter = 0 #用于target更新计时
        self.memory_counter = 0 #记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))#初始化记忆库
        
        #优化器和损失函数
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    
        self.loss_func = nn.MSELoss()   
        
        
    def choose_action(self, x):
    # 根据环境观测值选择动作的机制
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # 这里只输入一个 sample
        if np.random.uniform() < EPSILON:   # 选最优动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:   # 选随机动作
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action
        
    def store_transition(self, s, a, r, s_):
    # 存储记忆
        transition = np.hstack((s, [a, r], s_)) #hstack就是把数据水平连接起来
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        
    def learn(self):
    # 学习记忆库中的记忆
        # target 网络更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        #随机抽取记忆库中的batch数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(np.int64)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))
        
        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     
        #target net不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        #max函数返回的第一个数是最大值，第二个数是索引
        loss = self.loss_func(q_eval, q_target)
        
        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
dqn = DQN()


#学习过程
for i_episode in range(400):
    s = env.reset() #当前状态
    ep_r = 0
    while True:
        env.render() #渲染环境
        
        a = dqn.choose_action(s) #根据当前状态选择动作
        
        s_, r, done, info = env.step(a) #根据行为从环境中获得反馈
        
        # 修改reward,杆子越正、车越靠近中间, R越大, 使DQN快速学习
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        
        dqn.store_transition(s, a, r, s_) #存储状态、动作、奖励、下一个状态
        
        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))
            
        if done:
            break
        s = s_






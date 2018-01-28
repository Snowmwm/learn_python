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
SEED = 666
BATCH_SIZE = 32
LR = 0.01

EPSILON = 0.9 #选择最优动作的百分比
GAMMA = 0.9 #奖励递减参数
TARGET_REPLACE_ITER = 100 #Q现实网络的更新频率
MWMORY_CAPACITY = 2000 #记忆库大小

env = gym.make('CartPole-v0') #立杆子游戏
env = env.unwarapped
N_ACTIONS = env.action_space.n #杆子能做的动作
N_STATES = enc.observation_space.shape[0] #杆子能获取的环境信息数


#建立DQN神经网络
class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        
    def forward(self, x):
        pass
        
        return encoded, decoded


#建立DQN学习体系
class DQN(object):
    def __init__(self):
    # 建立 target net 和 eval net 还有 memory
        pass
        
    def choose_action(self, x):
    # 根据环境观测值选择动作的机制
        pass
        
    def store_transition(self, s, a, r, s_):
    # 存储记忆
        pass
        
    def learn(self):
    # target 网络更新
    # 学习记忆库中的记忆
        pass
        
dqn = DQN()


#学习过程
for i_episode in range(400):
    s = env.reset() #当前状态
    while True:
        env.render() #渲染环境
        
        a = dqn.choose_action(s) #根据当前状态选择动作
        
        s_, r, done, info = env.step(a) #根据行为从环境中获得反馈
        
        dqn.store_transition(s, a, r, s_) #存储状态、动作、奖励、下一个状态
        
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            
        if done:
            break
        s = s_






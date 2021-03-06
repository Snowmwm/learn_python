#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#超参数
DNA_SIZE = 1 #DNA的长度
N_GENERATIONS = 200 #繁衍的代数
DNA_BOUND = [0, 5] #DNA取值范围
MUT_STRENGHT = 5

def F(x):  #目标函数
    return np.sin(10*x)*x + np.cos(2*x)*x

def get_fitness(pred): #在环境中的适应度
    return pred.flatten()

def make_kid(parent):
    k = parent + MUT_STRENGHT * np.random.randn(DNA_SIZE) #没有crossover
    k = np.clip(k, *DNA_BOUND)
    return k

def kill_bad(parent, kid):
    global MUT_STRENGHT
    fp = get_fitness(F(parent))[0]
    fk = get_fitness(F(kid))[0]
    p_target = 0.2
    if fp < fk:
        parent = kid
        ps = 1.
    else:
        ps = 0.
    MUT_STRENGHT *= np.exp(1/np.sqrt(DNA_SIZE+1) * (ps-p_target)/(1-p_target)) 
    return parent

parent = 5 * np.random.rand(DNA_SIZE)    
    
plt.ion()
x = np.linspace(*DNA_BOUND, 200)

for _ in range(N_GENERATIONS):
    kid = make_kid(parent)
    py, ky = F(parent), F(kid)
    parent = kill_bad(parent, kid)

    plt.cla()
    plt.scatter(parent, py, s=128, lw=0, c='red', alpha=0.5)
    plt.scatter(kid, ky, s=128, lw=0, c='blue', alpha=0.5)
    plt.text(0, -7, 'Mutation strength=%.2f' %MUT_STRENGHT)
    plt.plot(x, F(x))
    plt.pause(0.01)
    
plt.ioff()
plt.show()















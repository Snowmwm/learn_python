#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#超参数
DNA_SIZE = 10 #DNA的长度
POP_SIZE = 100 #种群大小
CROSS_RATE = 0.8 #交叉配对概率
MUTATION_RATE = 0.003 #变异概率
N_GENERATIONS = 200 #繁衍的代数
X_BOUND = [0, 5] #X取值范围


def F(x):  #目标函数
    return np.sin(10*x)*x + np.cos(2*x)*x

    
def get_fitness(pred): #在环境中的适应度
    return pred + 1e-3 - np.min(pred)
    
def translateDNA(pop): #(翻译与表达DNA)将二进制转化成十进制
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) /(float(2 ** DNA_SIZE -1))* X_BOUND[1]
    
def select(pop, fitness): #自然选择
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                        p=fitness/fitness.sum()) 
    return pop[idx]
    
def crossover(parent, pop): #基因交叉配对
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)
        parent[cross_points] = pop[i_, cross_points]
    return parent
    
def mutate(child): #变异
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child
    
    
#pop = np.random.randint(0, 2, (1, DNA_SIZE)).repeat(POP_SIZE, asis=0)
pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE)) #随机产生DNA data
#[[0101000101],
# [1010100111],
# ...]

plt.ion()
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))
    
for _ in range(N_GENERATIONS):
    F_values = F(translateDNA(pop))
    
    if 'sca' in globals(): 
        sca.remove()
    sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5)
    plt.pause(0.05)
    
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child
    
plt.ioff()
plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
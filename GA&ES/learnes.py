#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#超参数
DNA_SIZE = 1 #DNA的长度
POP_SIZE = 100 #种群大小
N_GENERATIONS = 200 #繁衍的代数
DNA_BOUND = [0, 5] #DNA取值范围
N_KID = 50 #每次繁衍产生的子代个体数


def F(x):  #目标函数
    return np.sin(10*x)*x + np.cos(2*x)*x


def get_fitness(pred): #在环境中的适应度
    return pred.flatten()


def make_kid(pop, n_kid):
    kids = {}
    kids['DNA'] = np.empty((n_kid, DNA_SIZE))
    kids['mut_strength'] = np.empty_like(kids['DNA'])

    for kv, ks in zip(kids['DNA'], kids['mut_strength']):
        #crossover
        p1, p2 = np.random.choice(np.arange(POP_SIZE), size=2, replace=False)#从种群中随机选择父母p1 p2
        cp = np.random.randint(0,2,DNA_SIZE,dtype=np.bool) #选择DNA中交叉配对的点,因为DNA长度为1,所以实际上并没有交叉
        kv[cp] = pop['DNA'][p1, cp]
        kv[~cp] = pop['DNA'][p2, ~cp]
        ks[cp] = pop['mut_strength'][p1, cp]
        ks[~cp] = pop['mut_strength'][p2, ~cp]

        #mutate
        ks[:] = np.maximum(ks + (np.random.rand(*ks.shape)-0.5), 0.)#变异强度的变异, 必须大于0
        kv += ks * np.random.randn(*kv.shape)
        kv[:] = np.clip(kv, *DNA_BOUND) #不让DNA变异超出范围
    return kids


def kill_bad(pop, kids):
    for key in kids:#['DNA', 'mut_strength']
        pop[key] = np.vstack((pop[key], kids[key]))

    fitness = get_fitness(F(pop['DNA']))
    idx = np.arange(pop['DNA'].shape[0])
    good_idx = idx[fitness.argsort()][-POP_SIZE: ]
    for key in kids:
        pop[key] = pop[key][good_idx]
    return pop


pop = dict(DNA=5 * np.random.rand(POP_SIZE, DNA_SIZE), #range: [0,5]
      mut_strength=np.random.rand(POP_SIZE, DNA_SIZE)) #range: [0,1]


plt.ion()
x = np.linspace(*DNA_BOUND, 200)
plt.plot(x, F(x))


for _ in range(N_GENERATIONS):
    if 'sca' in globals():
        sca.remove()
    sca = plt.scatter(pop['DNA'], F(pop['DNA']), s=200, lw=0, c='red', alpha=0.5)
    plt.pause(0.05)

    kids = make_kid(pop, N_KID)
    pop = kill_bad(pop, kids)

plt.ioff()
plt.show()















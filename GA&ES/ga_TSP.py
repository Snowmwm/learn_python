#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#超参数
N_CITIES = 32 #DNA size
POP_SIZE = 20000 #种群大小
CROSS_RATE = 0.7 #交叉配对概率
MUTATION_RATE = 0.1 #变异概率
N_GENERATIONS = 1000 #繁衍的代数
KEEP_SIZE = 1 #精英保留


class GA(object):
    def __init__(self,DNA_size,cross_rate,mutation_rate,pop_size,keep_size):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.keep_size = keep_size

        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])  #随机产生初始种群

    def translateDNA(self, DNA, city_position): #翻译与表达DNA(生成路线图)
        lx = np.empty_like(DNA, dtype=np.float64)
        ly = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]
            lx[i, :] = city_coord[:, 0]
            ly[i, :] = city_coord[:, 1]

        lx = np.hstack((lx, lx[:,0][:,np.newaxis])) #回到起点
        ly = np.hstack((ly, ly[:,0][:,np.newaxis]))
        #lx = np.concatenate((lx,lx[:,0][:,np.newaxis]), axis=1)
        #ly = np.concatenate((ly,ly[:,0][:,np.newaxis]), axis=1)
        return lx, ly  #lx ly shape: (pop_size, DNA_size+1)

    def get_fitness(self, lx, ly): #在环境中的适应度(行走距离越短适应度越高)
        total_distance = np.empty((lx.shape[0],), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(lx, ly)):
            total_distance[i] = np.sum(
                np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys)))
                )  #第i个个体的总距离
        fitness = np.exp(self.DNA_size * 2 / total_distance)
        return fitness, total_distance  #shape:(pop_size, )

    def select(self, fitness): #自然选择
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size,
            replace=True, p=fitness/fitness.sum())
        return self.pop[idx]

    def keep(self, fitness): #精英保留
        keep_idx = np.argsort(-fitness)[:self.keep_size]
        return self.pop[keep_idx]

    def crossover(self, parent, pop): #基因交叉配对
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)
            cross_points = np.random.randint(0, 2, size=self.DNA_size).astype(np.bool)
            keep_city = parent[~cross_points] #参与交换的城市取反就是保留城市
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)] #找到与之前选择的城市不同的城市作为另一个parent的交换城市
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    def mutate(self, child): #变异
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapA, swapB
        return child

    def evolve(self, fitness): #进化
        keepn = self.keep(fitness)
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        pop[np.argsort(fitness)[:self.keep_size]] = keepn #用精英个体替换子代中适应度低的个体
        self.pop = pop


class TravelSalesPerson(object):
    def __init__(self, n_cities):
        self.city_position = np.random.rand(n_cities, 2)
        plt.ion()

    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T,
                s=30, c='b')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05,-0.05, 'Total distance=%.2f' %total_d,
                fontdict={'size':12, 'color':'red'})
        plt.xlim((-0.1,1.1))
        plt.ylim((-0.1,1.1))
        plt.pause(0.00001)


if __name__ == '__main__':
    ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, keep_size=KEEP_SIZE,
            mutation_rate=MUTATION_RATE, pop_size=POP_SIZE)

    env = TravelSalesPerson(N_CITIES)

    for generation in range(N_GENERATIONS):
        lx, ly = ga.translateDNA(ga.pop, env.city_position)
        fitness, total_distance = ga.get_fitness(lx, ly)

        ga.evolve(fitness)

        best_idx = np.argmax(fitness) #返回最佳个体索引
        print('Gen:', generation, '| best fit: %.2f' %fitness[best_idx],)

        env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])

    plt.ioff()
    plt.show()












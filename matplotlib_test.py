#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#可视化数据

from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

'''
myarray = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])#定义绘图的数据
plt.plot(myarray)#初始化绘图
plt.xlabel('x axis')#设定x,y轴
plt.ylabel('y axis')
plt.show()#绘图

myarray1 = np.array([1,2,3])
myarray2 = np.array([2,4,8])
plt.scatter(myarray1,myarray2)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()
'''
#导入CSV数据
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)
'''
#直方图（Histogtam）
data.hist()
plt.show()

#密度图
data.plot(kind = 'density', subplots = True, layout = (3, 3), sharex = False)
plt.show()

#箱线图
data.plot(kind = 'box', subplots = True, layout = (3, 3), sharex = False)
plt.show()
'''

#关系矩阵图：
correlations = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin = -1, vmax = 1)
fig.colorbar(cax)
ticks = np.arange(0, 9, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

#散点矩阵图：
scatter_matrix(data)
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

from random_walk import RandomWalk

#折线图
'''
myarray_x = [1, 2, 3]
myarray_y = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]]) #定义绘图的数据
plt.plot(myarray_x, myarray_y, linewidth = 5) #初始化绘图
plt.fill_between(myarray_x, myarray_y[:, 1], myarray_y[:, 2], facecolor='blue', alpha=0.2) #给图标区域上色
#添加标题，并给坐标轴加上标签
plt.title('Test', fontsize = 32)
plt.xlabel('x axis', fontsize = 16)
plt.ylabel('y axis', fontsize = 16)
plt.show() #打开matplotlib查看器，并显示绘制的图形
'''

#散点图
'''
myarray1 = list(range(1,1001))
myarray2 = [x**2 for x in myarray1]
plt.scatter(myarray1, myarray2, s=30, edgecolor='none', c=(0.5, 0.2 ,0)) 
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.axis([0, 1100, 0, 1100000]) #设置每个坐标的取值范围
#plt.tick_params(axis = 'both', which = 'major', labelsize = 14) #设置刻度标记大小
plt.savefig('matplotlib_test.png', bbox_inches='tight')
plt.show()
'''

#创建一个RandomWalk实例，并将其包含的点都绘制出来
'''
rw = RandomWalk(5000)
rw.fill_walk()
plt.figure(figsize=(12,8))
point_numbers = list(range(rw.num_points))
plt.scatter(rw.x_values, rw.y_values, s=1, c=point_numbers, \
cmap=plt.cm.Blues, edgecolor='none')
#突出起点和终点
plt.scatter(0, 0, c='green', edgecolor='none', s=40)
plt.scatter(rw.x_values[-1], rw.y_values[-1], c='red', edgecolor='none', s=40)
#隐藏坐标轴
plt.axes().get_xaxis().set_visible(False)
plt.axes().get_yaxis().set_visible(False)
plt.show()
'''

#导入CSV数据

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
'''
#可视化数据
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

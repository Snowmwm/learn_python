#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
'''
#定义绘图的数据
myarray = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
#初始化绘图
plt.plot(myarray)
#设定x,y轴
plt.xlabel('x axis')
plt.ylabel('y axis')
#绘图
plt.show()
'''

myarray1 = np.array([1,2,3])
myarray2 = np.array([2,4,8])

plt.scatter(myarray1,myarray2)

plt.xlabel('x axis')
plt.ylabel('y axis')

plt.show()
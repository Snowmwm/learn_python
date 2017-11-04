#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pandas import read_csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer

#导入CSV数据
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)

#将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

print('数据尺度调整:')
trans = MinMaxScaler(feature_range = (0, 1))
#数据转换
newX1 = trans.fit_transform(X)
#设定数据的精度
np.set_printoptions(precision = 3)
print(newX1)


print('\n正态化数据:')
trans = StandardScaler()
newX2 = trans.fit_transform(X)
print(newX2)

print('\n标准化数据:')
trans = Normalizer()
newX3 = trans.fit_transform(X)
print(newX3)

print('\n二值数据:')
trans = Binarizer(threshold = 0.5)
newX4 = trans.fit_transform(X)
print(newX4)
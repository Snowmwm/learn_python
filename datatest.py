#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pandas import read_csv, set_option
#使用pandas导入CSV数据
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)

print('数据维度:')
print(data.shape)

print('\n数据属性:')
print(data.dtypes)

print('\n查看前10行：')
peek = data.head(10)
print(peek)

print('\n统计性描述：')
set_option('display.width', 100)
set_option('precision',2)
print(data.describe())

print('\n数据分布统计：')
print(data.groupby('class').size())

print('\n数据的相关性：')
set_option('display.width', 100)
set_option('precision',2)
print(data.corr(method = 'pearson'))

print('\n数据的高斯偏离:')
print(data.skew())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pandas import read_csv
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression

#导入数据
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)

#将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

#重复随机分离
n_splits = 10
test_size = 0.33
seed = 7
kfold = ShuffleSplit(n_splits = n_splits, test_size = test_size, random_state = seed)

#分类正确率
model = LogisticRegression()
result = cross_val_score(model, X, Y, cv = kfold)
print('算法评估结果: %.3f%% (%.3f%%)' % (result.mean() * 100, result.std() * 100))
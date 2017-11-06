#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

#导入数据
filename = 'housing.csv'
data = read_csv(filename)

#将数据分为输入数据和输出结果
array = data.values
X = array[:, 8:]
Y = array[:, 6]

#K折交叉验证分离
num_folds = 10
seed = 7
kfold = KFold(n_splits = num_folds, random_state = seed)

#线性回归算法
model = LinearRegression()

#平均绝对误差
scoring1 = 'neg_mean_absolute_error'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring1)
print('MAE: %.3f (%.3f)' % (result.mean(), result.std()))

#均方误差
scoring2 = 'neg_mean_squared_error'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring2)
print('MSE: %.3f (%.3f)' % (result.mean(), result.std()))

#决定系数（R^2）
scoring3 = 'r2'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring3)
print('R2: %.3f (%.3f)' % (result.mean(), result.std()))
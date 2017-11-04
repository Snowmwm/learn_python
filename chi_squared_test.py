#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#卡方检验

from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest, chi2

#导入数据
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)

#将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

#特征选择
test = SelectKBest(score_func = chi2, k = 4)
fit = test.fit(X, Y)
set_printoptions(precision = 3)
print(fit.scores_)
features = fit.transform(X)
print(features)
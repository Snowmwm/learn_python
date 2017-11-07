#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import AdaBoostClassifier

#导入pima数据
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)

#将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

#K折交叉验证分离
num_folds = 10
seed = 7
kfold = KFold(n_splits = num_folds, random_state = seed)

#AdaBoost分类器组合
num_tree = 30
model = AdaBoostClassifier(n_estimators = num_tree)

#算法评估
result = cross_val_score(model, X, Y, cv = kfold)
print(result.mean())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

#导入数据
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

#逻辑回归
model = LogisticRegression()

#分类正确率
result = cross_val_score(model, X, Y, cv = kfold)
print('算法评估结果: %.3f%% (%.3f%%)' % (result.mean() * 100, result.std() * 100))

#对数损失函数(logloss)
scoring1 = 'neg_log_loss'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring1)
print('Logloss %.3f (%.3f)' % (result.mean(), result.std()))

#AUC
scoring2 = 'roc_auc'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring2)
print('AUC %.3f (%.3f)' % (result.mean(), result.std()))
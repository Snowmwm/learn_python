#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pandas import read_csv
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV

#导入pima数据
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)

#将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

#逻辑回归算法实例化
model = LogisticRegression()

#设置要遍历的参数
param_grid = {'C': [5, 2, 1, 0.1], 'max_iter': [50, 80, 90, 100, 110, 120]}

#通过网络搜索查询最优参数
grid = GridSearchCV(estimator = model, param_grid = param_grid)
grid.fit(X, Y)

#搜索结果
print('最高得分：%.3f' % grid.best_score_)
print('最优参数：%s, %s' % (grid.best_estimator_.C, grid.best_estimator_.max_iter))
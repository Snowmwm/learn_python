#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#特征重要性

from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier

#导入数据
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)

#将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

#特征选择
model = ExtraTreesClassifier()
fit = model.fit(X, Y)
print(fit.feature_importances_)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

#导入数据
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)

#将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

#分离训练集和测试集
test_size = 0.33
seed = 4
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = seed)

#逻辑回归
model = LogisticRegression()

#分类正确率
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print('算法评估结果: %.3f%%' % (result * 100))

predicted = model.predict(X_test)
#混淆矩阵
matrix = confusion_matrix(Y_test,predicted)
classes = ['0', '1']
dataframe = pd.DataFrame(data = matrix, index = classes, columns = classes)
print('混淆矩阵:')
print(dataframe)

#分类报告
report = classification_report(Y_test,predicted)
print('分类报告:')
print(report)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
from sklearn.neighbors import KNeighborsClassifier #K近邻
from sklearn.tree import DecisionTreeClassifier #分类和回归树
from sklearn.svm import SVC #支持向量机
from sklearn.naive_bayes import GaussianNB #朴素贝叶斯

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

#缩写字典
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['SVM'] = SVC()
models['NB'] = GaussianNB()

#算法评估
results = []
for name in models:
    result = cross_val_score(models[name], X, Y, cv = kfold)
    results.append(result)
    print('%s: %.3f (%.3f)' % (name, result.mean(), result.std()))

#图表显示
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
from sklearn.neighbors import KNeighborsClassifier #K近邻
from sklearn.tree import DecisionTreeClassifier #分类和回归树
from sklearn.svm import SVC #支持向量机

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

#转换成list
models1 = list(zip(models.keys(), models.values()))

#投票算法
ensemble_model = VotingClassifier(estimators = models1)

#算法评估
result = cross_val_score(ensemble_model, X, Y, cv = kfold)
print(result.mean())
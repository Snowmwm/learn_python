#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
#from sklearn.preprocessing import StandardScaler #正态化数据
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

#流程：
#1、数据预处理（调整数据尺度/正态化数据/标准化数据/二值数据）
#2、数据特征选择（降维）（利用统计方法比如卡方检验来选择特征/递归特征消除/主要成分分析/用决策树计算特征重要性）
#3、特征集合
#4、训练一个模型（分类模型：逻辑回归/LDA/贝叶斯分类器/KNN/CART/SVM; 回归模型：线性回归/岭回归/套索回归/弹性网络回归/KNN/CART/SVM）

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

#生成 feature union
features = []
features.append(('pca', PCA()))
features.append(('select_best', SelectKBest(k = 6)))

#生成Pipeline
steps = []
steps.append(('feature_union', FeatureUnion(features)))
steps.append(('logistic', LogisticRegression()))
#steps.append(('lda', LinearDiscriminantAnalysis()))

#将步骤封装
model = Pipeline(steps)

#评估
result = cross_val_score(model, X, Y, cv = kfold)
print(result.mean())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 1. 准备
# a) 导入类库
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler #正态化数据
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression #LR
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.tree import DecisionTreeClassifier #CART
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
from sklearn.naive_bayes import GaussianNB #朴素贝叶斯
from sklearn.svm import SVC #支持向量机
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
AdaBoostClassifier, GradientBoostingClassifier #集成

# b) 导入数据集
filename = 'sonar.all_data.csv'
data = read_csv(filename, header = None)

# 2. 概述数据
'''
# a) 描述性统计
print('数据维度:')
print(data.shape)
print('特征属性的类型:')
set_option('display.max_rows', 500)
print(data.dtypes)
print('\n查看数据前十行:')
set_option('display.width', 80)
print(data.head(10))
print('\n统计性描述:')
set_option('precision',2)
print(data.describe())
print('\n数据分布统计：')
print(data.groupby(60).size())

# b) 数据可视化
#直方图
data.hist(sharex = False, sharey = False, xlabelsize=1, ylabelsize=1)
plt.show()
#密度图
data.plot(kind = 'density', subplots = True, layout = (8, 8), \
sharex = False, legend = False, fontsize = 1)
plt.show()
#关系矩阵图
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data.corr(), vmin = -1, vmax = 1, interpolation = 'none')
fig.colorbar(cax)
plt.show()
'''

#将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:60].astype(float)
Y = array[:, 60]

#分离训练集和验证集
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, \
test_size = validation_size, random_state = seed)

#评估标准
num_folds = 10
seed = 7
kfold = KFold(n_splits = num_folds, random_state = seed)
scoring = 'accuracy'

#画图准备
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)

#正态化数据
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

#评价算法(原始数据)
'''
#形成评估基准(用于比较后续算法的改善)
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()

results = []
for name in models:
    result = cross_val_score(models[name], X_train, Y_train, \
    cv = kfold, scoring=scoring)
    results.append(result)
    print('%s: %.3f (%.3f)' % (name, result.mean(), result.std()))

plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()
'''

#评价算法(正态化数据)
'''
pipelines = {}
for key in models:
    pipelines['Scaler'+key] = Pipeline([('Scaler', StandardScaler()), \
    (key, models[key])])

results = []
for key in pipelines:
    result = cross_val_score(pipelines[key], X_train, Y_train, \
    cv = kfold, scoring = scoring)
    results.append(result)
    print('%s: %.3f (%.3f)' % (key, result.mean(), result.std()))
    
plt.boxplot(results)
ax.set_xticklabels(pipeline.keys())
plt.show()
'''

#超参数优化(KNN)
'''
param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]}
model = KNeighborsClassifier()
grid = GridSearchCV(estimator = model, param_grid = param_grid, \
scoring = scoring, cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)

print('最优: %s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))
result = zip(grid_result.cv_results_['mean_test_score'], \
grid_result.cv_results_['std_test_score'], grid_result.cv_results_['params'])
for mean, std, param in result:
    print('%.3f (%.3f) with %r' % (mean, std, param))
'''

#超参数优化(SVM)
'''
param_grid = {}
param_grid['C'] = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2] #惩罚系数
param_grid['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid'] #核函数
model = SVC()
grid = GridSearchCV(estimator = model, param_grid = param_grid, \
scoring = scoring, cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)

print('最优: %s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))
result = zip(grid_result.cv_results_['mean_test_score'], \
grid_result.cv_results_['std_test_score'], grid_result.cv_results_['params'])
for mean, std, param in result:
    print('%.3f (%.3f) with %r' % (mean, std, param))
'''

#集成算法
'''
ensemble = {}
ensemble['ABC'] = AdaBoostClassifier()
ensemble['GBC'] = GradientBoostingClassifier()
ensemble['RFC'] = RandomForestClassifier()
ensemble['ETC'] = ExtraTreesClassifier()

ensembles = {}
for key in ensemble:
    ensembles['Scaled-'+key] = Pipeline([('Scaler', StandardScaler()), \
    (key, ensemble[key])])
    
results = []
for key in ensembles:
    result = cross_val_score(ensembles[key], X_train, Y_train, \
    cv = kfold, scoring = scoring)
    results.append(result)
    print('%s: %.3f (%.3f)' % (key, result.mean(), result.std()))

plt.boxplot(results)
ax.set_xticklabels(ensembles.keys())
plt.show()
'''

#集成算法调参
'''
#GBC
model = GradientBoostingClassifier()
param_grid = {'n_estimators': [10, 30, 50, 80, 100, 200, 300, 400, 500, 600, 700, 800]}
grid = GridSearchCV(estimator = model, param_grid = param_grid, \
scoring = scoring, cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)
print('最优: %s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))
'''

#训练最终模型
model_svc = SVC(C = 1.5, kernel = 'rbf')
model_svc.fit(rescaledX, Y_train)

#评估算法模型
rescaled_validationX = scaler.transform(X_validation)
predictions = model_svc.predict(rescaled_validationX)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
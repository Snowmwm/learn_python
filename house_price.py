#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python机器学习项目模版

# 1. 准备
# a) 导入类库
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler #正态化数据
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet #线性回归/套索/弹性网络
from sklearn.tree import DecisionTreeRegressor #CART
from sklearn.neighbors import KNeighborsRegressor #KNN
from sklearn.svm import SVR #SVM
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, \
AdaBoostRegressor, GradientBoostingRegressor #集成
from sklearn.metrics import mean_squared_error #均方误差

# b) 导入数据集
filename = 'housing.csv'
data = read_csv(filename)

# 2. 概述数据
'''
# a) 描述性统计
print('数据维度:')
print(data.shape)
print('特征属性的字段类型:')
print(data.dtypes)
print('\n查看最开始的20条记录:')
set_option('display.width', 180)
print(data.head(20))

# b) 数据可视化
#直方图
data.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
plt.show()
#密度图
data.plot(kind = 'density', subplots = True, layout = (4, 5), \
sharex = False, fontsize = 1)
plt.show()
#箱线图
data.plot(kind = 'box', subplots = True, layout = (4, 5), \
sharex = False, sharey = False, fontsize = 8)
plt.show()
#散点矩阵图
scatter_matrix(data)
plt.show() 
#关系矩阵图：
correlations = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin = -1, vmax = 1, interpolation = 'none')
fig.colorbar(cax)
ticks = np.arange(0, 20, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
plt.show()
'''

#将数据分为输入数据和输出结果
array = data.values
X = array[:, 8:]
Y = array[:, 6]

#分离训练集和验证集
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, \
test_size = validation_size, random_state = seed)

#评估标准
num_folds = 10
seed = 7
kfold = KFold(n_splits = num_folds, random_state = seed)
scoring = 'neg_mean_squared_error'

#评价算法(原始数据)
'''
#形成评估基准(用于比较后续算法的改善)
models = {}
models['LR'] = LinearRegression()
models['LASSO'] = Lasso()
models['EN'] = ElasticNet()
models['KNN'] = KNeighborsRegressor()
models['CART'] = DecisionTreeRegressor()
models['SVM'] = SVR()

results = []
for name in models:
    result = cross_val_score(models[name], X_train, Y_train, \
    cv = kfold, scoring=scoring)
    results.append(result)
    print('%s: %.3f (%.3f)' % (name, result.mean(), result.std()))
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
'''

#超参数优化(KNN)
'''
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]}
model = KNeighborsRegressor()
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
ensemble['ABR'] = AdaBoostRegressor()
ensemble['ABK'] = AdaBoostRegressor(base_estimator = KNeighborsRegressor(n_neighbors = 3))
ensemble['ABL'] = AdaBoostRegressor(LinearRegression())
ensemble['ABC'] = AdaBoostRegressor(DecisionTreeRegressor())
ensemble['RFR'] = RandomForestRegressor()
ensemble['ETR'] = ExtraTreesRegressor()
ensemble['GBR'] = GradientBoostingRegressor()

ensembles = {}
for key in ensemble:
    ensembles['Scaled-'+key] = Pipeline([('Scaler', StandardScaler()), \
    (key, ensemble[key])])
    
results = []
for key in ensembles:
    result = cross_val_score(ensembles[key], X_train, Y_train, \
    cv = kfold, scoring = scoring)
    results.append(result)
    #print('%s: %.3f (%.3f)' % (key, result.mean(), result.std()))
    
#箱线图显示
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(ensembles.keys())
plt.show()
'''

#集成算法调参
'''
#正态化数据
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
#ABC
model = AdaBoostRegressor(DecisionTreeRegressor())
param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200,\
300, 400, 500, 600, 700, 800]}
grid = GridSearchCV(estimator = model, param_grid = param_grid, \
scoring = scoring, cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)
print('ABC:\n最优: %s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))

#ETR
model = ExtraTreesRegressor()
param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]}
grid = GridSearchCV(estimator = model, param_grid = param_grid, \
scoring = scoring, cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)
print('ETR:\n最优: %s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))

#GBR
model = GradientBoostingRegressor()
param_grid = {'n_estimators': [10, 30, 50, 80, 100, 200, 300, 400, 500, 600, 700, 800]}
grid = GridSearchCV(estimator = model, param_grid = param_grid, \
scoring = scoring, cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)
print('GBR:\n最优: %s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))
'''

#训练最终模型
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model_etr = ExtraTreesRegressor(n_estimators = 120)
model_etr.fit(rescaledX, Y_train)

#评估算法模型
rescaledX_validation = scaler.transform(X_validation)
predictions = model_etr.predict(rescaledX_validation)
print(mean_squared_error(Y_validation, predictions))
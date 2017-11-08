#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 1. 准备
# a) 导入类库
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix #散点矩阵图
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score 
from sklearn.linear_model import LogisticRegression #LR
from sklearn.tree import DecisionTreeClassifier #CART
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #朴素贝叶斯
from sklearn.svm import SVC #支持向量机

# b) 导入数据集
filename = 'iris.data.csv'
names = ['separ_length', 'separ_width', 'petal_length', 'petal_width', 'class']
data = read_csv(filename, names = names)
'''
# 2. 概述数据
# a) 描述性统计
print('数据维度:')
print('行:%s, 列:%s' % data.shape)
print('\n数据维度:')
print(data.head(10))
print('\n统计性描述:')
set_option('precision',2)
print(data.describe())
print('\n数据分布统计：')
print(data.groupby('class').size())

# b) 数据可视化
#箱线图
data.plot(kind = 'box', subplots = True, layout = (2, 2), sharex = False, sharey = False)
plt.show()
#直方图
data.hist()
plt.show()
#散点矩阵图
scatter_matrix(data)
plt.show()
'''
# 3. 预处理数据
# a) 数据清洗
# b) 特征选择
# c) 数据转换

# 4. 评估算法
# a) 分离数据集
#将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:4]
Y = array[:, 4]

#分离训练集和验证集
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = validation_size, random_state = seed)
# b) 评估选项和评估矩阵
num_folds = 10
kfold = KFold(n_splits = num_folds, random_state = seed) #K折交叉验证分离
# c) 算法审查
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()
# d) 算法比较
results = []
for name in models:
    result = cross_val_score(models[name], X_train, Y_train, cv = kfold)
    results.append(result)
    print('%s: %.3f (%.3f)' % (name, result.mean(), result.std()))
'''
#箱线图显示
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()
'''
# 5. 提高模型准确度
# a) 算法调参
# b) 集成算法

# 6. 序列化模型
# a) 预测评估数据集
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# b) 利用整个数据集生产模型
# c) 序列化模型 
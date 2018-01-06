#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#创建array
'''
a = np.array([[1,2,3],
            [2,3,4]],dtype=np.float)
b = np.zeros((5,6))
c = np.ones((3,4))
d = np.empty((4,5))
e = np.arange(10,30,2).reshape((5,2))
f = np.linspace(1,10,21).reshape((3,7))
print(f)
'''

#array属性

array = np.array([[1,2,3],
                [2,3,4]])
print(array)
print('number of dim:', array.ndim)
print('shape:', array.shape)
print('size:', array.size)

a = np.array([0,1,2,3])
print(a[:1])

#numpy基础运算
'''
a = np.array([[10,20],[30,40]])
b = np.arange(4).reshape((2,2))

print(a<30)
c = a - b
d = a + b
e = a * b
f = a**2
g = 10*np.sin(a)
print(c,d,e,f,g)
h = np.dot(c,d)
print(h)
i = a.dot(b)
print(i)

j = np.random.random((2,2))
print(j, np.sum(j), np.min(j), np.max(j))
print(np.sum(j,axis=1))
'''

'''
A = np.arange(2,14).reshape((3,4))
B = np.arange(14,2,-1).reshape((6,2))

print(np.argmin(A)) #最大值最小值索引
print(np.argmax(A))
print(np.mean(A))
print(np.mean(A,axis=0)) #指定计算平均数的维度
print(np.average(A))
print(np.median(A)) #中位数
print(A,'\n',np.cumsum(A).reshape(A.shape)) #累加数列
print(np.diff(np.cumsum(A).reshape(A.shape))) #差分矩阵
print(np.nonzero(A-7)) #输出非0数的位置
print(np.sort(B)) #逐行排序
print(np.transpose(A)) #矩阵转置
print(B.T)
print((B.T).dot(B))
print(np.clip(A,4,10)) #限制矩阵的取值范围
'''


#numpy的索引
'''
A = np.arange(3,15)
print(A, A[3])
A = A.reshape((3,4))
print(A)
print(A[2],A[:,0:2])
print(A[2][2], A[2,1])

for row in A: #row:行
    print(row)

for column in A.T: #column:列
    print(column)

print(A.flatten()) #展平  
for item in A.flat: #A.flat：迭代器
    print(item)
'''

#array的合并
'''
A = np.ones((2,3),dtype=np.int64)
B = np.arange(6).reshape(2,3)
print(A,B)
print(np.vstack((A,B))) #按行合并
print(np.hstack((A,B))) #按列合并
C = np.array([1,3,5])
print(C.T)
print(C[np.newaxis,:].T) #插入一个新的维度
print(C[:,np.newaxis])

print(np.concatenate((A,B,C[np.newaxis,:]), axis=0)) #在指定维度合并
print(np.vstack((A,B,C[np.newaxis,:])))
print(np.concatenate((A,A,B,B), axis=1))
print(np.hstack((A,A,B,B)))
'''

#array的分割
'''
A = np.arange(12).reshape((3,4))

print(A)
print(np.split(A,2,axis=1)) #纵向分割
B,C = np.split(A,2,axis=1)
print(B,C)
print(np.split(A,3,axis=0)) #横向分割

print(np.array_split(A,3,axis=1)) #不均等分割

print(np.vsplit(A,3)) #按行或列分割
print(np.hsplit(A,2))
'''

#copy 与 deepcopy
'''
a = np.arange(4)
print(a)
b = a.copy() #deepcopy
c = a
d = b
a[0] = 5
b[0] = 6
print(a,b,c,d)
'''

#numpy加速
'''
#能用view尽量不用copy，能用ravel尽量不用flatten
a_view1 = a[1:2, 3:6]    # 切片 slice
a_view2 = a[:100]        # 同上
a_view3 = a[::2]         # 跳步
a_view4 = a.ravel()      # 上面提到了

a_copy1 = a[[1,4,6], [2,4,6]]   # 用 index 选
a_copy2 = a[[True, True], [False, True]]  # 用 mask
a_copy3 = a[[1,2], :]        # 虽然 1,2 的确连在一起了, 但是他们确实是 copy
a_copy4 = a[a[1,:] != 0, :]  # fancy indexing
a_copy5 = a[np.isnan(a), :]  # fancy indexing


#使用 np.take(), 替代用 index 选数据的方法.
a = np.random.rand(1000000, 10)
N = 99
indices = np.random.randint(0, 1000000, size=10000)

def f1(a):
    for _ in range(N):
        _ = np.take(a, indices, axis=0)

def f2(b):
    for _ in range(N):
        _ = b[indices]

print('%f' % ((t1-t0)/N))    # 0.000393
print('%f' % ((t2-t1)/N))    # 0.000569


#使用 np.compress(), 替代用 mask 选数据的方法.
mask = a[:, 0] < 0.5
def f1(a):
    for _ in range(N):
        _ = np.compress(mask, a, axis=0)

def f2(b):
    for _ in range(N):
        _ = b[mask]

print('%f' % ((t1-t0)/N))    # 0.028109
print('%f' % ((t2-t1)/N))    # 0.031013

#使用out参数
a = a + 1         # 0.035230
a = np.add(a, 1)  # 0.032738
#前两个运算会copy
a += 1                 # 0.011219
np.add(a, 1, out=a)    # 0.008843
'''
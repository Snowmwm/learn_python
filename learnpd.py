#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
'''
s = pd.Series([1,2,3,np.nan,5,6])
#print(s)

dates = pd.date_range('20160101' ,periods=6)
#print(dates)

df = pd.DataFrame(np.random.randn(6,4),index=dates,
                    columns=['a','b','c','d'])            
#print(df)

df1 = pd.DataFrame(np.arange(12).reshape((3,4)))
#print(df1)

df2 = pd.DataFrame({'A':1.,
            'B':pd.Timestamp('20130101'),
            'C':pd.Series(1,index=list(range(4)),dtype='float32'),
            'D':np.array([3]*4,dtype='int32'),
            'E':pd.Categorical(['test','train','test','train']),
            'F':'foo'})
print(df2)
print(df2.dtypes)
print(df2.index) #打印列的name
print(df2.columns)#打印行的name
print(df2.values)
print(df2.describe())
print(df2.T)
print(df2.sort_index(axis=1, ascending=False)) #行的name倒序排序
print(df.sort_values(by='c'))  #对c列的值排序
#print(df.sort_values(by='20160103'))
'''



#pandas选择数据
'''
dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,
                    columns=['A','B','C','D'])  

                    
print(df['A'],'\n',df.A)
print(df[0:3],'\n',df['20170102':'20170105'])

print(df.loc['20170102']) #根据标签(loc)选择
print(df.loc[:,['A','C']])

print(df.iloc[[1,3,5],1:3])#根据位置(iloc)选择

print(df.ix[:3,['A','D']])#混合(ix)选择

print(df[df.A>8])#条件判断选择
'''

#pandas设置值
'''
dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,
                    columns=['A','B','C','D'])    

df.iloc[2,2] = 'c'
df.loc['20170103','B'] = 'b'
df.A[df.A>4] = 0
df['E'] = np.nan
df['F'] = pd.Series([6,5,4,3,2,1],
            index=pd.date_range('20170101',periods=6))
print(df)
'''

#处理丢失数据
'''
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan

print(df.dropna(axis=0,how='any')) #按行丢弃数据 how={'any','all'}
print(df.dropna(axis=1,how='any')) #按列丢弃数据

print(df.fillna(value=0)) #在缺失数据的位置填上0

print(df.isnull()) #判断是否缺失数据
print(np.any(df.isnull()) == True)
'''

#pandas导入导出数据
"""
Format	Data Description	    Reader	            Writer
Type

text	CSV	                    read_csv	        to_csv
text	JSON	                read_json	        to_json
text	HTML	                read_html	        to_html
text	Local clipboard	        read_clipboard	    to_clipboard
binary	MS Excel	            read_excel      	to_excel
binary	HDF5 Format	            read_hdf	        to_hdf
binary	Feather Format	        read_feather	    to_feather
binary	Parquet Format	        read_parquet	    to_parquet
binary	Msgpack	                read_msgpack	    to_msgpack
binary	Stata	                read_stata	        to_stata
binary	SAS	                    read_sas	 
binary	Python Pickle Format	read_pickle	        to_pickle
SQL  	SQL	                    read_sql	        to_sql
SQL	    Google Big Query	    read_gbq	        to_gbq
"""

'''
D = ['CSV','JSON','HTML','Local clipboard','MS Excel',
    'HDF5 Format','Feather Format','parquet Feather',
    'Msgpack','Stata','SAS','Python Pickle Format',
    'SQL','Google Big Query']

L = ['csv','json','html','clipboard','excel','hdf','feather',
    'parquet','msgpack','stata','sas','pickle','sql','gbq']

Reader = []
Writer = []
for format in L:
    Reader.append('read_' + format)
    Writer.append('to_' + format)
    
df = pd.DataFrame({
    'Format Type':np.nan,
    'Data Description':D,
    'Reader':Reader,
    'Writer':Writer,
})

df.ix[:3, 'Format Type'] = 'text'
df.ix[4:11, 'Format Type'] = 'binary'
df.ix[12:, 'Format Type'] = 'SQL'

df.iloc[10,3] = np.nan

pop = df.pop('Format Type') #交换两列位置
df.insert(0,'Format Type', pop)

df.to_csv('pandas_rl.csv')
'''

#pandas合并 connat 
'''
#concat(concatenating)
df1 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*3, columns=['a','b','c','d'])

res = pd.concat([df1,df2,df3],axis=0,ignore_index=True) #竖向合并
#print(res)

#join,['inner','outer']
df1 = pd.DataFrame(np.ones((3,4))*1, 
    columns=['a','b','c','d'],
    index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*2, 
    columns=['b','c','d','e'],
    index=[2,3,4])

print(pd.concat([df1,df2],join='outer',ignore_index=True))
print(pd.concat([df1,df2],join='inner')) #inner只合并相同部分

#join_axes
print(pd.concat([df1,df2],axis=1))
print(pd.concat([df1,df2],axis=1,join_axes=[df1.index]))

#append
res = df1.append([df2,df1],ignore_index=True)
print(res)
s1 = pd.Series([1,2,4,5],index=['a','b','d','e'])
res = res.append(s1,ignore_index=True)
print(res)
'''

#pandas合并 merge
'''
left = pd.DataFrame({'key1':['K0','K1','K2','K3'],
                     'key2':['K0','K1','K0','K1'],
                     'A':['A0','A1','A2','A3'],
                     'B':['B0','B1','B2','AB']})
right = pd.DataFrame({'key1':['K1','K2','K3','K0'],
                      'key2':['K0','K1','K1','K0'],
                      'C':['C0','C1','C2','C3'],
                      'D':['D0','D1','D2','D3']})

print(pd.merge(left, right, on='key1'))
print(pd.merge(left, right, on=['key1','key2'])) #默认how='inner'
print(pd.merge(left, right, on=['key1','key2'],how='outer'))
print(pd.merge(left, right, on=['key1','key2'],how='left'))
print(pd.merge(left, right, on=['key1','key2'],how='right'))

df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})

#indicator默认False,可改名
res = pd.merge(df1,df2,on='col1',indicator=True) 
res = pd.merge(df1,df2,on='col1', how='outer',indicator=True) 
print(res)
#index
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                     index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])

print(left.merge(right,
    left_index=True,right_index=True,how='outer'))

#解决overlapping问题
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})

print(pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='outer'))
'''


#pandas plot画图
'''
import matplotlib.pyplot as plt

#Series
data = pd.Series(np.random.randn(1000),index=np.arange(1000)+1)

#DataFrame
data = pd.DataFrame(np.random.randn(1000,4),
        index=np.arange(1000)+1,
        columns=list('ABCD'))

data = data.cumsum()#累加

#画图
#data.plot()
#plt.show()

#plot methods:
#'bar','hist','box','kde','ares','scatter','hexbin','pie'

ax = data.plot.scatter(x='A',y='B',color='Darkblue',label='Class 1')
data.plot.scatter(x='A',y='C',color='DarkGreen',label='Class 2',
        ax=ax)

plt.show()
'''


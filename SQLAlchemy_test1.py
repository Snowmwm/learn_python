#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import pymysql

#建立连接
connect = pymysql.Connect(
    user='root',
    passwd='SnowWM179328654',
    db='test',
    charset='utf8'
)
cursor = connect.cursor()

#创建表
cursor.execute('create table user(id varchar(20) primary key, name varchar(20))')
# 插入数据，注意MySQL的占位符是%s:
cursor.execute('insert into user (id, name) values (%s, %s)', ['1', 'Michael'])
#print(cursor.rowcount)
# 提交事务:
connect.commit()    
cursor.close()

cursor = connect.cursor()
#查询数据
cursor.execute('select * from user where id = %s', ('1',))
values = cursor.fetchall()
print(values)

#查询Mysql版本
cursor.execute('select version()')
v = cursor.fetchall()
print ("Database version : %s " % v)

cursor.close()
connect.close()

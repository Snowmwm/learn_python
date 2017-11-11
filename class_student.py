#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#创建类
class Student(object):
    '''存储学生的信息。并根据分数评出等级'''

    def __init__(self, name, score, sex): #通过“__init__”方法绑定属性
        #把存储在形参中的值传给私有变量
        self.__name = name
        self.__score = score
        self.__sex = sex

    def print_score(self):
        print('%s:%s'%(self.__name,self.__score))

    def get_name(self):
        return self.__name

    def get_score(self):
        return self.__score
        
    def get_sex(self):
        if self.__sex == 'M':
            print('%s is a boy' % self.__name)
        else:
            print('%s is a girl' % self.__name)

    def set_score(self,score):
        if 0 <= score <= 100:
            self.__score=score
        else:
            raise ValueError('bad score')

    def get_grade(self,la,lb):
        if self.__score >= la:
            return'A'
        elif self.__score >= lb:
            return'B'
        else:
            return'C'
            
#创建子类            
class Boy(Student):
    def __init__(self, name, score, sex='M'):
        super().__init__(name, score, sex)

class Girl(Student):
    def __init__(self, name, score, sex='F'):
        super().__init__(name, score, sex)
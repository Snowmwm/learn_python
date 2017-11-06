#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#创建类
class student(object):
#通过“__init__”方法绑定属性
    def __init__(self,name,score):
        self.__name = name
        self.__score = score

    def print_score(self):
        print('%s:%s'%(self.__name,self.__score))

    def get_name(self):
        return self.__name

    def get_score(self):
        return self.__score

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

#创建实例
bart = student('Bart Simpson',59)
lisa = student('Lisa Simpson',87)

#调用实例方法
bart.print_score()
print(bart.get_grade(90,60))
bart.set_score(92)
bart.print_score()
print(bart.get_grade(90,60))
lisa.print_score()
print(lisa.get_grade(90,60))

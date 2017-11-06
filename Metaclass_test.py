#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#创建元类
class SayMetaClass(type):
    def __new__(cls, name, bases, attrs):
        attrs['say_'+name] = lambda self, value, saying = name:print(saying+','+value+'!')
        return type.__new__(cls, name, bases, attrs)
        
#创建类
class Hello(object, metaclass = SayMetaClass):
    pass
    
#创建实例
hello = Hello()

#调用实例方法
hello.say_Hello('world!')
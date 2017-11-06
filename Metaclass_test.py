#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#创建Say元类
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


#创建List元类
class ListMetaClass(type):
    def __new__(cls, name, bases, attrs):
        attrs['add'] = lambda self, value:self.append(value)
        return type.__new__(cls, name, bases, attrs)
        
#创建类
class MyList(list, metaclass = ListMetaClass):
    pass
    
#创建实例
L = MyList()

#调用实例方法
L.add(1)

print(L)
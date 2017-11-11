#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from class_student import Student, Girl

class TestStudent(unittest.TestCase):
    '''针对Student类及其子类的单元测试'''
    def setUp(self):
        '''创建三个对象供测试使用'''
        self.bart = Student('Bart Simpson', 59, 'M')
        self.lisa = Student('Lisa Simpson', 87, 'F')
        self.ana = Girl('Ana Conda', 100)
    
    def test_get_grade(self):
        '''测试评级方法'''
        self.assertEqual('C', self.bart.get_grade(90, 60))
        self.assertEqual('B', self.lisa.get_grade(90, 60))
        self.assertEqual('A', self.ana.get_grade(90, 60))
        
    def test_set_score(self):
        '''测试分数修改方法'''
        self.bart.set_score(99)
        self.assertEqual('A', self.bart.get_grade(90, 60))
        
unittest.main()
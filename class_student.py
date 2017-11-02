#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class student(object):
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

bart = student('Bart Simpson',59)
lisa = student('Lisa Simpson',87)

bart.print_score()
print(lisa.get_grade(90,60))
lisa.set_score(92)
print(lisa.get_grade(90,60))
bart.set_score(999)
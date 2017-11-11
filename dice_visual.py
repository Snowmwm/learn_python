#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from die import Die
import pygal

#创建两个D6
die_1 = Die(20)
die_2 = Die(20)

#掷D6，并将结果存储在一个列表中
results = []
for roll_num in range(1000):
    result = die_1.roll()+die_2.roll()
    results.append(result)
    
#分析结果
frequencies = []
max_result = die_1.num_sides + die_2.num_sides
for value in range(2, max_result+1):
    frequency = results.count(value)
    frequencies.append(frequency)
    
#可视化结果
hist = pygal.Bar()
hist.title = 'Results of rolling one D6 1000 times.'
hist.x_labels = []
for value in range(2, max_result+1):
    hist.x_labels.append(str(value))
hist.x_title = 'result'
hist.y_title = 'Frequency of Result'

hist.add('D20 + D20', frequencies)
hist.render_to_file('dice_visual.svg')
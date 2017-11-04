#!/usr/bin/env python3
# -*- coding: utf-8 -*-

with open('C:\\PTW\\readtest.txt', 'r') as f:
    print(f.read(4))
    
    for line in f.readlines():
        print(line.strip())

        '''
with open('C:\\PTW\\readtest.txt', 'w') as f:
    f.write('1234567890')
        '''
        '''
with open('G:\\MNIST\\train-images.idx3-ubyte', 'rb') as f:
    print(f.read(100))
    '''
    
from io import StringIO
f = StringIO()
f.write('hello')
f.write(' ')
f.write('world!')
print(f.getvalue())

from io import BytesIO
f = BytesIO()
f.write('中文'.encode('UTF-8'))
print(f.getvalue())

import pickle
d = dict(name = 'Bob', age = 20, score = 90)
pickle.dumps(d)

with open('C:\\PTW\\dumptest.txt', 'wb') as f:
    pickle.dump(d,f)

with open('C:\\PTW\\dumptest.txt', 'rb') as f:
    d = pickle.load(f)

print(d)

import json
json_str = json.dumps(d)
print(json_str)
print(json.loads(json_str))

class student(object):
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score
        
s = student('Ana', 21, 92)

def student2dict(std):
    return {
        'name': std.name,
        'age': std.age,
        'score': std.score
    }

print(json.dumps(s, default = lambda obj: obj.__dict__))

def dict2student(d):
    return student(d['name'], d['age'], d['score'])
json_str = '{"age":19, "score":98, "name": "Jane"}'
print(json.loads(json_str, object_hook = dict2student))



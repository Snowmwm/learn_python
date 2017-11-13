#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import os

def get_imlist(path):
    '''返回目录中所有JPG图像的文件名列表'''
    return [os.path.join(path, f) for f in os.listdir(path) \
    if f.endswith('.jpg')] #or f.endswith('.png')

#格式转换
'''
filelist = get_imlist('C:\\PTW\\learn_python')
print(filelist)

for infile in filelist:
    outfile = os.path.splitext(infile)[0] + '.png'
    if infile != outfile:
        try:
            Image.open(infile).save(outfile)
        except IOError:
            print('cannot convert', infile)
'''
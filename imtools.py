#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import os

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]

filelist = get_imlist('C:\\PTW\\learn_python')
print(filelist)
    
for infile in filelist:
    outfile = os.path.splitext(infile)[0] + '.png'
    if infile != outfile:
        try:
            Image.open(infile).save(outfile)
        except IOError:
            print('cannot convert', infile)
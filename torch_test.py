#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#torch test
import torch
x = torch.FloatTensor([[0,1],[-1,-2]])
s = 1/(1+x)
print(s,x)

#cuda test
y = torch.Tensor([1,0])
yy = y.cuda()
print(yy)

#cudnn test
from torch.backends import cudnn
print(cudnn.is_acceptable(yy))
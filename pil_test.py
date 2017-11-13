#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image,ImageFont,ImageDraw
import os
import numpy as np

import imtools as it

#制作微信+1头像
'''
ImageFile = 'test.png'
SaveFile = 'test1.png'

def AddNumToImg(Imagefile,SaveFile):
    # 打开Imagefile，将其模式转换为RGBA
    with Image.open(Imagefile).convert('RGBA') as im:
        # 创建一个新图片，大小和模式直接使用Imagefile的
        txt = Image.new(im.mode,im.size)
        # 设置字体和字号
        font = ImageFont.truetype('msyh.ttf', 66)
        # 编辑txt新图片
        d = ImageDraw.Draw(txt)
        # 画一个圆，并且设置为红色
        d.ellipse((490,50,590,150), ('red'))
        # 增加一个数字，位置要处于上面的圆的中间，内容为1，字体为微软雅黑，填充颜色为白色，最后的fill里面的值可以为色值和颜色名称
        d.text((520, 55), '1', font=font, fill=(255, 255, 255))
        # 合并图片
        out = Image.alpha_composite(im,txt)
        # 保存图片
        out.save(SaveFile)
        # 展示保存后的图片
        out.show()
        
#AddNumToImg(ImageFile,SaveFile)
'''

#利用数组操作图像
'''
im1 = np.array(Image.open(ImageFile).convert('L'))
im2 = 255-im1 #对图像进行反相处理
im3 = (100.0/255) * im1 + 100 #将图像像素值变换到100——200区间
im4 = 255.0 * (im1 / 255.0)**2 #对图像像素值求平方后得到的图像(使暗的更暗)
im5, cdf = it.histeq(im) #直方图均衡化
im6 = it.compute_average(it.get_imlist('C:\\PTW'))#图像平均

out1 = Image.fromarray(np.uint8(im1))
out2 = Image.fromarray(np.uint8(im2))
out3 = Image.fromarray(np.uint8(im3))
out4 = Image.fromarray(np.uint8(im4))
out5 = Image.fromarray(np.uint8(im5))
out6 = Image.fromarray(np.uint8(im6))
out1.save('test1.jpg')
out2.save('test2.jpg')
out3.save('test3.jpg')
out4.save('test4.jpg')
out5.save('test5.jpg')
out6.save('test6.jpg')
'''







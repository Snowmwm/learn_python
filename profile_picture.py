#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image,ImageFont,ImageDraw
import os

ImageFile = 'test.png'
SaveFile = 'profile_picture.png'

def AddNumToImg(Imagefile,SaveFile):
    '''制作微信+1头像'''
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
        
AddNumToImg(ImageFile,SaveFile)
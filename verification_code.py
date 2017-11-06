#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

for n in range(16):
    #随机字母：
    def randChar():
        return chr(random.randint(65, 90))

    #随机颜色1：
    def randColor1():
        return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))
        
    #随机颜色2：
    def randColor2():
        return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))
        
    #大小240 x 60：
    width = 60 * 4
    height = 60
    image = Image.new('RGB', (width, height), (255, 255, 255))

    #创建font对象：
    font = ImageFont.truetype('arial.ttf', 36)

    #创建Draw对象：
    draw = ImageDraw.Draw(image)


    #填充每个像素：
    for x in range(width):
        for y in range(height):
            draw.point((x, y), fill = randColor1())
            
    #输出文字：
    for t in range(4):
        draw.text((60 * t + 12, 12),randChar(), font = font, fill = randColor2())
        
    #模糊：
    image = image.filter(ImageFilter.BLUR)
    image.save('code%d.jpg' % n, 'jpeg')
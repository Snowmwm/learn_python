#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygal

wm = pygal.maps.world.World()

#突出美洲
'''
wm.title = 'North, Central, and South America'
wm.add('North America', ['ca', 'mx', 'us'])
wm.add('Central America', ['bz', 'cr', 'gt', 'hn', 'ni', 'pa', 'sv'])
wm.add('South America', ['ar', 'bo', 'br', 'co', 'ec', 'gf', 'gy', 'pe', 'py', 'sr', 'uy', 've'])
'''

#显示三个北美国家的人口数量
wm.title = 'Populations of Countries in North America'
wm.add('North America', {'ca': 34126000, 'mx': 113423000, 'us': 309349000})

#保存地图
wm.render_to_file('na_populations.svg')
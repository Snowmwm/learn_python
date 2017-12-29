#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

'''
#figure
x = np.linspace(-1,1,50)
y1 = 2*x+1
y2 = x**2
"""
plt.figure()
plt.plot(x,y1)

plt.figure(num=3,figsize=(8,5))
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')
"""

#设置坐标轴
plt.figure()

plt.xlim((-2,2))
plt.ylim((-1,4))
plt.xlabel('X')
plt.ylabel('Y')
"""
new_ticks = np.linspace(-1,2,5)
plt.xticks(new_ticks)

plt.yticks([-2,-1.8,-1,1.22,3],
    [r'$really\ bad$',r'$bad$',r'$normal$',r'$good$',r'$really\ good$'])
"""
#gca = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))#['outward','axes']
ax.spines['left'].set_position(('data',0))


#Legend 图例
l1, = plt.plot(x,y2,label='up')
l2, = plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--',label='down')
plt.legend(handles=[l1,l2,],labels=['aaa','bbb' ],loc='best')
"""
 'best' : 0,          
 'upper right'  : 1,
 'upper left'   : 2,
 'lower left'   : 3,
 'lower right'  : 4,
 'right'        : 5,
 'center left'  : 6,
 'center right' : 7,
 'lower center' : 8,
 'upper center' : 9,
 'center'       : 10,
"""

#Annotation 标注
x0 = 1
y0 = 2*x0+1
plt.scatter(x0,y0,s=50,color='b') #散点图
plt.plot([x0,x0],[y0,0],'k--',lw=2.5) #黑色虚线

###### method1 ######
plt.annotate(r'$2x+1=%s$' %y0,xy=(x0,y0),xycoords='data',xytext=(+30,-30), 
    textcoords='offset points',fontsize=16,
    arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.3'))

###### method2 ######
plt.text(-2,3,r'$This\ is\ the\ some\ text.\ \mu\ \sigma_i\ \alpha_t$',
    fontdict=({'size':8,'color':'red'}))


#tick能见度

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='white',edgecolor='None',alpha=0.7))
'''

#散点图
'''
n = 1024
X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,n)
T = np.arctan2(X,Y) #

plt.scatter(X,Y,s=32,c=T,alpha=0.5)
#plt.scatter(np.arange(5),np.arange(5))

plt.xlim((-3,3))
plt.ylim((-3,3))
plt.xticks(())
plt.yticks(())
'''

#柱状图
'''
n = 12
X = np.arange(n)
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n) #产生0.5-1的随机数
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

plt.bar(X, +Y1,facecolor='#66ccff',edgecolor='white') #画柱状图
plt.bar(X, -Y2,facecolor='#ffcc66',edgecolor='white')

for x,y in zip(X,Y1):
    #ha: horizontal alignment 横向对齐
    #va: vertical alignment
    plt.text(x, y+0.05, '%.2f' %y, ha='center', va='bottom')
    
for x,y in zip(X,Y2):
    plt.text(x, -y-0.05, '%.2f' %y, ha='center', va='top')

plt.xlim(-.5, n)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())
'''

#等高线图 Contours
'''
def f(x,y):
    # the height function
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X,Y = np.meshgrid(x, y) #生成网格矩阵

plt.contourf(X,Y,f(X,Y),16,alpha=0.75,cmap=plt.cm.hot)

C = plt.contour(X,Y,f(X,Y),16,colors='black',linewidth=0.1)

plt.clabel(C, inline=True, fontsize=8)

plt.xticks(())
plt.yticks(())
'''

#Image图片
'''
a = np.array([0.23525, 0.29752, 0.38175,
              0.34512, 0.39281, 0.55555,
              0.41542, 0.60001, 0.76543]).reshape(3,3)

plt.imshow(a,interpolation='nearest', cmap='bone', origin='lower')
plt.colorbar(shrink=0.9)

plt.xticks(())
plt.yticks(())
'''

#3D 数据
'''
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

X = np.arange(-4,4,0.25)
Y = np.arange(-4,4,0.25)
X,Y = np.meshgrid(X,Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

#rstride 和 cstride 分别代表 row 和 column 的跨度
ax.plot_surface(X,Y,Z,rstride=1, cstride=1,cmap='rainbow')

ax.contourf(X,Y,Z,zdir='z',offset=-2, cmap='rainbow')

ax.set_zlim(-2,2)
'''

#Subplot 多合一显示
'''
plt.figure()

plt.subplot(211)
plt.plot([0,1],[0,1])

plt.subplot(234)
plt.plot([0,1],[0,2])

plt.subplot(235)
plt.plot([0,1],[0,3])

plt.subplot(236)
plt.plot([0,1],[0,4])
'''

#分格显示
###### m1:subplot2grid ######
"""
plt.figure()
ax1 = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=1)
ax1.plot([1,2],[2,1])
ax1.set_title('ax1 title')
ax2 = plt.subplot2grid((3,3),(1,0),colspan=2,rowspan=1)
ax3 = plt.subplot2grid((3,3),(1,2),colspan=1,rowspan=2)
ax4 = plt.subplot2grid((3,3),(2,0),colspan=1,rowspan=1)
ax5 = plt.subplot2grid((3,3),(2,1),colspan=1,rowspan=1)
"""

###### m2:gridspec ######
"""
import matplotlib.gridspec as gridspec

plt.figure()
gs = gridspec.GridSpec(3,3)

ax1 = plt.subplot(gs[0,:])
ax2 = plt.subplot(gs[1,:2])
ax3 = plt.subplot(gs[1:,2])
ax4 = plt.subplot(gs[-1,0])
ax5 = plt.subplot(gs[-1,-2])
"""

###### m3:subplots ######
"""
f,((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2,sharex=True,sharey=True)

ax11.scatter([1,3],[2,-1])

plt.tight_layout()
"""

#图中图
'''
fig = plt.figure()
# 创建数据
x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 3, 4, 2, 5, 8, 6]

left,bottom,width,height = 0.1,0.1,0.8,0.8
ax1 = fig.add_axes([left,bottom,width,height])
ax1.plot(x,y,'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('t1')

left,bottom,width,height = 0.2,0.6,0.25,0.25
ax1 = fig.add_axes([left,bottom,width,height])
ax1.plot(y,x,'b')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('t2')

plt.axes([.6,.2,.25,.25])
plt.plot(y[::-1],x,'g')
plt.xlabel('x')
plt.ylabel('y')
plt.title('t3')
'''

#次坐标轴
'''
x = np.arange(0, 10, 0.1)
y1 = 0.05 * x**2
y2 = -1 * y1

fig,ax1 = plt.subplots()
ax2 = ax1.twinx()  #对ax1调用twinx()方法，生成如同镜面效果后的ax2
ax1.plot(x,y1,'g-')
ax2.plot(x,y2,'b--')

ax1.set_xlabel('X data')
ax1.set_ylabel('Y1',color='g')
ax2.set_ylabel('Y2',color='b')
'''

#Animation 动画
from matplotlib import animation

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x+i/100))
    return line,

def init():
    line.set_ydata(np.sin(x))
    return line,
    
    
ani = animation.FuncAnimation(fig=fig, func=animate, frames=100,
    init_func=init, interval=20, blit=False)

plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
from tkinter import *
import tkinter.messagebox as mb

class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        self.nameInput = Entry(self)
        self.nameInput.pack()
        self.alertButton = Button(self, text='Hello', command=self.hello)
        self.alertButton.pack()
        
    def hello(self):
        name = self.nameInput.get() or 'world'
        mb.showinfo('Massage', 'Hello, %s' % name)

app = Application()
# 设置窗口标题:
app.master.title('Hello World')
# 主消息循环:
app.mainloop()
'''

import tkinter as tk

window = tk.Tk()
window.title('my window')
window.geometry('640x640')

#listbox
'''
var1 = tk.StringVar()

l = tk.Label(window, bg='yellow', width=8, textvariable=var1)
l.pack()

def print_selection():
    value = lb.get(lb.curselection())
    var1.set(value)

b = tk.Button(window, text='print selection', height=2, command=print_selection)
b.pack()

var2 = tk.StringVar()
var2.set([11,22,33,44])

lb = tk.Listbox(window, listvariable=var2)

list_items = [1, 2, 3, 4]
for item in list_items:
    lb.insert('end', item)
lb.insert(1,'first')
lb.insert(2,'second')
lb.delete(4)
lb.pack()
'''

#radiobutton
'''
var = tk.StringVar()
l = tk.Label(window, bg='yellow', width=24, text='empty')
l.pack()

def print_selection():
    l.config(text='you have selected ' + var.get())
    

r1 = tk.Radiobutton(window, text='Option A', variable=var, value='A', command=print_selection)
r1.pack()

r2 = tk.Radiobutton(window, text='Option B', variable=var, value='B', command=print_selection)
r2.pack()

r3 = tk.Radiobutton(window, text='Option C', variable=var, value='C', command=print_selection)
r3.pack()
'''

#scale
'''
l = tk.Label(window, bg='yellow', width=24, text='empty')
l.pack()

def print_selection(v):
    l.config(text='you have selected ' + v)

s = tk.Scale(window, label='try me', from_=5, to=11, orient=tk.HORIZONTAL,
    length=200, showvalue=0, tickinterval=3, resolution=0.01,
    command=print_selection)
s.pack()
'''

#checkbutton
'''
l = tk.Label(window, bg='yellow', width=24, text='empty')
l.pack()

def print_selection():
    if (var1.get() == 1) and (var2.get() == 0):    
        l.config(text='I love only Python')
    elif (var1.get() == 0) and (var2.get() == 1):    
        l.config(text='I love only C++')
    elif (var1.get() == 0) and (var2.get() == 0):    
        l.config(text='I do not love either')
    else:    
        l.config(text='I love both!')

var1 = tk.IntVar()
var2 = tk.IntVar()    
    
c1 = tk.Checkbutton(window, text='python', 
    variable=var1, onvalue=1, offvalue=0, command=print_selection)
c2 = tk.Checkbutton(window, text='C++',
    variable=var2, onvalue=1, offvalue=0, command=print_selection)

c1.pack()
c2.pack()
'''

#canvas
'''
canvas = tk.Canvas(window, bg='blue', height=320, width=640)
image_file = tk.PhotoImage(file='./thumbnail.png')
image = canvas.create_image(0,0, anchor='nw', image=image_file)
x0,y0,x1,y1 = 160,160,200,200
line = canvas.create_line(x0,y0,x1,y1)
oval = canvas.create_oval(x0,y0,x1,y1, fill='red')
canvas.pack()

def moveit():
    canvas.move(oval,0,2)

b = tk.Button(window,text='move', command=moveit)
b.pack()
'''

#menubar
'''
l = tk.Label(window, bg='yellow', width=24, text='')
l.pack()

counter = 0
def do_job():
    global counter
    l.config(text='do'+str(counter))
    counter+=1

menubar = tk.Menu(window)
filemenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='File', menu=filemenu)
filemenu.add_command(label='New', command=do_job)
filemenu.add_command(label='Open', command=do_job)
filemenu.add_command(label='Save', command=do_job)
filemenu.add_separator()
filemenu.add_command(label='Exit',command=window.quit)

editmenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='Edit', menu=editmenu)
editmenu.add_command(label='Cut', command=do_job)
editmenu.add_command(label='Copy', command=do_job)
editmenu.add_command(label='Paste', command=do_job)

submenu = tk.Menu(filemenu)
filemenu.add_cascade(label='Import',menu=submenu, underline=0)
submenu.add_command(label='Submenu1', command=do_job)

window.config(menu=menubar)
'''

#frame
'''
l = tk.Label(window, text='on the window').pack()

frm = tk.Frame(window)
frm.pack()

frm_l = tk.Frame(frm,)
frm_r = tk.Frame(frm)
frm_l.pack(side='left')
frm_r.pack(side='right')

tk.Label(frm_l, text='on the frm_l_1').pack()
tk.Label(frm_l, text='on the frm_l_2').pack()
tk.Label(frm_r, text='on the frm_r_1').pack()
'''

#messagebox
'''
import tkinter.messagebox as mb
def hit_me():
    #mb.showinfo(title='Hi', message='hhhhh')
    #mb.showwarning(title='Hi', message='ahhhhh')
    #mb.showerror(title='Hi', message='Error!')
    #mb.askquestion(title='Hi', message='hhhhh')
    #mb.askyesno(title='Hi', message='hhhhh')
    #mb.asktrycancel(title='Hi', message='hhhhh')
    mb.askokcancel(title='Hi', message='hhhhh')
tk.Button(window,text='hit me',command=hit_me).pack()
'''

#pack grid place
'''
#tk.Label(window,text=1).pack(side='top')
#tk.Label(window,text=2).pack(side='bottom')
#tk.Label(window,text=3).pack(side='left')
#tk.Label(window,text=4).pack(side='right')


for i in range(4):
    for j in range(3):
        tk.Label(window,text=1).grid(row=i,column=j,ipadx=20,ipady=20)

        
tk.Label(window,text=1).place(x=160,y=320,anchor='w')
'''



window.mainloop()
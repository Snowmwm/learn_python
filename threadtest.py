#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, threading
'''
def loop():
    print('thread %s is running...' % threading.current_thread().name)
    n=0
    while n<5:
        n = n + 1
        print('thread %s >>> %s' % (threading.current_thread().name, n))
        time.sleep(1)
    print('thread %s ended.' % threading.current_thread().name)
    
print('thread %s is running...' % threading.current_thread().name)
t = threading.Thread(target = loop, name = 'LoopThread')
t.start()
t.join()
print('thread %s ended.' % threading.current_thread().name)
'''
'''
balance = 0
lock = threading.Lock()

def change_it(n):
    # 先存后取，结果应该为0:
    global balance
    balance = balance + n
    balance = balance - n

def run_thread(n):
    for i in range(10000000):
        lock.acquire()
        try:
            change_it(n)
        finally:
            lock.release()
            
start = time.time()
t1 = threading.Thread(target = run_thread, args=(5,))
t2 = threading.Thread(target = run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
end = time.time()
print(balance)
print('Thread runs %.3f seconds.' % (end-start))
'''
#创建全局threadlocal对象：
local_school = threading.local()

def process_student():
    #获取当前线程关联的student：
    std = local_school.student
    print('Hello, %s (in %s)' % (std, threading.current_thread().name))

def process_thread(name):
    #绑定threadlocal的student：
    local_school.student = name
    process_student()
    
t1 = threading.Thread(target = process_thread, args = ('Alice',), name = 'Thread-A')
t2 = threading.Thread(target = process_thread, args = ('Bob',), name = 'Thread-B')
t1.start()
t2.start()
t1.join()
t2.join()
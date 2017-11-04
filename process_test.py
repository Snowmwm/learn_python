#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from multiprocessing import Pool, Process, Queue
import os, time, random
'''
def long_time_task(name):
    print('Run task %s (%s)...' % (name,os.getpid()))
    start = time.time()
    time.sleep(random.random()*3)
    end = time.time()
    print('Task %s runs %.2f seconds.' % (name, (end-start)))
    
if __name__=='__main__':
    print('Parent process %s' % os.getpid())
    p = Pool(3)
    for i in range(3):
        p.apply_async(long_time_task, args = (i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocessses done.')
''' 
'''
import subprocess

print('$ nslookup www.python.org')
r = subprocess.call(['nslookup', 'www.python.org'])
print('Exit code:', r)
'''

#写数据进程执行的代码：
def write(q):
    print('Process to white: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random,random(1))
        
#读数据进程执行的代码：
def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)
        
if __name__=='__main__':
    q = Queue()
    pw = Process(target = write, args = (q,))
    pr = Process(target = read, args = (q,))
    pw.start()
    pr.start()
    pw.join()
    pr.terminate()
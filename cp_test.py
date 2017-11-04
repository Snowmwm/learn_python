#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def consumer():
    r = ''

    print('*' * 10, '1', '*' * 10)
    while True:
        print('*' * 10, '2', '*' * 10)
        n = yield r
        if not n:
            print('*' * 10, '3', '*' * 10)
            return
        print('[CONSUMER] Consuming %s...' % n)
        r = '200 OK'

def produce(c):
    print('*' * 10, '4', '*' * 10)
    c.send(None)
    n = 0
    print('*' * 10, '5', '*' * 10)
    while n < 5:
        n = n + 1
        print('[PRODUCER] Producing %s...' % n)
        r = c.send(n)
        print('[PRODUCER] Consumer return: %s' % r)
    c.close()

c = consumer()
print('*' * 10, '6', '*' * 10)
produce(c)
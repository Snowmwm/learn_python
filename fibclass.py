#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Fib(object):
    def __init__(self):
        self.a, self.b = 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        if self.a > 1000000:
            raise StopIteration()
        return self.a

L1 = []
for n in Fib():
    L1.append(n)
print(L1[-5:])
print(L1[:20:4])
    
class Fib2(object):
    def __getitem__(self,n):
        if isinstance(n,int):
            a, b = 1, 1
            for x in range(n):
                a, b = b, a + b
            return a
        if isinstance(n,slice):
            start = n.start
            stop = n.stop
            if start is None:
                start = 0
            a, b = 1, 1
            L = []
            for x in range(stop):
                if x >= start:
                    L.append(a)
                a, b = b, a + b
            return L
        
f=Fib2()
print(f[50])
print(f[20:30])
print(f[:10])

if __name__=='__main__':
    import doctest
    doctest.testmod()
    
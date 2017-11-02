#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
'''
md5 = hashlib.md5()
md5.update('Hello World!'.encode('utf-8'))
md5.update('I am XueYe.'.encode('utf-8'))
print(md5.hexdigest())
'''
def get_md5(password):
    md5 = hashlib.md5()
    md5.update(password.encode('utf-8'))
    return md5.hexdigest()

db = {}

def register(username, password):
    db[username] = get_md5(password + username + 'the-Salt')

def login(username, password):
    if not username in db:
        print('账户不存在，请先注册！')
        return
    if db[username] == get_md5(password + username + 'the-Salt'):
        print('登录成功！')
    else:
        print('密码错误！')
    pass

def register_input():
    while True:
        username = input('请输入注册账户：')
        password = input('请输入注册密码：')
        if username != "" or password != "":
            register(username, password)
            return
        else:
            print('注册账户和密码不能为空，请重新输入...')

def login_input():
    while True:
        username = input('请输入登录账户：')
        password = input('请输入登录密码：')
        if username != "" or password != "":
            login(username, password)
            return
        else:
            print('注册账户和密码不能为空，请重新输入...')

if __name__ == '__main__':
    while True:
        print('注册请输入1，登录请输入2，输入其他退出！')
        select_type = input('请输入：')
        if select_type == '1':
            register_input()
        elif select_type == '2':
            login_input()
        else:
            exit()
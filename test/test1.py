# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：2019/12/1620:50
# 文件名：test1.py
# 开发工具：PyCharm
# 功能：测试全局变量

import time
import sys

def show():
    global a
    print(a)
    a = 3


def s():
    global a
    print(a)
    a = 5


if __name__ == '__main__':
    a = 1
    show()
    print(a)
    s()
    print(a)

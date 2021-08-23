# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：2019/12/1921:39
# 文件名：test2.py
# 开发工具：PyCharm
# 功能：列表中元素是否有小于特定值的元素

import numpy as np

a = np.array([['#1', 1, 3, 3], ['#2', 2, 4, 5]])
data = list(a.values())
path_length = [int(x) for x in np.array(data[:, 1])]
print((np.array(path_length) < 2))

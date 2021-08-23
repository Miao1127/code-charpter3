# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：2019/12/1514:38
# 文件名：combine_distance.py
# 开发工具：PyCharm
# 功能：测试将各个usv计算出来的距离各个分区的距离列表合并成一个矩阵

import numpy as np
from file_operator import read_csv


u1_d = read_csv.run('../distance/u2z_usv1_distance.csv')
u2_d = read_csv.run('../distance/u2z_usv2_distance.csv')
u3_d = read_csv.run('../distance/u2z_usv3_distance.csv')
u4_d = read_csv.run('../distance/u2z_usv4_distance.csv')
u5_d = read_csv.run('../distance/u2z_usv5_distance.csv')

r, c = np.array(u1_d).shape
d = np.zeros((1, c))
d = np.append(d, u1_d, axis=0)
d = np.append(d, u2_d, axis=0)
d = np.append(d, u3_d, axis=0)
d = np.append(d, u4_d, axis=0)
d = np.append(d, u5_d, axis=0)

print(d)



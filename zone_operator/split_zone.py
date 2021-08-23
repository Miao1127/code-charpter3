# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：2019/12/912:18
# 文件名：split_zone.py
# 开发工具：PyCharm
# 功能：对超过分区均衡系数的大分区进行拆分

import numpy as np
from zone_operator import vertex_zone
from file_operator import read_json, dict2json
import time


def run(zone_dict, z_ratio):
    """
    功能：对大分区进行拆分
    分析：首先界定大的分区标准，通过最大分区中栅格数目与最小分区中栅格分区数目的比值定义，分区有两种拆分方式，一种是用平行于
    Y轴的直线拆分，一种是用平行于X轴的直线拆分，选择标准是垂直于分区的最长边的线进行拆分，使拆分后的两个分区栅格数量大致相当。
    ；
    :param zone_dict:分区字典
    :param z_ratio: 分区均衡系数，初始值最大值设定为集群个体数目的倒数，目的是保证初始分配每个usv能够分得至少一个分区
    :return:拆分后的分区字典
    """

    while True:
        # 1.找到最小分区的栅格数，根据最大分区与最小分区的比例找到需要拆分的分区
        list_zone = []
        grid_sum = 0
        for key in zone_dict:
            grid_sum += len(zone_dict[key])
        for key in zone_dict:
            if len(zone_dict[key]) / grid_sum > z_ratio:
                list_zone = np.append(list_zone, key)
        if len(list_zone) == 0:
            break
        else:
            print(list_zone)
        # 2.找出分区的最长边，顶点集可能存在的三种情况，有两个元素（分区为一列或者一行栅格）
        # 例如：{'#3': array([[ 47. ,  49. ,   0.5],
        #        [131. ,  49. ,   0.5]]]),
        #        '#4':array([[ 51. ,  60. ,   0.5],
        #        [51. ,  70. ,   0.5]])}
        # 有三个元素（分区为三角形）
        # 例如：
        # {'#3': array([[ 47. ,  49. ,   0.5],
        #        [ 51. ,  60. ,   0.5],
        #        [131. ,  60. ,   0.5]])}
        #  或者：
        #  {'#3': array([[ 47. ,  49. ,   0.5],
        #        [131. ,  49. ,   0.5],
        #        [131. ,  60. ,   0.5]])}
        # 有四个元素（四边形，边不一定是直线，此种情况最多）
        # 例如：{'#3': array([[ 47. ,  49. ,   0.5],
        #        [131. ,  49. ,   0.5],
        #        [ 51. ,  60. ,   0.5],
        #        [131. ,  60. ,   0.5]])}
        operate_dict = {}
        for key in list_zone:
            operate_dict[key] = zone_dict[key]
        zone_point = vertex_zone.run(operate_dict)  # 计算每个分区的顶点集
        x_zone = {}  # 存储用平行于x轴直线拆分的分区名称和相应的行号范围
        y_zone = {}  # 存储用平行于y轴直线拆分的分区名称和相应的列号范围
        for name in list_zone:
            if len(zone_point[name]) == 2:  # 顶点集两个元素
                if zone_point[name][0][0] - zone_point[name][1][0] != 0:  # 一列栅格
                    x_zone[name] = [zone_point[name][0][0]]
                    x_zone[name] = np.append(x_zone[name], [zone_point[name][1][0]])
                else:  # 一行栅格
                    y_zone[name] = [zone_point[name][0][1]]
                    y_zone[name] = np.append(y_zone[name], [zone_point[name][1][1]])
            if len(zone_point[name]) == 3:  # 顶点集三个元素
                if zone_point[name][0][1] - zone_point[name][1][1] == 0:  # 判断第一、二顶点是同列
                    x_zone[name] = [zone_point[name][0][0]]
                    x_zone[name] = np.append(x_zone[name], [zone_point[name][1][0]])
                else:  # 第二、三顶点是同列
                    x_zone[name] = [zone_point[name][1][0]]
                    x_zone[name] = np.append(x_zone[name], [zone_point[name][2][0]])
            if len(zone_point[name]) == 4:  # 顶点集四个元素
                row_max = max(zone_point[name][0][0], zone_point[name][2][0], zone_point[name][1][0],
                                  zone_point[name][3][0])
                row_min = min(zone_point[name][0][0], zone_point[name][2][0], zone_point[name][1][0],
                                  zone_point[name][3][0])
                col_0 = abs(zone_point[name][0][1] - zone_point[name][2][1])
                if row_max - row_min > col_0:  # 行数大于列数
                    x_zone[name] = [row_min]
                    x_zone[name] = np.append(x_zone[name], [row_max])
                else:  # 列数大于行数
                    y_zone[name] = [zone_point[name][0][1]]
                    y_zone[name] = np.append(y_zone[name], [zone_point[name][2][1]])
        # 3.选择合适的行号或者列号，对分区进行拆分，新建分区，用来容纳拆分出的栅格
        # 利用平行与Y轴的直线对分区进行拆分
        num = len(zone_dict) + 1  # 用于新分区的编号计数
        for key in y_zone:
            # 统计分区每一列栅格数目
            start = int(y_zone[key][0])  # 起始列号
            end = int(y_zone[key][1] + 1)  # 终止列号
            count = 0
            num_in_col = []
            for i in range(start, end):
                for j in range(len(zone_dict[key])):
                    if zone_dict[key][j][1] == i:
                        count += 1
                num_in_col = np.append(num_in_col, count)
                count = 0
            # 寻找划分位置
            sum_left = 0  # 划分线左侧栅格总和
            sum_right = len(zone_dict[key])  # 划分线右侧栅格总和
            for i in range(len(num_in_col)):
                sum_left += num_in_col[i]
                sum_right -= num_in_col[i]
                if sum_left > sum_right:
                    position = int(sum_left - num_in_col[i])  # 记录分区划分位置
                    break
            # 新建分区
            new_key = '#' + str(num)
            num += 1
            zone_dict[new_key] = zone_dict[key][position:len(zone_dict[key])]
            zone_dict[key] = zone_dict[key][0:position]

        # 利用平行与X轴的直线对分区进行拆分
        temp_dict = {}
        for key in x_zone:
            # 统计每行栅格数
            start = int(x_zone[key][0])  # 起始行号
            end = int(x_zone[key][1] + 1)  # 终止行号
            count = 0
            num_in_row = []
            for i in range(start, end):  # 遍历每一行
                for j in range(len(zone_dict[key])):
                    if zone_dict[key][j][0] == i:  # 找到行号为i的所有栅格
                        count += 1
                num_in_row = np.append(num_in_row, count)
                count = 0
            # 寻找划分位置
            sum_up = 0  # 划分线上侧栅格总和
            sum_down = len(zone_dict[key])  # 划分线下侧栅格总和
            for i in range(len(num_in_row)):
                sum_up += num_in_row[i]
                sum_down -= num_in_row[i]
                if sum_up > sum_down:
                    position = i + start  # 记录分区划分位置
                    break
            # 新建分区
            new_key = '#' + str(num)
            num += 1
            flag_1 = 0
            flag_2 = 0
            for i in range(len(zone_dict[key])):   # 新分区
                for j in range(position, end):
                    if zone_dict[key][i][0] == j:
                        if flag_1 == 0:
                            temp_dict[new_key] = [zone_dict[key][i]]
                            flag_1 = 1
                        else:
                            temp_dict[new_key] = np.append(temp_dict[new_key], [zone_dict[key][i]], axis=0)

            for i in range(len(zone_dict[key])):  # 旧分区剩余栅格
                for j in range(start, position):
                    if zone_dict[key][i][0] == j:
                        if flag_2 == 0:
                            temp_dict[key] = [zone_dict[key][i]]
                            flag_2 = 1
                        else:
                            temp_dict[key] = np.append(temp_dict[key], [zone_dict[key][i]], axis=0)
        for key in temp_dict:
            zone_dict[key] = temp_dict[key]
    dict2json.run(zone_dict, 'split_dict')
    return zone_dict


if __name__ == '__main__':
    # 测试
    zone_ratio = 0.1
    data = read_json.run('zone_dict.json')
    result = run(data, zone_ratio)
    print(result)

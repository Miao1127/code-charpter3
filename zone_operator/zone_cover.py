# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：2019/12/1021:20
# 文件名：zone_cover.py
# 开发工具：PyCharm
# 功能：分区遍历搜索路径生成算法（zigzag方式）
import numpy as np
from file_operator import read_json


def run(start, grid_list):
    """
    输入参数为起点栅格编号，例如[1 2 0.5]，分区栅格列表，例如，[[1 2 0.5], [2 3 0.5]]
    输出为列表形式的路径，例如，[[1 2 0.5] [2 3 0.5]]
    """

    # 1.计算除去首尾两列的中间栅格列数，最右侧列编号-最左侧列编号-1
    n_other = int(grid_list[-1][1]) - int(grid_list[0][1])   # 去掉起点所在列

    # 2.判断起点栅格是否为分区左上角栅格，即栅格列表的第一元素
    if (start[0:2] == grid_list[0][0:2]).all():
        c_path = grid_list[np.where(grid_list[:, 1] == start[1])]
        for n in range(1, n_other + 1):
            if n % 2 == 1:
                middle_grid = grid_list[np.where(grid_list[:, 1] == start[1] + n)][::-1]  # 奇数中间列,逆序
            else:
                middle_grid = grid_list[np.where(grid_list[:, 1] == start[1] + n)]        # 偶数中间列

            if len(middle_grid) > 0:
                c_path = np.append(c_path, middle_grid, axis=0)

    # 3.判断起点栅格是否为分区左下角栅格，即栅格列表的第一列的最后一个栅格
    elif start[1] == grid_list[0][1] and start[0] != grid_list[0][0]:
        c_path = grid_list[np.where(grid_list[:, 1] == start[1])][::-1]
        for n in range(1, n_other + 1):
            if n % 2 == 0:
                middle_grid = grid_list[np.where(grid_list[:, 1] == start[1] + n)][::-1]  # 偶数中间列,逆序
            else:
                middle_grid = grid_list[np.where(grid_list[:, 1] == start[1] + n)]        # 奇数中间列

            if len(middle_grid) > 0:
                c_path = np.append(c_path, middle_grid, axis=0)

    # 4.判断起点栅格是否为分区右下角栅格，即栅格列表的最后一个栅格
    elif (start[0:2] == grid_list[-1][0:2]).all():
        c_path = grid_list[np.where(grid_list[:, 1] == start[1])][::-1]
        for n in range(1, n_other + 1):  # 中间列从后往前
            if n % 2 == 0:
                middle_grid = grid_list[np.where(grid_list[:, 1] == start[1] - n)][::-1]  # 偶数中间列,逆序
            else:
                middle_grid = grid_list[np.where(grid_list[:, 1] == start[1] - n)]        # 奇数中间列

            if len(middle_grid) > 0:
                c_path = np.append(c_path, middle_grid, axis=0)

    # 5.判断起点栅格是否为分区右上角栅格，即栅格列表的最后一列的第一个栅格
    else:
        c_path = grid_list[np.where(grid_list[:, 1] == start[1])]
        for n in range(1, n_other + 1):  # 中间列从后往前
            if n % 2 == 1:
                middle_grid = grid_list[np.where(grid_list[:, 1] == start[1] - n)][::-1]  # 奇数中间列,逆序
            else:
                middle_grid = grid_list[np.where(grid_list[:, 1] == start[1] - n)]        # 偶数中间列

            if len(middle_grid) > 0:
                c_path = np.append(c_path, middle_grid, axis=0)

    return c_path[:, 0:2]


if __name__ == '__main__':
    zone_num = '#2'
    # 加载分区顶点信息
    v_dict = read_json.run('../zone_operator/vertex_dict.json')
    start_point = v_dict[zone_num][3][0:2]
    # 加载分区信息
    zone_dict = read_json.run('../zone_operator/split_dict.json')
    zone_list = zone_dict[zone_num]
    print('输入量：')
    print('start_point:%s' % start_point)
    print('zone_list:%s' % zone_list)
    path = run(start_point, zone_list)
    print('输出量：')
    print('path:%s' % path)

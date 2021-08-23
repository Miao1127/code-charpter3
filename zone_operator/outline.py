# _*_ coding:utf-8 _*_
# 103中山分队
# 开发人员：Miao
# 开发时间：2019/12/818:38
# 文件名：outline.py
# 开发工具：PyCharm
# 功能：统计分区轮廓的栅格，方向为逆时针

from file_operator import dict2json, read_json
import numpy as np


def run(zone_dict):
    """
    功能：计算出每个分区的轮廓，用于对分区涂色
    分析：分区轮廓由分区最左侧的一列、去掉首尾列的每一列最后一个栅格（栅格顺序为从左到右）、最右侧栅格（栅格顺序从下往上）、
    除去首尾列每列的第一的栅格（栅格顺序为从右到左）
    :param zone_dict: 分区字典
    :return: 返回逆时针的轮廓栅格序列列表
    """
    outline_dict = {}
    for key in zone_dict:
        # 计算除去首尾两列的中间栅格列数，最右侧列编号-最左侧列编号-1
        n_middle = int(zone_dict[key][-1][1]) - int(zone_dict[key][0][1]) - 1

        # 1.取出分区最左侧一列栅格
        outline_dict[key] = zone_dict[key][np.where(zone_dict[key][:, 1] == zone_dict[key][0][1])]

        # 2.取出除掉首尾两列的其他列的第一个和最后一个栅格编号
        head_list = []     # 存储每列第一个栅格编号
        for n in range(1, n_middle + 1):
            middle_grid = zone_dict[key][np.where(zone_dict[key][:, 1] == int(zone_dict[key][0][1]) + n)]  # 中间列
            if len(middle_grid) > 0:
                outline_dict[key] = np.append(outline_dict[key], [middle_grid[-1]], axis=0)  # 追加列尾
                if len(head_list) == 0:                                                      # 记录列首
                    head_list = [middle_grid[0]]
                else:
                    head_list = np.append(head_list, [middle_grid[0]], axis=0)

        # 3.取出分区最右侧一列栅格编号，并从下往上的顺序添加到轮廓字典中
        right_grid_line = zone_dict[key][np.where(zone_dict[key][:, 1] == zone_dict[key][-1][1])][::-1]
        outline_dict[key] = np.append(outline_dict[key], right_grid_line, axis=0)

        # 4.将列首的栅格编号列表逆序后添加到轮廓字典中
        outline_dict[key] = np.append(outline_dict[key], head_list[::-1], axis=0)
        dict2json.run(outline_dict, 'outline_dict')
    return outline_dict


if __name__ == '__main__':
    # 测试
    data = read_json.run('split_dict.json')
    result = run(data)
    print(result)

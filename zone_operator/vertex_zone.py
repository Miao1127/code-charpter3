# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：2019/12/912:21
# 文件名：vertex_zone.py
# 开发工具：PyCharm
# 功能：统计每个分区的顶点集合
import numpy as np
from file_operator import read_json, dict2json


def run(zone_dict):
    """
    功能：取出每个分区的四个顶点
    分析：由于采用的是Y射线法，所以除了最左侧和最优侧的分区外，其他分区顶点必定处在分区的最左侧和最右侧的栅格列的两端
    而最左侧分区的有效顶点必定处在分区的最右侧栅格列两端，最右侧分区的有效顶点必行存在与分区的最左侧栅格列的两端，故分
    区顶点选择思路为：分区栅格列表第一个元素和最后一个元素一定是顶点，只需找到同处一列的另一端的栅格号即可
    :param zone_dict: 分区字典
    :return: 返回包含分区四个顶点的字典
    """
    vertex_dict = {}
    for key in zone_dict:
        # 取出分区最左侧列中第一个栅格
        vertex_dict[key] = np.array([zone_dict[key][0]])
        # 取出分区最左侧列中最后一个栅格
        if len(np.where(zone_dict[key][:, 1] == zone_dict[key][0][1])[0]) > 1:    # np.where()函数返回符合条件的下标
            new_vertex = zone_dict[key][np.where(zone_dict[key][:, 1] == zone_dict[key][0][1])[0][-1]]
            vertex_dict[key] = np.append(vertex_dict[key], [new_vertex], axis=0)

        # 取出分区最右侧列中第一个栅格
        if zone_dict[key][0][1] != zone_dict[key][-1][1]:  # 判断分区是否只有一列栅格
            if len(np.where(zone_dict[key][:, 1] == zone_dict[key][-1][1])[0]) > 1:
                new_vertex = zone_dict[key][np.where(zone_dict[key][:, 1] == zone_dict[key][-1][1])[0][0]]
                vertex_dict[key] = np.append(vertex_dict[key], [new_vertex], axis=0)
            # 取出分区最最右侧列中最后一个栅格
            vertex_dict[key] = np.append(vertex_dict[key], [zone_dict[key][-1]], axis=0)
    return vertex_dict


if __name__ == '__main__':
    # 测试
    data = read_json.run('split_dict.json')
    result = run(data)
    dict2json.run(result, 'vertex_dict')
    print(result)

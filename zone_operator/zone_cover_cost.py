# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：2019/12/1016:20
# 文件名：zone_cover_cost.py
# 开发工具：PyCharm
# 功能：统计分区内栅格数目，以列表的形式输出，只是单纯的统计栅格数目作为估算分区遍历代价，
# 还有改进空间，实际上还需要考虑分区内的转弯次数，分区形状，分区内风速和水流速度对usv航行的影响。

from file_operator import read_json


def run(zone_dict):
    """
    计算每个分区的价值
    :return:价值列表
    """
    num = len(zone_dict)
    cost_list = []
    for k in range(0, num):
        key = '#' + str(k + 1)
        cost_list.append(len(zone_dict[key]))
    return cost_list


if __name__ == '__main__':
    # 测试
    data = read_json.run('split_dict.json')
    result = run(data)
    print(result)
# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：2019/12/914:59
# 文件名：zone_value.py
# 开发工具：PyCharm
# 功能：计算分区价值
from file_operator import read_json


def run(zone_dict):
    """
    计算每个分区的价值
    :return:价值列表
    """
    num = len(zone_dict)
    value_list = []
    for k in range(0, num):
        key = '#' + str(k + 1)
        value_list.append(sum(zone_dict[key][:, 2]))
    return value_list


if __name__ == '__main__':
    # 测试
    data = read_json.run('split_dict.json')
    result = run(data)
    print(result)

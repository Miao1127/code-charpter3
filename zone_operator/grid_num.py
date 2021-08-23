# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：2019/12/912:56
# 文件名：grid_num.py
# 开发工具：PyCharm
# 功能：统计每个分区内栅格数目
from file_operator import dict2json, read_json


def run(zone_dict):
    """
    功能：统计每个分区内包含的栅格数目
    :param zone_dict:
    :return: 返回每个分区包含栅格数目的字典
    """
    grid_num = {}
    for k in range(len(zone_dict)):
        key = "#" + str(k + 1)
        grid_num[key] = len(zone_dict[key])
    return grid_num


if __name__ == '__main__':
    # 测试
    data = read_json.run('split_dict.json')
    result = run(data)
    dict2json.run(result, 'grid_num')
    print(result)
 
    # 统计总栅格数
    total = 0
    for kk in result:
        total += int(result[kk])
    print(total)

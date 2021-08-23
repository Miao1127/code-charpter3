# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：2019/12/911:15
# 文件名：y_zoning.py
# 开发工具：PyCharm
# 功能：Y射线法初步划分分区

import numpy as np
import time
from file_operator import read_csv, dict2json


def run(grid_data):
    """
    功能：利用Y射线法对任务进行分区
    :param grid_data: 任务区域的栅格信息
    :return: 划分好的分区字典
    """
    # 将需要在垂直方向上进行扫描线处理的矩阵转置，从而转变到对行向量进行处理
    grid_data = np.array(grid_data)
    rows, cols = grid_data.shape
    new_map = np.zeros((rows, cols, 2))   # 定义包含分区编号的栅格新地图
    for i in range(rows):
        for j in range(cols):
            new_map[i][j][0] = grid_data[i][j]
    # 从前往后查找0，将其变为0（代表障碍物），直至遇到不为0的数值为止
    for i in range(rows):
        for j in range(cols):
            if grid_data[i][j] > 0:
                break
            else:
                grid_data[i][j] = 0
    # 从后往前查找0，将其变为0（代表障碍物），直至遇到不为0的数值为止
    for i in range(rows):
        for j in range(cols):
            if grid_data[i][-j - 1] > 0:
                break
            else:
                grid_data[i][-j - 1] = 0
    print(grid_data)
    # 在处理后的矩阵中找到存在0元素的列向量索引，相邻的索引为同一个障碍物,index每个元素代表当前扫描线穿过的障碍物数量
    index = np.zeros(rows)
    for i in range(rows):
        for j in range(cols):
            if grid_data[i][j] == 0:
                index[i] += 1

    start = 0  # 用于存储空闲栅格片段起始位置索引
    open_dict = {}  # 用于存储射线扫描算法处于变化的分区的栅格信息（栅格编号和对应的概率值）
    closed_dict = {}  # 用于存储不再变化的分区的栅格信息
    num = 1  # 分区起始编号
    part_num = 0  # 记录空闲栅格片段的数量

    # 对每列的栅格进行遍历，对空闲栅格区域进行分区划分
    for i in range(cols):
        # 若本列栅格全部为被占据栅格，则不作处理，跳到下一列
        if sum(grid_data[:, i] == 0) == rows:
            continue
        # 若本列存在空闲栅格，则选出空闲栅格片段
        for j in range(rows):
            if start == 0 and grid_data[j][i] != 0:    # 寻找空闲栅格片段的起始位置索引
                start = j
            elif start != 0 and grid_data[j][i] == 0:  # 寻找空闲栅格片段的终止位置索引
                end = j
                # 存储空闲栅格片段的起始位置到结束位置的栅格编号
                if part_num == 0:
                    temp = np.array([[start, end, i]])  # 存储第一个空闲栅格片段编号范围(起始行，终止行，所在列)
                else:
                    temp = np.append(temp, np.array([[start, end, i]]), axis=0)  # 将一列的空闲栅格片段合并到一个列表中
                part_num += 1
                start = 0
        # ☆☆☆☆重要提示☆☆☆☆：temp中存储了每列的空闲栅格分段的范围
        # 根据遇到的第一列有空闲栅格的片段数量，初始创建对应数量的分区
        if len(open_dict) == 0 and len(closed_dict) == 0:
            for ii in range(part_num):  # 遍历每个片段
                key = '#' + str(num)    # 字典关键字-分区名称，分区编号从1开始
                num += 1
                open_dict[key] = np.array([[temp[ii][0], i, grid_data[temp[ii][0], i]]])  # 将片段的第一个空闲栅格追加到分区内
                for iii in range(temp[ii][0] + 1, temp[ii][1]):  # 将片段的剩余空闲栅格追加到分区内
                    open_dict[key] = np.append(open_dict[key], [[iii, i, grid_data[iii, i]]], axis=0)

        # 若open字典或closed字典中已经存在分区，表示已有分区生成
        # 比较open字典中的分区数量和空闲栅格片段数量（通过调整open字典中分区数量，使分区数量和空闲栅格片段数量对等）
        else:

            # 1.open字典中的分区数量小于空闲栅格片段数量情况下(射线穿过的障碍物数目增加），需要扩充open字典中的分区数量
            # 将open字典中分区内空闲栅格与多个当前列中空闲栅格片段出现重叠的分区移动到closed字典中，并为出现行列号重叠的每
            # 个空闲栅格分段在open字典中新建分区，将其一一存储；而open字典中只与一个分区产生行列号重叠的空闲栅格分段，将此
            # 分段存储到该分区内
            if len(open_dict) < part_num:
                # 遍历open字典中分区内空闲栅格，并与本列扫描得到的空闲栅格做比较
                # 提取出与当前分区有行号重叠的片段
                for k in list(open_dict.keys()):  # 由于需要对open字典进行循环遍历改动，需遍历open字典中的关键字
                    grid_id_open = open_dict[k][:, 0:2]  # 提取每个分区内存储的空闲栅格行列号

                    # 提取与当前open字典中有行号重叠的片段
                    temp_part = np.array([[0, 0, 0]])
                    flag = 0
                    for ii in range(part_num):  # 遍历每个片段，并与open字典中的分区中栅格进行对比
                        # 存储空闲栅格片段栅格编号
                        grid_id = np.array([[temp[ii][0], i - 1]])  # 将片段的第一个空闲栅格追加到分区内，列号减1是为了与上一列比较
                        for iii in range(temp[ii][0] + 1, temp[ii][1]):  # 将片段的剩余空闲栅格追加到分区内
                            grid_id = np.append(grid_id, [[iii, i - 1]], axis=0)
                        # 比较当前列中空闲栅格行列号和open字典中存储的空闲栅格的行列号
                        # 判断左移一列的当前空闲栅格片段是否与open字典中的分区栅格有重叠
                        # 比较当前列中空闲栅格行列号和open字典中存储的空闲栅格的行列号
                        for j in range(len(grid_id)):
                            for jj in range(len(grid_id_open)):
                                if (grid_id[j] == grid_id_open[jj]).all() and flag == 0:
                                    temp_part = np.array([temp[ii]])
                                    flag = 1
                                    break
                                elif (grid_id[j] == grid_id_open[jj]).all() and flag != 0:
                                    temp_part = np.append(temp_part, [temp[ii]], axis=0)
                                    break
                            else:
                                continue
                            break

                    if len(temp_part) == 1:  # 分区只有一个片段与其重叠时，将该片段加入此分区
                        cover_array = np.array([[temp_part[0][0], i, grid_data[temp_part[0][0], i]]])
                        for iii in range(temp_part[0][0] + 1, temp_part[0][1]):  # 将片段的剩余空闲栅格追加到分区内
                            cover_array = np.append(cover_array, [[iii, i, grid_data[iii, i]]], axis=0)
                        open_dict[k] = np.append(open_dict[k], cover_array, axis=0)
                    elif len(temp_part) > 1:
                        # 将出现多次重叠的分区移动到closed字典中
                        closed_dict[k] = open_dict[k]  # 复制键值
                        open_dict.pop(k)  # 删除键值
                        # n个片段与当前遍历分区有重叠，再创建n个新分区，将n个片段添加到分区内
                        for j in range(len(temp_part)):  # 遍历每个片段
                            key = '#' + str(num)  # 字典关键字-分区名称，分区编号从1开始
                            num += 1
                            # 将片段的第一个空闲栅格追加到分区内
                            open_dict[key] = np.array([[temp_part[j][0], i, grid_data[temp_part[j][0], i]]])
                            for iii in range(temp_part[j][0] + 1, temp_part[j][1]):  # 将片段的剩余空闲栅格追加到分区内
                                open_dict[key] = np.append(open_dict[key], [[iii, i, grid_data[iii, i]]], axis=0)

            # 2.open字典中的分区数量大于空闲栅格片段数量情况下(射线穿过的障碍物数目减少），需要移除open字典中的部分分区
            # 将open字典中多个分区内空闲栅格与一个当前列中空闲栅格片段出现重叠的分区移动到closed字典中，并此片段创建新的分区
            # 而open字典中只与一个分区产生行列号重叠的空闲栅格分段，将此分段存储到该分区内
            elif len(open_dict) > part_num:
                # 遍历当前扫描列中的空闲栅格片段，并与open字典中的不同分区内空闲栅格做比较
                for ii in range(part_num):  # 遍历每个片段
                    flag = 0
                    # 存储空闲栅格片段栅格编号
                    grid_id = np.array([[temp[ii][0], i - 1]])  # 将片段的第一个空闲栅格追加到分区内，列号减1是为了与上一列比较
                    for iii in range(temp[ii][0] + 1, temp[ii][1]):  # 将片段的剩余空闲栅格追加到分区内
                        grid_id = np.append(grid_id, [[iii, i - 1]], axis=0)
                    # 比较当前列中空闲栅格行列号和open字典中存储的空闲栅格的行列号
                    # 判断左移一列的当前空闲栅格片段是否与open字典中的分区栅格有重叠
                    key_array = []  # 用于存储与栅格片段有重叠的分区名称
                    for k in list(open_dict.keys()):  # 由于需要对open字典进行循环遍历改动，需取出open字典键值进行遍历
                        grid_id_open = open_dict[k][:, 0:2]  # 提取每个分区内存储的空闲栅格行列号
                        # 比较当前列中空闲栅格行列号和open字典中存储的空闲栅格的行列号
                        index_id = []
                        for j in range(len(grid_id)):
                            for jj in range(len(grid_id_open)):
                                if (grid_id[j] == grid_id_open[jj]).all():
                                    index_id.append(j)
                        if len(index_id) > 0:
                            flag += 1
                            # 将与当前遍历分区有重叠的片段栅格信息取出
                            key_array = np.append(key_array, k)

                    if flag == 1:  # 分区只有一个片段与其重叠时，将该片段加入此分区
                        open_dict[key_array[0]] = np.append(open_dict[key_array[0]], [[temp[ii][0], i, grid_data[temp[ii][0], i]]], axis=0)
                        for iii in range(temp[ii][0] + 1, temp[ii][1]):  # 将片段的剩余空闲栅格追加到分区内
                            open_dict[key_array[0]] = np.append(open_dict[key_array[0]], [[iii, i, grid_data[iii, i]]], axis=0)
                    elif flag > 1:
                        # 将与同一分段出现重叠的多个分区移动到closed字典中
                        for j in range(len(key_array)):
                            closed_dict[key_array[j]] = open_dict[key_array[j]]  # 复制键值
                            open_dict.pop(key_array[j])  # 删除键值
                        # 新建1个分区存储此片段
                        key = '#' + str(num)  # 字典关键字-分区名称，分区编号从1开始
                        num = num + 1
                        # 将片段的第一个空闲栅格追加到分区内
                        open_dict[key] = np.array([[temp[ii][0], i, grid_data[temp[ii][0], i]]])
                        for iii in range(temp[ii][0] + 1, temp[ii][1]):  # 将片段的剩余空闲栅格追加到分区内
                            open_dict[key] = np.append(open_dict[key], [[iii, i, grid_data[iii, i]]], axis=0)

            # 3.当栅格片段与open字典中的分区数量相等时，找到对应存在重叠的片段和分区，将片段加入到分区内
            # 空闲栅格片段与open字典中分区数目相同时，open字典中的分区一定已经存储了一部分空闲栅格，所以只要
            # 将当前列的空闲栅格左移一列，与open字典中的分区内进行行列号比较即可，只要两者存在重叠区域，即当前片段
            # 就可以分配给相应的分区，需要注意两类特殊情况：1.若干个片段和同一分区存在行号重叠；2. 若干个分区与同
            # 一片段行号重叠，所以需要首先判断一下每个片段是否与两个分区存在行号重叠，一旦出现那么将open字典中的所
            # 有分区移动到closed字典中，对应片段数量建立新的分区，将片段一一存入
            else:
                for k in list(open_dict.keys()):
                    grid_id_open = open_dict[k][:, 0:2]  # 提取每个分区内存储的空闲栅格行列号
                    # 1.查找是否存在一个分区与两个片段同时存在行号重叠
                    # 提取与当前open字典中有行号重叠的片段
                    temp_part = np.array([[0, 0, 0]])
                    flag = 0
                    # 提取当前扫描列的空闲栅格片段中栅格编号
                    if len(temp_part) == 1:
                        for ii in range(len(temp)):  # 遍历每个片段
                            # 存储空闲栅格片段栅格编号
                            grid_id = np.array([[temp[ii][0], i - 1]])  # 将片段的第一个空闲栅格追加到分区内，列号减1是为了与上一列比较
                            for iii in range(temp[ii][0] + 1, temp[ii][1]):  # 将片段的剩余空闲栅格追加到分区内
                                grid_id = np.append(grid_id, [[iii, i - 1]], axis=0)
                            # 比较当前列中空闲栅格行列号和open字典中存储的空闲栅格的行列号
                            # 判断左移一列的当前空闲栅格片段是否与open字典中的分区栅格有重叠
                            # 比较当前列中空闲栅格行列号和open字典中存储的空闲栅格的行列号
                            for j in range(len(grid_id)):
                                for jj in range(len(grid_id_open)):
                                    if (grid_id[j] == grid_id_open[jj]).all() and flag == 0:
                                        temp_part = np.array([temp[ii]])
                                        flag = 1
                                        break
                                    elif (grid_id[j] == grid_id_open[jj]).all() and flag != 0:
                                        temp_part = np.append(temp_part, [temp[ii]], axis=0)
                                        break
                                else:
                                    continue
                                break
                    else:
                        break

                # 2.若分区与分段一一对应，那么就将分段向分区一一对应写入
                if len(temp_part) == 1:
                    for k in list(open_dict.keys()):

                        grid_id_open = open_dict[k][:, 0:2]  # 提取每个分区内存储的空闲栅格行列号

                        for ii in range(len(temp)):  # 遍历每个片段
                            new_array = np.array([[temp[ii][0], i, grid_data[temp[ii][0], i]]])
                            for iii in range(temp[ii][0] + 1, temp[ii][1]):  # 将片段的剩余空闲栅格追加到分区内
                                new_array = np.append(new_array, [[iii, i, grid_data[iii, i]]], axis=0)

                            # 存储空闲栅格片段栅格编号
                            grid_id = np.array([[temp[ii][0], i - 1]])  # 将片段的第一个空闲栅格追加到分区内，列号减1是为了与上一列比较
                            for iii in range(temp[ii][0] + 1, temp[ii][1]):  # 将片段的剩余空闲栅格追加到分区内
                                grid_id = np.append(grid_id, [[iii, i - 1]], axis=0)

                            # 比较当前列中空闲栅格行列号和open字典中存储的空闲栅格的行列号
                            index_id = []
                            for j in range(len(grid_id)):
                                for jj in range(len(grid_id_open)):
                                    if (grid_id[j] == grid_id_open[jj]).all():
                                        index_id.append(j)
                            if len(index_id) > 0:
                                open_dict[k] = np.append(open_dict[k], new_array, axis=0)
                # 3. 若出现一个分区对应两个片段的情况，将open字典中的所有分区移动到closed字典中，在open中为每一分段重新创建分区
                elif len(temp_part) > 1:
                    # 将与同一分段出现重叠的多个分区移动到closed字典中
                    for kk in open_dict:
                        closed_dict[kk] = open_dict[kk]
                    # 为每一个分段新建一个分区进行存储
                    temp_dict = {}  # 暂存新的分区
                    for j in range(len(temp_part)):  # 遍历每个片段
                        key = '#' + str(num)  # 字典关键字-分区名称，分区编号从1开始
                        num += 1
                        # 将片段的第一个空闲栅格追加到分区内
                        temp_dict[key] = np.array([[temp_part[j][0], i, grid_data[temp_part[j][0], i]]])
                        for iii in range(temp_part[j][0] + 1, temp_part[j][1]):  # 将片段的剩余空闲栅格追加到分区内
                            temp_dict[key] = np.append(temp_dict[key], [[iii, i, grid_data[iii, i]]], axis=0)
                    open_dict = temp_dict

        part_num = 0  # 用于下一次重新开始对空闲栅格片段进行计数
    for kk in open_dict:
        closed_dict[kk] = open_dict[kk]

    # 将分区结果写入栅格地图和存储成文件
    for k in range(len(closed_dict)):
        key = '#' + str(k + 1)
        r, c = closed_dict[key].shape
        for i in range(r):
            new_map[int(closed_dict[key][i][0])][int(closed_dict[key][i][1])][1] = k + 1
        closed_dict[key] = closed_dict[key].tolist()  # 因为json不认numpy的array，所以需要转化一下
    dict2json.run(closed_dict, 'zone_dict')
    print("初步分区完成...")
    return closed_dict


if __name__ == '__main__':
    # ###############################################测试部分###########################################################
    tic = time.time()
    # 实例0. 小规模综合测试
    a = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0],
         [0, 0.5, 0.5, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0],
         [0, 0, 0.5, 0.5, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0],
         [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0],
         [0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0.5, 0],
         [0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0.5, 0],
         [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0.5, 0],
         [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0],
         [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # 实例1.测试空闲栅格片段等于open字典中的分区数量
    a1 = [[0, 0, 0, 0, 0, 0],
          [0, 0.5, 0.5, 0.5, 0.5, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0.5, 0.5, 0.5, 0.5, 0],
          [0, 0, 0, 0, 0, 0]]
    # 实例2.测试空闲栅格片段大于open字典中的分区数量
    a2 = [[0, 0, 0, 0, 0, 0],
          [0, 0.5, 0.5, 0.5, 0.5, 0],
          [0, 0.5, 0.5, 0.5, 0.5, 0],
          [0, 0.5, 0, 0, 0, 0],
          [0, 0.5, 0.5, 0.5, 0.5, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0.5, 0.5, 0.5, 0.5, 0],
          [0, 0.5, 0.5, 0.5, 0.5, 0],
          [0, 0, 0, 0, 0, 0]]
    # 实例3.测试空闲栅格片段小于open字典中的分区数量
    a3 = [[0, 0, 0, 0, 0],
          [0, 0.5, 0.5, 0.5, 0],
          [0, 0, 0, 0.5, 0],
          [0, 0.5, 0.5, 0.5, 0],
          [0, 0.5, 0.5, 0.5, 0],
          [0, 0, 0, 0, 0]]
    # 实例4.测试Y射线左右两侧有相同数量的障碍物，且同时与射线相切的情况
    a4 = [[0, 0, 0, 0, 0, 0],
          [0, 0.5, 0.5, 0.5, 0.5, 0],
          [0, 0, 0, 0.5, 0.5, 0],
          [0, 0.5, 0.5, 0.5, 0.5, 0],
          [0, 0.5, 0.5, 0, 0, 0],
          [0, 0.5, 0.5, 0.5, 0.5, 0],
          [0, 0, 0, 0, 0, 0]]

    # 实例1-4测试启动代码
    # b = np.array(a4)
    # result = run(b)
    # print(result)

    # 实例5.对任务分区进行分区划分测试
    data = read_csv.r_c('../file_operator/matrix_map_data.csv')
    result = run(data)
    print(result)
    
    # 统计运行时间
    print('Y射线法分区划分总共花费时间为：')
    print(time.time() - tic)

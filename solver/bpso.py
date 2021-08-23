# _*_ coding:utf-8 _*_
# 103中山分队
# 开发人员：runlong
# 开发时间：2019/11/915:42
# 文件名：bpso.py
# 开发工具：PyCharm
# 功能：PSO参考程序，用于每个USV对分区分配最优问题的求解，前提假设为：运行此算法的当前USV依次接收到其他USV的通信信号后，
# 按照与当前USV通信的先后顺序对USV集群个体进行排序，然后再随机选择若干分区与之匹配，计算适应度
# 改进记录：2019.12.13 解决未分配分区数目小于usv数目情况。
# 下一步需要改进的地方：将分区剩余遍历路径长度添加到分区距离矩阵的对角线上

import numpy as np
import random
import matplotlib.pyplot as plt
import math
import time
import itertools
from file_operator import read_csv, read_json
from zone_operator import zone_value, zone_cover_cost


class PSO:
    def __init__(self, usv_zone, occupied_zone, value_list, cost_list, zone_zone_distance, usv_zone_distance, pn=5,
                 max_iter=5, w1=0.5, w2=0.5, w3=0.3, w4=0.3, w5=0.4, w_start=0.9, w_end=0.4, c1=2, c2=2):
        """
        BPSO算法
        """

        self.w5 = w5  # 时间协同差异代价
        self.w4 = w4  # 遍历指定分区代价权重
        self.w3 = w3  # 到达指定分区代价权重
        self.w2 = w2  # 分区空间协同系数，越分散越好
        self.w1 = w1  # 分区价值系数
        self.z_z_distance = np.array(zone_zone_distance)  # 分区形心距离矩阵，矩阵对角线元素用来保存在当前分区内usv剩余的遍历路径长度
        self.u_z_distance = np.array(usv_zone_distance)   # usv与各个分区间距离
        self.value_list = value_list                  # 分区价值列表
        self.cost_list = cost_list                    # 分区遍历代价列表
        self.usv_zone = usv_zone                      # 当前USV及其通信范围内邻接USV编号字典，形式{'#3':[4]},usv编号和所在分区号
        self.pn = pn                                  # 粒子数量
        self.dim = len(self.value_list)               # 搜索维度，分区数量
        self.max_iter = max_iter                      # 迭代次数
        self.w_start = w_start                        # 初始惯性权重
        self.w_end = w_end                            # 终止惯性系数
        self.c1 = c1                                  # 个体认知因子
        self.c2 = c2                                  # 社会认知因子
        self.X = np.zeros((self.pn, self.dim))        # 所有粒子的位置列表
        self.V = np.zeros((self.pn, self.dim))        # 所有粒子的速度列表
        self.p_best = np.zeros((self.pn, self.dim))   # 个体经历的最佳位置
        self.g_best = np.zeros((1, self.dim))         # 种群中全局最佳位置
        self.p_fit = np.zeros(self.pn)                # 每个个体的历史最佳适应值
        self.u_best = ()                              # 存储usv最优排列
        self.fit = -1e10                              # 全局最佳适应值
        self.usv_list = []                            # usv编号列表
        self.occupied_zone = list(occupied_zone)      # usv占据分区编号列表
        self.pre_operation()                          # 对分区和usv编号预处理

    def pre_operation(self):
        """
        统计usv占据分区编号以及其所有的排列形式
        :return:
        """
        usv_list = []
        for key in self.usv_zone:
            usv_list.append(key)  # usv编号的列表
        # 统计USV占据的分区编号
        for key in tuple(usv_list):
            if self.usv_zone[key][0] not in self.occupied_zone:
                self.occupied_zone.append(self.usv_zone[key][0])
        if len(self.value_list) - len(self.occupied_zone) > len(self.usv_zone):  # 检查剩余未分配分区数目足够每个usv分配
            # 对所有usv编号随机排序，用于后续与分配分区进行随机排列组合
            self.usv_list = list(itertools.permutations(usv_list))
        else:
            # 从所有的usv中选出与剩余分区数目相对等的usv进行不放回抽样排列组合
            self.usv_list = list(itertools.combinations(usv_list, len(self.value_list) - len(self.occupied_zone)))
        print(self.usv_list)

    def fitness(self, x, u_tuple):
        """
        目标函数
        :param u_tuple: 以元组表示的一种usv排列形式
        :param x:分区分配方案
        :return:
        """
        j_1 = np.dot(x, self.value_list)  # 分区价值收益
        j_2 = 0
        c_1 = 1e-10                       # 到达指定分区的路程，包括当前分区剩余遍历长度和分区间巡航长度，初始值设置为非零
        c_2 = np.dot(x, self.cost_list)   # 分区遍历代价
        ind = [i for i, n in enumerate(x) if n == 1]  # 统计x中元素为1的索引值
        # 计算分区代价cost必备的三个信息：
        # 1.当前USV与其通信范围内的邻接USV所在分区编号，经过通信后获得，每次迭代需要随机打乱顺序；
        # 2.目标分区编号(x中为1的元素索引值+1，x中每个元素的索引代表一个分区，其顺序为#1、#2、#3……)；
        # 3.分区距离矩阵self.distance_mat中对应分区编号的元素。
        t_p = []   # 记录巡航时间
        t_c = []   # 记录遍历时间
        t_std = 0  # 时间方差
        for i in range(len(u_tuple)):
            zone_start = u_tuple[i][2]   # 注意：usv的编号形式为'##2'，有两个#
            zone_goal = ind[i] + 1                                              # 加1的原因是：分区编号是从1开始，而x中的索引是从0开始
            c_1 += self.usv_zone[u_tuple[i]][1] / self.usv_zone[u_tuple[i]][3]  # 将usv所在分区的剩余遍历路径长度添加进到达下一分区的长度中
            c_1 += self.u_z_distance[int(zone_start), int(zone_goal)]           # 各个usv的总巡航路程代价
            t_p.append(self.u_z_distance[int(zone_start), int(zone_goal)] / self.usv_zone[u_tuple[i]][2])   # 巡航耗时
            t_c.append(self.usv_zone[u_tuple[i]][1] / self.usv_zone[u_tuple[i]][3])                         # 剩余遍历路径耗时

        for i in tuple(ind):
            for j in tuple(ind):
                j_2 += self.z_z_distance[i, j]  # 选定分区间的距离，目的是使选定的分区充分分散
        j_2 = j_2 / 2                           # 距离在上两行代码中叠加了两次

        # 计算时间协同收益
        # 1.计算平均时间，需要的参数有各个usv巡航速度和遍历速度，到达分配分区的巡航路程以及所处当前分区内剩余的遍历路径长度
        t_total = sum(t_p) + sum(t_c)
        t_mean = t_total/len(self.usv_zone)
        # 2.计算各个usv时间的标准差作为时间协同差异代价，表达的含义为使各个usv保持在时间上一致
        for i in range(len(u_tuple)):
            t_std += (t_p[i] + t_c[i] - t_mean) ** 2
        # print("stander is %d" % t_std)
        if t_std == 0:  # 防止t_std出现0值
            t_std = 1e-10
        c_3 = t_std ** 0.5
        return (self.w1 * j_1 + self.w2 * j_2) / (self.w3 * c_1 + self.w4 * c_2 + self.w5 * c_3)

    def init_population(self):
        """
        初始化种群
        :return:
        """
        ele_flag = list(range(1, self.dim + 1))
        for k in tuple(self.occupied_zone):  # 目标分区列表中需要去掉已被遍历的分区，需要注意多个usv在同一分区的情况，故采用异常处理
            try:
                ele_flag.remove(int(k))
            except:
                print("已经删除元素%d." % k)
        for i in range(self.pn):
            # 随机选出与usv_list列表中数目相同的分区分配给对等数目的usv
            elements = random.sample(ele_flag, len(self.usv_list[0]))
            for j in tuple(elements):
                self.X[i][j - 1] = 1
                self.V[i][j - 1] = random.uniform(0, 1)
            self.p_best[i] = self.X[i]
            tmp = self.fitness(self.X[i], self.usv_list[0])
            self.p_fit[i] = tmp
            if tmp > self.fit:
                self.fit = tmp
                self.g_best = self.X[i]

    def iterator(self):
        """
        更新粒子位置和速度
        :return:适应度值和决策结果
        """
        plan = {}    # 存储usv新分配的分区编号
        g_best_list = np.zeros((len(self.usv_list), len(self.g_best)))  # 存储每种usv排列得到的最优分配方案
        fitness_list = np.zeros(len(self.usv_list))  # 存储每种排列组合的最优适应度值

        for num in range(len(self.usv_list)):
            for t in range(self.max_iter):
                w = self.w_start - (self.w_start - self.w_end) * (self.max_iter - t) / self.max_iter
                # 个体和全局最优位置更新
                for i in range(self.pn):
                    temp = self.fitness(self.X[i], self.usv_list[num])
                    if temp > self.p_fit[i]:          # 更新个体最优
                        self.p_fit[i] = temp
                        self.p_best[i] = self.X[i]
                        if self.p_fit[i] > self.fit:  # 更新全局最优
                            self.g_best = self.X[i]
                            self.fit = self.p_fit[i]
                # 位置和速度更新,更新次数为usv_num，即X[i]中元素1的个数
                for i in range(self.pn):
                    self.V[i] = w * self.V[i] + self.c1 * random.random() * (self.p_best[i] - self.X[i]) + \
                                self.c2 * random.random() * (self.g_best - self.X[i])
                    x = []
                    for m in range(len(self.X[i])):
                        x.append(str(int(self.X[i][m])))
                    if random.random() < 1 / (1 + math.exp(-sum(self.V[i]))):  # X中元素1左移
                        num_int = int(''.join(x), 2) + 1                       # 将X[i]列表拼接成二进制，再转化为十进制，再加1
                        while num_int < 2 ** len(self.value_list):
                            str_bin = ''.join([str((num_int >> y) & 1)
                                               for y in range(len(self.value_list) - 1, -1, -1)])  # 转回二进制
                            if str_bin.count('1') == len(self.usv_list[num]):
                                # 统计元素1的位置，用于排除usv当前时刻占据的分区
                                list_bin = list(str_bin)         # 将二进制字符串转化为字符列表
                                list_int = []
                                for ii in range(len(list_bin)):  # 将字符列表转化为数字列表
                                    list_int.append(int(list_bin[ii]))
                                ind = [i + 1 for i, n in enumerate(list_int) if n == 1]  # 统计元素为1的索引值
                                flag_occupied = False
                                for ii in tuple(ind):
                                    if ii in self.occupied_zone:   # 判断self.X[i]中元素1分区编号是否与usv占据分区编号重叠
                                        flag_occupied = True
                                        break
                                if not flag_occupied:
                                    for n in range(len(str_bin)):
                                        self.X[i][n] = list_int[n]
                                    break
                                else:
                                    num_int += 1
                            else:
                                num_int += 1
                    else:                                 # X[i]中元素1右移
                        num_int = int(''.join(x), 2) - 1  # 将X[i]列表拼接成二进制，再转化为十进制，再减1
                        while num_int > 2 ** (len(self.usv_list[num]) + 1) - 1:
                            str_bin = ''.join(
                                [str((num_int >> y) & 1) for y in range(len(self.value_list) - 1, -1, -1)])  # 转回二进制
                            if str_bin.count('1') == len(self.usv_list[num]):
                                # 统计元素1的索引，用于排除usv当前时刻占据的分区
                                list_bin = list(str_bin)         # 将二进制字符串转化为字符列表
                                list_int = []
                                for ii in range(len(list_bin)):  # 将字符列表转化为数字列表
                                    list_int.append(int(list_bin[ii]))
                                ind = [i + 1 for i, n in enumerate(list_int) if n == 1]  # 统计元素为1的索引值
                                flag_occupied = False
                                for ii in tuple(ind):
                                    if ii in self.occupied_zone:  # 判断self.X[i]中元素1分区编号是否与usv占据分区编号重叠
                                        flag_occupied = True
                                        break
                                if not flag_occupied:
                                    for n in range(len(str_bin)):
                                        self.X[i][n] = list_int[n]
                                    break
                                else:
                                    num_int -= 1
                            else:
                                num_int -= 1
                fitness_value = self.fit
                print(self.g_best)
                print(self.fit)
                # print(self.fit)  # 输出最优值
                # print(self.X)
            g_best_list[num] = self.g_best
            fitness_list[num] = self.fit
            print(self.usv_list[num])
        fitness_list = fitness_list.tolist()
        max_fitness_num = fitness_list.index(max(fitness_list))
        id1 = [i for i, n in enumerate(g_best_list[max_fitness_num]) if n == 1]
        for k in range(len(self.usv_list[max_fitness_num])):
            plan[self.usv_list[max_fitness_num][k]] = self.usv_zone[self.usv_list[max_fitness_num][k]][0:1]
            plan[self.usv_list[max_fitness_num][k]].append(id1[k] + 1)  # 将目标分区添加到字典中，目标分区选择范围从1开始
        plan["f"] = np.round(fitness_value, 4)  # 将适应度值添加到字典中
        return plan
        # 返回结果中new_usv_zone举例：
        # {'#1':[1 3], '#3':[5 11], 'f': 33}表示含义为#1USV需要从1分区到达3分区，#3USV需要从5分区到达11分区，f为适应度


if __name__ == '__main__':
    tic = time.time()
    # #############################################################
    # 算例
    # 读取分区距离矩阵
    z2z_d = read_csv.run("../distance/z2z_distance_v2.csv")            # 分区形心距离矩阵
    u2z_d = read_csv.run('../distance/u2z_distance.csv')               # usv与各个分区的距离
    zone_data = read_json.run("../zone_operator/split_dict.json")      # 读取分区信息
    v_list = zone_value.run(zone_data)                                 # 分区价值列表
    c_list = zone_cover_cost.run(zone_data)                            # 分区遍历代价列表，即每个分区内的栅格数目列表

    # 键代表usv编号，值列表中各个元素表示各usv所处的分区编号、剩余遍历栅格数量、巡航速度、遍历速度、所在栅格行号、列号
    # （巡航速度定义为单位1，表示栅格尺寸为单位时间usv巡航速度走过的距离）
    # u_zone = {'#1': [1, 0], '#2': [3, 0]}
    u_zone = {'##1': [1, 0, 1, 0.25], '##2': [3, 0, 1, 0.25], '##3': [10, 0, 1, 0.25], '##4': [3, 0, 1, 0.25],
              '##5': [15, 0, 0.5, 0.25]}     # usv状态信息
    o_zone = [2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16]                        # 已经被遍历过的分区

    # 加载usv与分区距离
    u1_d = read_csv.run('../distance/u2z_usv1_distance.csv')
    u2_d = read_csv.run('../distance/u2z_usv2_distance.csv')
    u3_d = read_csv.run('../distance/u2z_usv3_distance.csv')
    u4_d = read_csv.run('../distance/u2z_usv4_distance.csv')
    u5_d = read_csv.run('../distance/u2z_usv5_distance.csv')

    r, c = np.array(u1_d).shape
    u2z_d = np.zeros((1, c))
    u2z_d = np.append(u2z_d, u1_d, axis=0)
    u2z_d = np.append(u2z_d, u2_d, axis=0)
    u2z_d = np.append(u2z_d, u3_d, axis=0)
    u2z_d = np.append(u2z_d, u4_d, axis=0)
    u2z_d = np.append(u2z_d, u5_d, axis=0)

    # #####################################执行BPSO算法#######################################
    # 参数列表：usv所在分区，已分配分区列表，价值列表，分区剩余遍历路径长度，距离矩阵，种群数目，最大迭代次数，价值权重，分布权重，
    # 到达指定分区代价权重，遍历指定分区代价权重，协同时间差异代价权重
    print('输入量：')
    print('u_zone:%s' % u_zone)
    print('o_zone:%s' % o_zone)
    print('v_list:%s' % v_list)
    print('c_list:%s' % c_list)
    print('z2z_d:%s' % z2z_d)
    print('u2z_d:%s' % u2z_d)
    zone_pso = PSO(u_zone, o_zone, v_list, c_list, z2z_d, u2z_d, 5, 10, 0.5, 0.5, 0.3, 0.3, 0.4)
    zone_pso.init_population()
    usv_plan = zone_pso.iterator()
    print('输出量：')
    print(usv_plan)
    print(time.time() - tic)
    # 输出结果说明：
    # u_zone = {'#1': [1, 0, 1, 0.25, 4], '#2': [3, 0, 1, 0.25, 5], '#3': [10, 0, 1, 0.25, 13],
    # '#4': [3, 0, 1, 0.25, 18], '#5': [15, 0, 1, 0.25, 16]}
    # 列表中各元素代表含义：usv所在分区编号，指定分区内遍历剩余遍历路径，巡航速度，遍历速度，下一次目标分区

    # 绘制适应度变化曲线
    # plt.figure(1)
    # plt.title("Figure1")
    # plt.xlabel("iterators", size=14)
    # plt.ylabel("fitness", size=14)
    # t = np.array([t for t in range(0, 120)])
    # fitness = usv_plan['f']
    # plt.plot(t, fitness, color='b', linewidth=3)
    # plt.show()

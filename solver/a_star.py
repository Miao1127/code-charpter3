# _*_ coding:utf-8 _*_
# 103中山分队
# 开发人员：runlong
# 开发时间：2019/10/714:42
# 文件名：A_map.py
# 开发工具：PyCharm
# 功能：测试A Star算法，输入栅格地图、起点和终点，输出路径栅格编号

# 所有节点的g值并没有初始化为无穷大
# 当两个子节点的f值一样时，程序选择最先搜索到的一个作为父节点加入closed
# 对相同数值的不同对待，导致不同版本的A*算法找到等长的不同路径
# 最后closed表中的节点很多，如何找出最优的一条路径
# 撞墙之后产生较多的节点会加入closed表，此时开始删除closed表中不合理的节点，1.1版本的思路
# 1.2版本思路，建立每一个节点的方向指针，指向f值最小的上个节点
# 参考《无人驾驶概论》、《基于A*算法的移动机器人路径规划》王淼驰，《人工智能及应用》鲁斌

import time
import numpy as np
from distance import path_distance
from file_operator import read_csv


class AStar(object):
    """
    A*算法类，输入np.array形式的起点和终点
    """

    def __init__(self, map, start, goal, max_ite):
        """
        初始化
        """
        # self.g = 0  # g初始化为0
        self.max_ite = max_ite  # 最大迭代次数
        self.map = map          # 栅格地图
        self.start = start      # 起点坐标
        self.goal = goal        # 终点坐标
        self.open = np.array([[], [], [], [], [], []])     # 先创建一个空的open表, 记录坐标，方向，g值，f值
        self.closed = np.array([[], [], [], [], [], []])   # 先创建一个空的closed表
        self.best_path_array = np.array([[], []])          # 回溯路径表

    def h_value_tem(self, son_p):
        """
        计算拓展节点和终点的h值
        :param son_p:子搜索节点坐标
        :return:
        """
        h = (son_p[0] - self.goal[0]) ** 2 + (son_p[1] - self.goal[1]) ** 2
        h = np.sqrt(h)  # 计算h
        return h

    # def g_value_tem(self, son_p, father_p):
    #     """
    #     计算拓展节点和父节点的g值
    #     其实也可以直接用1或者1.414代替
    #     :param son_p:子节点坐标
    #     :param father_p:父节点坐标，也就是self.current_point
    #     :return:返回子节点到父节点的g值，但不是全局g值
    #     """
    #     g1 = father_p[0] - son_p[0]
    #     g2 = father_p[1] - son_p[1]
    #     g = g1 ** 2 + g2 ** 2
    #     g = numpy.sqrt(g)
    #     return g

    def g_accumulation(self, son_point, father_point):
        """
        累计的g值
        :return:
        """
        g1 = father_point[0] - son_point[0]
        g2 = father_point[1] - son_point[1]
        g = g1 ** 2 + g2 ** 2
        g = np.sqrt(g) + father_point[4]  # 加上累计的g值
        return g

    def f_value_tem(self, son_p, father_p):
        """
        求出的是临时g值和h值加上累计g值得到全局f值
        :param father_p: 父节点坐标
        :param son_p: 子节点坐标
        :return:f
        """
        f = self.g_accumulation(son_p, father_p) + self.h_value_tem(son_p)
        return f

    def child_point(self, x):
        """
        拓展的子节点坐标
        :param x: 父节点坐标
        :return: 子节点存入open表，返回值是每一次拓展出的子节点数目，用于撞墙判断
        当搜索的节点撞墙后，如果不加处理，会陷入死循环
        """
        # 取得栅格地图的行列数
        rows_max, cols_max = np.array(self.map).shape
        # 获取障碍物栅格编号

        # 开始遍历周围8个节点
        for j in range(-1, 2, 1):
            for q in range(-1, 2, 1):

                if j == 0 and q == 0:  # 搜索到父节点去掉
                    continue
                m = [x[0] + j, x[1] + q]
                # print(m)
                # 搜索点出了边界去掉，编号从0开始，需要将行列数减1
                if m[0] < 0 or m[0] > rows_max - 1 or m[1] < 0 or m[1] > cols_max - 1:
                    continue

                if self.map[int(m[0]), int(m[1])] == 0:  # 搜索到障碍物去掉
                    continue

                record_g = self.g_accumulation(m, x)
                record_f = self.f_value_tem(m, x)  # 计算每一个节点的f值

                x_direction, y_direction = self.direction(x, m)  # 每产生一个子节点，记录一次方向

                para = [m[0], m[1], x_direction, y_direction, record_g, record_f]  # 将参数汇总一下
                # print(para)

                # 在open表中，则去掉搜索点，但是需要更新方向指针和self.g值
                # 而且只需要计算并更新self.g即可，此时建立一个比较g值的函数
                a, index = self.judge_location(m, self.open)
                if a == 1:
                    # 说明open中已经存在这个点

                    if record_f <= self.open[5][index]:
                        self.open[5][index] = record_f
                        self.open[4][index] = record_g
                        self.open[3][index] = y_direction
                        self.open[2][index] = x_direction

                    continue

                # 在closed表中,则去掉搜索点
                b, index2 = self.judge_location(m, self.closed)
                if b == 1:

                    if record_f <= self.closed[5][index2]:
                        self.closed[5][index2] = record_f
                        self.closed[4][index2] = record_g
                        self.closed[3][index2] = y_direction
                        self.closed[2][index2] = x_direction
                        self.closed = np.delete(self.closed, index2, axis=1)
                        self.open = np.c_[self.open, para]
                    continue

                self.open = np.c_[self.open, para]  # 参数添加到open中
                # print(self.open)

    def judge_location(self, m, list_co):
        """
        判断拓展点是否在open表或者closed表中
        :return:返回判断是否存在，和如果存在，那么存在的位置索引
        """
        jud = 0
        index = 0
        for i in range(list_co.shape[1]):

            if m[0] == list_co[0, i] and m[1] == list_co[1, i]:

                jud = jud + 1

                index = i
                break
            else:
                jud = jud
        # if a != 0:
        #     continue
        return jud, index

    def direction(self, father_point, son_point):
        """
        建立每一个节点的方向，便于在closed表中选出最佳路径
        非常重要的一步，不然画出的图像参考1.1版本
        x记录子节点和父节点的x轴变化
        y记录子节点和父节点的y轴变化
        如（0，1）表示子节点在父节点的方向上变化0和1
        :return:
        """
        x = son_point[0] - father_point[0]
        y = son_point[1] - father_point[1]
        return x, y

    def path_backtrace(self):
        """
        回溯closed表中的最短路径
        :return:
        """
        best_path = self.goal  # 回溯路径的初始化
        self.best_path_array = np.array([[self.goal[0]], [self.goal[1]]])
        j = 0
        a = self.closed.shape[1]
        for j in range(self.closed.shape[1]):
            for i in range(self.closed.shape[1]):
                if best_path[0] == self.closed[0][i] and best_path[1] == self.closed[1][i]:
                    x = self.closed[0][i] - self.closed[2][i]
                    y = self.closed[1][i] - self.closed[3][i]
                    best_path = [x, y]
                    self.best_path_array = np.c_[self.best_path_array, best_path]
                    break  # 如果已经找到，退出本轮循环，减少耗时
                else:
                    continue
            if best_path[0] == self.start[0] and best_path[1] == self.start[1]:
                break
            else:
                continue
        self.best_path_array[0] = np.flipud(self.best_path_array[0])  # 将路径顺序转换为从起点到终点
        self.best_path_array[1] = np.flipud(self.best_path_array[1])
        return self.best_path_array

    def run(self):
        """
        main函数
        :return:
        """
        best = self.start                            # 起点放入当前点，作为父节点
        h0 = self.h_value_tem(best)
        init_open = [best[0], best[1], 0, 0, 0, h0]  # 将方向初始化为（0，0），g_init=0,f值初始化h0
        self.open = np.column_stack((self.open, init_open))  # 起点放入open,open初始化

        ite = 1  # 设置迭代次数小于200，防止程序出错无限循环
        while ite < self.max_ite:
            # open列表为空，退出
            if self.open.shape[1] == 0:
                print('没有搜索到路径！')
                return

            self.open = self.open.T[np.lexsort(self.open)].T  # open表中最后一行排序(联合排序）

            # 选取open表中最小f值的节点作为best，放入closed表
            best = self.open[:, 0]
            # print('检验第%s次当前点坐标*******************' % ite)
            # print(best)
            self.closed = np.c_[self.closed, best]

            if best[0] == self.goal[0] and best[1] == self.goal[1]:  # 如果best是目标点，退出
                print('搜索成功！')
                return

            self.child_point(best)  # 生成子节点并判断数目
            # print(self.open)
            self.open = np.delete(self.open, 0, axis=1)  # 删除open中最优点

            # print(self.open)

            ite = ite + 1


if __name__ == '__main__':
    tic = time.time()
    # 算例1
    # 定义一个含有障碍物的20×20的栅格地图
    # 10表示可通行点
    # 0表示障碍物
    # 2表示起点,数字只是一个标记，可以设置为任意其他数字
    # 3表示终点，可以设置为任意其他数字

    # map_grid = np.full((20, 20), 1, dtype=np.int8)
    # map_grid[3, 3:8] = 0
    # map_grid[3:10, 7] = 0
    # map_grid[10, 3:8] = 0
    # map_grid[17, 13:17] = 0
    # map_grid[10:17, 13] = 0
    # map_grid[10, 13:17] = 0
    # map_grid[5, 2] = 2
    # map_grid[15, 19] = 3
    # print(map_grid)
    #
    # a1 = AStar(map_grid, np.array([5, 3]), np.array([15, 19]), 100000)           # 实例化一个对象
    # a1.run()                                                                     # 调用对象中的主函数方法
    # a1.path_backtrace()

    # 算例2
    # 读入栅格数据

    data = np.array(read_csv.run('../file_operator/matrix_map_data_add_start.csv'))
    a2 = AStar(data, [11, 5], [49, 1], 100000)                              # 实例化一个对象
    a2.run()                                                                # 调用对象中的主函数方法
    path = a2.path_backtrace()
    print(path)
    p_distance = path_distance.run(path[0], path[1])
    print(p_distance)

    toc = time.time()
    print(toc - tic)

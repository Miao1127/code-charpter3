# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：2019/12/1914:47
# 文件名：usv_simulator_v2.py
# 开发工具：PyCharm
# 功能：usv仿真器

import time
import threading
import serial.tools.list_ports
import numpy as np
from queue import Queue
import copy
from file_operator import read_json, read_csv
from zone_operator import zone_cover_cost, zone_value, zone_cover
from solver import bpso, a_star
from distance import point2zone


def read_msg(q2,):
    """
    接收信息
    :return:
    """
    print('等待接收指令！')
    while True:
        if ser.in_waiting:
            data_received = ser.readline().decode("gbk")
            print(str(data_received))
            data = data_received.split('$')
            print(data)
            q2.put(data)


def send_msg(q1,):
    """
    发送信息
    :return:
    """
    while True:
        ser.write(str.encode('\n'))
        string_to_send = str(q1.get())
        ser.write(str.encode(string_to_send))
        print("sending message(%s)..." % string_to_send)
        time.sleep(T_send)
        print(time.time())


class InitUSV:
    def __init__(self, usv_info):
        self.usv_info = usv_info
        self._running = True

    def terminate(self):
        self._running = False

    def run(self):
        while self._running:
            string_to_send = 'I' + '$' + str(self.usv_info) + '$' + '\n'
            ser.write(str.encode(string_to_send))
            time.sleep(T_send)


# usv集群行为，主要包括：BPSO分区协同分配、最终决策结果选择、奔向目标分区、遍历分区
def solver(q1, q2):
    global z2z_mat, map_data, zone_data, vertex_dict, usv_info, usv_list, usv_dict
    global usv_num, start_point, cruise_speed, cover_speed, path
    usv_start_info = copy.deepcopy(usv_info)  # 巡航路径规划起始点信息
    v_list = zone_value.run(zone_data)                                         # 分区价值列表
    c_list = zone_cover_cost.run(zone_data)                                    # 分区遍历代价列表，即每个分区内的栅格数目列表
    occupied_zone = []  # 存储已经被遍历过的分区，每次进行完BPSO分区协同任务分配后，将取得一致的最终决策分区代号写入此列表
    while True:
        # TODO 改进方向，设置事件触发机制，是否要开启三个线程，一个负责不断传递状态信息，一个负责计算完距离后进行BPSO，一个负责接收信息。
        # 1.各个usv根据自身所在位置计算距离每个分区的最短距离，并生成到达各个分区的巡航路径选项
        u2z_d, usv_path = point2zone.run(usv_start_info, occupied_zone, vertex_dict, map_data)  # 计算usv从当前位置到各个分区的距离

        # 2.向其他usv发送计算出的距离，直至集齐其他usv发送的距离信息为止
        u2z_d = [round(i, 1) for i in u2z_d]                         # 限制距离的小数位数，避免超出串口单次发送的最大数据量
        d_send_msg = 'D' + '$' + str(u2z_d) + '$' + '\n'
        u2z_mat = np.zeros((len(usv_list) + 1, len(zone_data) + 1))  # 初始化usv到分区的距离矩阵

        while (u2z_mat[1:, 0] == 0).any():
            q1.put(d_send_msg)
            time.sleep(0.5)
            r_msg = q2.get()
            if r_msg[0] == 'D':
                d_receive_msg = eval(r_msg[1])
                usv_key = list(d_receive_msg.keys()[0])
                usv_row = int(usv_key[2])
                u2z_mat[usv_row] = d_receive_msg[usv_key]

        # 3.BPSO分区协同分配，并将结果发送出去
        # TODO 发送自身决策结果的同时也需要接受其他usv的决策结果
        # usv完成分区遍历任务，一旦收集完成开始决策协商。
        # 参数列表：各个usv所在分区(通过串口通信收集)，已分配分区列表(每个usv建立一个列表进行存储)，分区价值列表(zone_value.py)，
        # 分区剩余遍历路径长度，距离矩阵(加载../distance/zone2zone_distance_mat.csv)
        # 默认参数(若修改，需要进入/solver/bpso.py中设定）：种群数目，最大迭代次数，价值权重，分布权重，到达指定分区代价权重，
        # 遍历指定分区代价权重，协同时间差异代价权重
        zone_pso = bpso.PSO(usv_dict, occupied_zone, v_list, c_list, z2z_mat, u2z_mat)
        zone_pso.init_population()
        b_result = zone_pso.iterator()  # BPSO决策结果，例如：{'##3': [10, 17], '##4': [3, 18], 'f': 0.7377635825268328}
        usv_plan = b_result
        usv_plan['u'] = [usv_num]

        # 3.决策方案协商，发送自身决策结果，收集其他usv决策结果并进行对比，找到f值最大者，将当前usv编号添加到响应的最后方案中
        # 决策发送协议：
        # {'##4': [3, 4], '##1': [1, 5], '##5': [15, 7], '##3': [10, 14], '##2': [3, 18],
        # 'f': 0.973871766408908, 'u':['##1']}
        # 协议说明： 键'#4'表示usv编号，值[3, 4]表示决策结果，#4 usv从所在的3分区到4分区，f表示整个决策的效能指数
        # 'u'对应的值表示usv认可的最优决策，即若此方案最优，当前usv将会把自身编号追加到此列表中，并发送出去
        while len(usv_plan['u']) != len(usv_list):  # TODO 需要加一个时间限制，或者某usv已经完成指定分区的遍历
            r_msg = q2.get()
            if r_msg[0] == 'B':
                other_plan = eval(r_msg[1])
                if other_plan['f'] > usv_plan['f'] and usv_num not in other_plan['u']:
                    other_plan['u'].append(usv_num)
                    usv_plan = other_plan
                elif other_plan['f'] == usv_plan['f'] and usv_num not in other_plan['u']:
                    other_plan['u'].append(usv_num)
                    usv_plan = other_plan
            s_plan = 'P' + '$' + str(usv_plan) + '$' + '\n'
            q1.put(s_plan)

        for key in usv_plan:
            o_zone = '#' + str(usv_plan[key][1])
            occupied_zone = np.append(occupied_zone, [o_zone], axis=0)  # 将达成一致的分配方案中分区的编号加入到已被分配的分区列表中

        # 4.执行决策结果，若usv在初始出发位置，则需要从出发位置到决策指定分区进行路径规划，若usv在分区内，则需要确定分区遍历路径终点，以
        #   此终点为到达下一个分区路径的起点做路径规划，规划完成后usv奔向各自的指定分区，同时执行以指定分区遍历终点为各自usv起点的，到达
        #   剩余未分配分区的距离计算
        # 4.1 解析最终决策方案，将目标分区更新到usv_info，并将目标分区和下次巡航路径规划的起点位置存储到usv_start_info中
        #     然后从第1步的巡航路径选项中选择对应巡航路径，传递给执行器
        target_zone = '#' + str(s_plan[usv_num][1])
        usv_info[usv_num][0] = target_zone
        plan_path = usv_path[target_zone]
        speed = np.ones(len(plan_path)) * cruise_speed  # 在路径的矩阵中追加一列巡航速度项
        plan_path = np.c_[plan_path, speed]
        if (plan_path[0][0:2] == path[-1][0:2]).all():
            path = np.append(path, plan_path, axis=0)

        # 4.2 对本次分配的分区进行遍历路径规划，传递给执行器，记录遍历路径终点
        cover_start = plan_path[-1]
        cover_grid_list = zone_data[target_zone]
        cover_path = zone_cover.run(cover_start, cover_grid_list)
        speed = np.ones(len(cover_path)) * cover_speed
        cover_path = np.c_[cover_path, speed]
        if (cover_path[0][0:2] == path[-1][0:2]).all():
            path = np.append(path, cover_path, axis=0)
            usv_start_info[usv_num][0] = target_zone  # 将目标分区的编号更新到usv_start_info中
            usv_start_info[usv_num][4] = path[-1][0]  # 更新下次巡航遍历的起点行编号
            usv_start_info[usv_num][4] = path[-1][1]  # 更新下次巡航遍历的起点列编号
        # 4.3 将本次分配的分区编号添加到占据分区内，并判断是否还存在未分配的分区，若无，则以遍历路径终点为起点，出发点为终点，规划返航路径
        if len(occupied_zone) == len(v_list):  # 不存在还未分配的分区
            recovery = a_star.AStar(map_data, path[-1], start_point)
            recovery.run()
            recovery_path = recovery.path_backtrace()
            recovery_speed = np.ones(len(recovery_path)) * cruise_speed
            recovery_path = np.c_[recovery_path, recovery_speed]
            path = np.append(path, recovery_path, axis=0)
            break


def action(q1,):  # 参数（起始位置、角度和目标位置、角度）
    """
    刷新显示各个usv的位置和艏向
    :param q1:进程间通信
    :param v:速度，两种选择：巡航速度和遍历速度
    :param x_s:起始横坐标
    :param y_s:起始纵坐标
    :param x_g:目标横坐标
    :param y_g:目标纵坐标
    :return:
    """
    global path, usv_info
    while True:
        if len(path) > 3:
            x = path[0][0]
            y = path[0][1]
            usv_info[usv_num][4] = x
            usv_info[usv_num][5] = y
            state_data = 'S' + '$' + str(usv_info) + '$' + '\n'
            q1.put(state_data)
            dt = 1 / path[0][2]
            time.sleep(dt)
            del path[0]


def multi_threading():
    global usv_info, usv_list, usv_dict
    # 需要在程序运行过程中停止的线程参考下方t线程编写
    ini_usv = InitUSV(usv_info)
    t = threading.Thread(target=ini_usv.run)

    # 在程序运行过程中不需要停止的线程置于下方
    t1 = threading.Thread(target=read_msg, args=(q2,))
    t2 = threading.Thread(target=send_msg, args=(q1,))
    t3 = threading.Thread(target=solver, args=(q1, q2))
    t4 = threading.Thread(target=action, args=(q1,))

    t.start()   # 开启初始化发送信息线程
    t1.start()  # 开启串口接收信息线程
    while True:  # 响应监控端发出的初始化指令
        data_serial = q2.get()
        # 收集集群成员信息
        if len(data_serial) == 3 and data_serial[0] == 'I':
            i_msg = eval(data_serial[1])  # 接收到数据格式为：{"##2": ["#0", 0, 1, 0.25, 11, 5]}
            usv_name = list(i_msg)[0]
            if usv_name not in usv_list:
                usv_list.append(usv_name)
                usv_dict.update(i_msg)   # 存储各个usv初始位置信息，用于计算各个usv到达各个分区距离
        if len(data_serial) == 3 and data_serial[0] == 'C':
            c_msg = eval(data_serial[1])     # 接收到数据为usv编号列表
            if len(c_msg) == len(usv_list):  # 判断是否收到监控端开启集群指令和集群成员是否都集齐
                ini_usv.terminate()          # 停止usv向监控端发送usv编号和初始位置线程
                t2.start()                   # 开启串口发送线程，定时发送usv自身的状态信息
                q1.put('S' + '$' + str(usv_info) + '$' + '\n')  # 回应监控端的首条信息
                time.sleep(0.1)
                t3.start()                   # 开启求解器线程
                t4.start()                   # 开启执行器线程
                print("连接监控端...")
                break


if __name__ == '__main__':
    # 线程间信息传递
    q1 = Queue()  # 用于放置当前usv的状态信息，此信息需要发送给其他usv，实现集群协同
    q2 = Queue()  # 用于接收其他usv发送的状态信息，实现集群协同

    # 加载集群信息
    # BPSO算法参数列表：usv所在分区，已分配分区列表，价值列表，分区剩余遍历路径长度，距离矩阵，种群数目，最大迭代次数，价值权重，
    # 分布权重，到达指定分区代价权重，遍历指定分区代价权重，协同时间差异代价权重
    z2z_mat = read_csv.run("../distance/z2z_distance_v2.csv")                  # 分区形心距离矩阵，用于在BPSO中衡量分区分散程度
    map_data = read_csv.run('../file_operator/matrix_map_data_add_start.csv')  # 读取地图信息
    zone_data = read_json.run("../zone_operator/split_dict.json")              # 读取分区信息
    vertex_dict = read_json.run('../zone_operator/vertex_dict.json')           # 读取分区顶点信息

    # 在计算分区价值时，需要排除已被遍历过的分区
    usv_num = '##2'                                  # 当前usv编号
    start_point = [13, 9]                            # 当前usv出发点
    cruise_speed = 1                                 # 巡航速度
    cover_speed = 0.25                               # 遍历速度
    usv_info = {usv_num: ["#0", 0, cruise_speed, cover_speed, start_point[0], start_point[1]]}    # 当前usv信息
    usv_list = [usv_num]                             # usv列表初始化，首先存储当前usv编号，后续用于存储收集到的其他usv编号
    usv_dict = copy.deepcopy(usv_info)               # usv字典初始化，首先存储当前usv信息，后续用于存储收集到的其他usv的状态信息
    path = np.array([[start_point[0], start_point[1], cruise_speed]])  # 全局变量，用于存储当前usv的航行的路径，初始化存储出发点

    # 串口通信参数
    serial_port = 'com4'
    serial_baud = 115200
    time_out = 10
    T_send = 0.5

    # 开启串口通信端口
    ser = serial.Serial(serial_port, baudrate=serial_baud, timeout=time_out)
    time.sleep(0.5)

    # 开启三个线程进行收发数据和建立usv运动模型
    multi_threading()

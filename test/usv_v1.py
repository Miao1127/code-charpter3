# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：2019/12/418:26
# 文件名：usv_2.py
# 开发工具：PyCharm
# 功能：模拟#2 usv，具备串口收发能力和运动能力

import time
import threading
import serial.tools.list_ports
import numpy as np
from queue import Queue


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
            data = data_received.split('#')
            print(data)
            q2.put(data)


def send_msg(q1,):
    """
    发送信息
    :return:
    """
    while True:
        ser.write(str.encode('\n'))
        string_to_send = q1.get()
        ser.write(str.encode(string_to_send))
        print("sending message(%s)..." % string_to_send)
        time.sleep(T_send)
        print(time.time())


class InitUSV:
    def __init__(self):
        self._running = True

    def terminate(self):
        self._running = False

    def run(self):
        while self._running:
            ser.write(str.encode('\n'))
            string_to_send = '#I' + usv_Num + '#' + str(x_start) + '#' + str(y_start) + '#'
            ser.write(str.encode(string_to_send))
            time.sleep(T_send)


# 移动位置，从一个位置移动到下一个位置
def move_to_pose(q1, x_s, y_s, x_g, y_g, v):  # 参数（起始位置、角度和目标位置、角度）
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

    x = x_s
    y = y_s

    x_diff = x_g - x
    y_diff = y_g - y

    rho = np.sqrt(x_diff**2 + y_diff**2)  # 计算距离
    while rho > 0.001:
        x_diff = x_g - x
        y_diff = y_g - y

        rho = np.sqrt(x_diff**2 + y_diff**2)                               # 计算当前位置与期望位置间的距离
        # 运动模型
        x = x + v * dt
        y = y + v * dt
        state_data = '#S' + usv_Num + '#' + str(x) + '#' + str(y) + '#'
        q1.put(state_data)
        time.sleep(dt)


def multi_threading():
    # 需要在程序运行过程中停止的线程参考下方t线程编写
    ini_usv = InitUSV()
    t = threading.Thread(target=ini_usv.run)

    # 在程序运行过程中不需要停止的线程置于下方
    t1 = threading.Thread(target=read_msg, args=(q2,))
    t2 = threading.Thread(target=send_msg, args=(q1,))
    t3 = threading.Thread(target=move_to_pose, args=(q1, x_start, y_start, x_goal, y_goal, usv_v))

    t.start()
    t1.start()
    while True:  # 响应监控端发出的初始化指令
        data_serial = q2.get()
        if len(data_serial) > 2:
            if data_serial[1] == 'S':
                ini_usv.terminate()  # 停止usv向监控端发送usv编号和初始位置线程
                t2.start()           # 开启串口发送线程
                t3.start()           # 开启usv运动进程
                print("连接监控端...")
                break


if __name__ == '__main__':
    # 线程间信息传递
    q1 = Queue()  # 用于发送信息
    q2 = Queue()  # 用于接收信息
    # 运动仿真参数，速度为定常的巡航速度或者遍历速度，角速度采用线性控制，现实应用中角速度可改为PID控制
    usv_Num = '#2'
    x_start = 10
    y_start = 0
    x_goal = 20
    y_goal = 20
    usv_v = 0.5
    Kp_alpha = 15
    Kp_beta = -3
    dt = 0.1  # 时间步长

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

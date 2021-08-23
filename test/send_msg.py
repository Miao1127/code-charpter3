# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：2019/12/1321:28
# 文件名：send_msg.py
# 开发工具：PyCharm
# 功能：发送端，1.测试将字典和列表整体发送；2.测试多线程同时发送数据；

import serial.tools.list_ports
import time
import numpy as np
from queue import LifoQueue
import threading


def send_cmd(sd_q, sp_q):
    """
    发送指令
    :return:
    """
    while True:
        if not sd_q.empty():               # 实现q1和q2无阻塞发送信息
            string_to_send = sd_q.get()
            ser.write(str.encode(string_to_send))
            print("sending message %s" % string_to_send)
            time.sleep(T_send)
            print(time.time())

        if not sp_q.empty():              # 实现q1和q2无阻塞发送信息
            string_to_send = sp_q.get()
            ser.write(str.encode(string_to_send))
            print("sending message %s" % string_to_send)
            time.sleep(T_send)
            print(time.time())


def d_msg(d_q):
    k = 0
    while True:
        # 测试发送usv到各个分区距离信息的发送
        d = [k, 40.14213562, 59.59797975, 93.56854249, 71.59797975, 77.39696962, 82.39696962, 124.39696962,
                         143.45079349, 125.59797975, 138.627417, 140.69848481, 170.59797975, 95.52691193, 43.14213562,
                         107.61017306, 159.75230868, 171.84062043, 64.48528137]
        d = [round(i, 2) for i in d]  # 需要限制每个数据的小数位数，否则容易超出串口单次发送的最大数据量
        s_msg = 'D' + '$' + str(k) + '$' + '\n'
        d_q.put(s_msg)
        k += 1
        # if k % 5 == 0:
        #     d_q.queue.clear()


def p_msg(p_q):
    while True:
        # 测试发送usv的决策信息
        plan = {'##3': [10, 17], '##4': [3, 18], '##1': [10, 17], '##2': [10, 17], '##5': [10, 17],
                'f': 0.7377635825268328,
                'u': ['##1', '##2', '##3', '##4', '##5']}
        s_msg = 'P' + '$' + str(plan) + '$'
        p_q.put(s_msg)
        time.sleep(2)


if __name__ == '__main__':
    q1 = LifoQueue(3)
    q2 = LifoQueue()
    t1 = threading.Thread(target=d_msg, args=(q1,))
    t2 = threading.Thread(target=p_msg, args=(q2,))
    t3 = threading.Thread(target=send_cmd, args=(q1, q2))
    # 串口通信参数
    serial_port = 'com4'
    serial_baud = 115200
    time_out = 10
    T_send = 0.5
    # 开启串口通信端口
    ser = serial.Serial(serial_port, baudrate=serial_baud, timeout=time_out)
    time.sleep(0.2)
    t1.start()
    t2.start()
    t3.start()


# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：2019/12/1321:33
# 文件名：receive_msg.py
# 开发工具：PyCharm
# 功能：接收信息，1.测试通过全局变量进行线程间的信息传递，结果失败；2.线程间建立多个quene进行信息交换；3.测试quene中get()方法每次只取
#                  一个数据

import serial.tools.list_ports
import time
import threading
from queue import LifoQueue


def read_msg(rd_q, rp_q):
    """
    接收信息
    :return:
    """
    print('开始接收数据')
    while True:
        if ser.in_waiting:
            r_data = ser.readline().decode("utf-8")
            r_data = r_data.split('$')
            # print("接收到的数据为：%s" % r_data)
            print(time.time())
            if r_data[0] == 'D':
                print('*'*5 + 'D' + '*'*5)
                d_mat = r_data[1]
                print(d_mat)
                rd_q.put(d_mat)
            if r_data[0] == 'P':
                print('*' * 5 + 'P' + '*' * 5)
                plan = r_data[1]
                print(plan)
                rp_q.put(plan)


def show(sd_q, sp_q):
    while True:
        if not sd_q.empty():
            d_mat = sd_q.get()
            print(d_mat)
        if not sp_q.empty():
            plan = sp_q.get()
            print(plan)


if __name__ == '__main__':
    # 串口通信参数
    serial_port = 'com6'
    serial_baud = 115200
    time_out = 10
    T_send = 0.5
    # 开启串口通信端口
    ser = serial.Serial(serial_port, baudrate=serial_baud, timeout=time_out)

    # 建立信息队列
    q1 = LifoQueue()
    q2 = LifoQueue()

    # 开启双线程
    t1 = threading.Thread(target=read_msg, args=(q1, q2))
    t2 = threading.Thread(target=show, args=(q1, q2))

    t1.start()
    t2.start()

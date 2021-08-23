# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：2019/12/1110:57
# 文件名：send_msg.py
# 开发工具：PyCharm
# 功能：串口发送数据

import serial
import time


def open_serial(serial_port, serial_baud, time_out):
    # 开启串口通信端口
    ser = serial.Serial(serial_port, baudrate=serial_baud, timeout=time_out)
    time.sleep(0.5)
    return ser


def send_cmd(s, t_send, command):
    """
    发送指令
    :return:
    """
    while True:
        # ser.write(str.encode('\n'))
        string_to_send = command + '\n'
        s.write(str.encode(string_to_send))
        print("sending message(%s)..." % string_to_send)
        time.sleep(t_send)
        print(time.time())


def read_msg(q1, s):
    """
    接收信息
    :return:
    """
    print('开始接收数据')
    while True:
        if s.in_waiting:
            data_received = s.readline().decode("gbk")
            data = data_received.split('$')
            q1.put(data)
            print("接收到的数据为：%s" % data)

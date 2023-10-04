from scapy.all import *
from scapy.layers.inet import IP, TCP, ICMP
from scapy.layers.l2 import Ether


import socket



def sendMoreFrame():
    with open("F:\modbus-data\output66.txt", "r", encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            modbus_payload = line
            modbus_packet = bytes.fromhex(modbus_payload)
            # 构造数据帧
            data = modbus_packet
            # 发送数据帧到目标地址
            sock.send(data)
            # 接收响应数据
            response = sock.recv(1024)
        sock.close()

def sendOneFrame():
    # 定义 Modbus 数据
    modbus_payload = "00ca0000001b01100000002014000000000016002b00200021002e0017002201bcfcca"
    modbus_packet = bytes.fromhex(modbus_payload)
    data = modbus_packet
    # 发送数据帧到目标地址
    sock.send(data)
    # 接收响应数据
    response = sock.recv(1024)

    sock.close()

if __name__ == "__main__":
    # 创建一个UDP套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 设置目标地址和端口
    target_ip = "127.0.0.1"
    target_port = 502
    # 连接到目标地址
    sock.connect((target_ip, target_port))
    sendOneFrame()
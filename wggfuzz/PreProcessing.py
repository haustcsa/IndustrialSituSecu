import numpy
from numpy import argmax
from torch.autograd import Variable
import matplotlib as plt
from torch import Tensor
import re
import os
import tensorflow as tsl
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset
import torch.nn.functional as F




# 删除文件的多余空格 处理wireshark初始数据************************
def dataDelete():
    path1 = "D:/data"
    path2 = "D:/data/"
    files = os.listdir(path1)
    for file in files:
        position = "F:\modbus-data\modbus0711.txt"
        position_out = "F:\modbus-data\modbusPre0711.txt"
        with open(position, 'r') as f:
            with open(position_out, 'w') as f_out:
                for line in f.readlines():
                    if line != '\n':
                        f_out.writelines(line.replace(line[0:6], '').replace(line[54:], '').replace(' ', ''))
                    else:
                        f_out.writelines(line)

def dataPre():
    with open("F:\\getModbusALL1.txt", "r", encoding='utf-8') as f:

        lines = f.read().splitlines()
        dataP = list()
        for line in lines:
            char_list = [line[i:i + 2] for i in range(0, len(line), 2)]
            dataP.append(char_list)
        dataP = [[int(ch, 16) for ch in data] for data in dataP]


        # 使用pad_sequence进行填充
        padded_sequences = pad_sequence([torch.tensor(seq) for seq in dataP], batch_first=True,padding_value=256)

        # 进行独热编码
        one_hot_sequence = F.one_hot(padded_sequences, num_classes=257)

        return one_hot_sequence,padded_sequences

# 填充字符，修改成独热编码************************
def dataLineNums1():
    # f1 = open("D:/data/data2.txt", "w")
    # define universe of possible input values
    alphabet = '0123456789abcdef'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))

    with open("F:\modbus-data\modbusPre0711.txt", "r", encoding='utf-8') as f:
        # 按换行符分割,读取文件所有行，存入list
        lines = f.read().splitlines()
        # print(len(lines))
        # 获取文件最长行
        res = max(lines, key=len, default='')
        length = len(res)  # 112
        data = list()
        print(length, [res])
        for line in lines:
            l = len(line)
            a = int((length - l) / 3)
            b = (length - l) % 3
            # print(l, '  a:', a, '  b:', b)
            # line = line + a * 'cad'
            # if b == 1:
            #     line = line + 'c'
            # if b == 2:
            #     line = line + 'ca'
            # print(len(line), [line])
            #
            integer_encoded = [char_to_int[char] for char in line]
            # print(integer_encoded)
            # one hot encode

            onehot_encoded = list()
            for value in integer_encoded:
                letter = [0 for _ in range(len(alphabet))]
                letter[value] = 1
                onehot_encoded.append(letter)
            for i in range(length-len(onehot_encoded)):
                onehot_encoded.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            # print(len(onehot_encoded))
            # print("##########")
            # print(numpy.array(onehot_encoded).shape)
            # print("##########@@@@@")
            # print(onehot_encoded)
            data.append(onehot_encoded)
            #
            # integer_encoded = np.array(integer_encoded)
            # data = np.append(integer_encoded)
            # f1.write(str(onehot_encoded) + '\n')
            # invert encoding
            inverted = int_to_char[argmax(onehot_encoded[0])]
            # f1.write(line + '\n')
            # print(inverted)
        real_datas = np.array(data)
        print(real_datas)
        real_datas = torch.LongTensor(real_datas)
    return real_datas


def rep_data():
    with open("D:/data/data2.txt", "r", encoding='utf-8') as f:
        lines = f.readlines()
        a = len(lines)
        b = len(re.findall('\[', lines[0])) - 1
        c = len(re.findall('\[.*?\]', lines[0])[0].split(', '))
        data_mat = np.ones(shape=(a, b, c), dtype=np.double)
        for i in range(a):
            line = lines[i].replace(']\n', '')[1:]
            list1 = re.findall('\[.*?\]', line)
            for j in range(b):
                list2 = list1[j][1:-1].split(', ')
                for k in range(c):
                    data_mat[i][j][k] = int(list2[k])
        real_datas = torch.LongTensor(data_mat)

        print(real_datas.shape)
        return real_datas


# 填充字符，修改成独热编码
def dataLineNums():
    f1 = open("D:/data/data2.txt", "w")
    # define universe of possible input values
    alphabet = '0123456789abcdef'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))

    f = open("D:/data/data1.txt", "r", encoding='utf-8')
    # print("xxxxxxx",f.readlines())
    with open("D:/data/data1.txt", "r", encoding='utf-8') as f:
        # 按换行符分割,读取文件所有行，存入list
        lines = f.read().splitlines()
        # print(len(lines))
        # 获取文件最长行
        res = max(lines, key=len, default='')
        length = len(res)  # 112
        data = list()
        # print(length, [res])
        for line in lines:
            l = len(line)
            a = int((length - l) / 3)
            b = (length - l) % 3
            # print(l, '  a:', a, '  b:', b)
            line = line + a * 'cad'
            if b == 1:
                line = line + 'c'
            if b == 2:
                line = line + 'ca'
            # print(len(line), [line])
            list_int = np.array([char_to_int[char] for char in line])
            # print(type(list_int),list_int)
            data.append(list_int)
        real_datas = np.array(data)

            #
            # integer_encoded = np.array(integer_encoded)
            # data = np.append(integer_encoded)
        real_datas = torch.LongTensor(real_datas)
    return real_datas

    # one hot encode
    # onehot_encoded = list()
    # for value in integer_encoded:
    #     letter = [0 for _ in range(len(alphabet))]
    #     letter[value] = 1
    #     onehot_encoded.append(letter)
    # print("##########")
    # print(numpy.array(onehot_encoded).shape)
    # print("##########@@@@@")
    # print(onehot_encoded)
    # f1.write(str(onehot_encoded) + '\n')
    # # invert encoding
    # inverted = int_to_char[argmax(onehot_encoded[0])]
    # # f1.write(line + '\n')
    # print(inverted)


def dataset():
    data = rep_data()



def Mydataset():
    real_datas = dataLineNums()
    print(real_datas.shape)
    # print('1111111111111111111111111',type(real_datas),real_datas.shape)
    real_datas = torch.LongTensor(real_datas)

    x_train = TensorDataset(real_datas)

    return x_train

def padCharater():
    file_path = 'D:/data/data1.txt'
    data = list()
    max_length = 0
    sequences = []

    alphabet = '0123456789abcdef'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # 读取txt文件
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # 获取文件最长行
        res = max(lines, key=len, default='')
        length = len(res)  # 112
        # 逐行处理
        for line in lines:
            list_int = np.array([char_to_int[char] for char in line])
            # print(type(list_int),list_int)
            data.append(list_int)
        print(data)


            # 移除换行符并将每个数字作为整数存储到列表中
            # sequence = [char for char in line.strip()]
            # 更新最长序列长度
            # max_length = max(max_length, len(sequence))
            # 将序列添加到sequences列表中
            # sequences.append(sequence)
            # list_int = np.array([char_to_int[char] for char in sequence])



    # 使用<pad>填充序列
    # padded_sequences = []
    # for sequence in sequences:
    #     padded_sequence = sequence +['<pad>'] * (max_length - len(sequence))
    #     padded_sequences.append(padded_sequence)

    # 输出结果
    # print(f"Max Length: {max_length}")
    # print("Padded Sequences:")
    # for sequence in sequences:
    #
    #     all_sequences.append(list_int)

def randomData():
    import random

    # 读取文件并将每一行存储在列表中
    with open('F:\getModbus24.txt', 'r') as file:
        lines = file.readlines()

    # 使用随机化算法对列表进行打乱
    random.shuffle(lines)

    # 将打乱后的列表写回到文件中
    with open('F:\getModbus24(1).txt', 'w') as file:
        file.writelines(lines)



if __name__ == "__main__":
    randomData()

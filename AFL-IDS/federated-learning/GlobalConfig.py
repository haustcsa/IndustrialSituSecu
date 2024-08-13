import numpy as np


class GlobalValues:
    def __init__(self):
        # 初始化一个空的二维数组
        self._global_2d_array = np.zeros((20, 1000), dtype=float)

    # 获取整个二维数组
    def get_global_2d_array(self):
        return self._global_2d_array


    # 设置二维数组中的某个元素
    def set_global_2d_array_element(self, row, col, value):
        row = int(row)  # 将row转换为整数
        col = int(col)  # 将col转换为整数

        self._global_2d_array[row][col] = value

    # 计算指定行的和
    def sum_row(self, row):
        if 0 <= row < len(self._global_2d_array):
            return sum(self._global_2d_array[row])
        else:
            raise IndexError("Row index out of range")

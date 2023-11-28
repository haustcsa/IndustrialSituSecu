# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/2/10 0:34
@file: random_range.py
@desc: https://www.coder.work/article/344931
"""
import math
import os

import ecdsa


def generate_random_number(curve=ecdsa.SECP256k1) -> int:
    """
    Generates a random number based on the given elliptic curve.
    基于椭圆曲线E并选取一个点 P，生成随机数r
    :return 示例：112985091235665144469915463903003347177563068442259414201636019349956393372861
    """
    n = curve.order
    random_bytes = os.urandom(math.ceil(n.bit_length() / 8))
    return int.from_bytes(random_bytes, byteorder="big") % n

# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/2/10 0:35
@file: identity.py
@desc: 身份
"""
import hashlib


def calculate_pseudo_identity(data: bytes):
    """Calculate the pseudo-identity of data using SHA-256 计算伪身份"""
    sha256 = hashlib.sha256()
    sha256.update(data)
    return sha256.hexdigest()

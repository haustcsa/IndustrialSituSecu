# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/4/7 22:03
@file: handle_cert.py
@desc: 处理证书相关的操作
"""
import base64
from pathlib import Path
from typing import NoReturn


def read_bytes_file(path: Path) -> str:
    """读取字节类型的文件"""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read())
    return data.decode()


def write_bytes_file(path: Path, content: str) -> NoReturn:
    """写入字节类型的文件"""
    with open(path, "wb") as f:
        f.write(base64.b64decode(content.encode()))

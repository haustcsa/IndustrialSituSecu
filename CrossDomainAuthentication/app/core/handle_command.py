# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/4/2 0:33
@file: handle_command.py
@desc: 调用系统命令
"""
import platform
import subprocess

from .logger import logger

system = platform.system()


def get_common_encoding() -> str:
    """
    获取当前操作系统的名称
    :return: 如果操作系统不是 Windows、Linux 或 macOS，返回默认编码
    """
    return "gbk" if system == "Windows" else "utf-8"


def get_common_separator() -> str:
    """
    在同一个子进程中运行多条命令，需要使用分隔符将这些命令串联起来，以便在一个进程中运行
    :return: 不同平台的分隔符
    """
    return " & " if system == "Windows" else "; "


def run_command(command: list[str]) -> tuple[int, str]:
    """
    运行命令
    :param command:
    :return:
    """
    separator = get_common_separator()
    command_str = separator.join(command)

    result = subprocess.run(
        command_str,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding=get_common_encoding(),
        timeout=10,
    )
    output = result.stdout
    logger.info(f"command: {command_str}\nreturncode: {result.returncode}\n{output}")
    return result.returncode, output

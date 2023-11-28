# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/4/2 0:35
@file: test_command.py
@desc: 测试调用系统命令
"""
import platform

from app.core import run_command

system = platform.system()


class TestRunCommand:
    def test_run_command(self):
        run_command(["cd ..", "dir" if system == "Windows" else "ls"])

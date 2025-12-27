import os
import logging
import json
import re
from subprocess import call

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def printDecisionAndMigrations(decision, migrations):  # printDecisionAndMigrations 函数的目的是打印一个决策列表，并根据每个决策是否已经迁移来改变其显示颜色。
	print('Decision: [', end='')
	for i, d in enumerate(decision):
		if d not in migrations: print(color.FAIL, end='')  # 红色加粗
		print(d, end='')
		if d not in migrations: print(color.ENDC, end='')  # 恢复正常颜色
		print(',', end='') if i != len(decision)-1 else print(']')
	print()
            

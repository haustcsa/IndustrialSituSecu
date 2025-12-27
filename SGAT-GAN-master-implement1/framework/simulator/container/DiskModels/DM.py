# DM 代表的数据结构或对象可以分配一个 "容器"（container）
class DM():
	def __init__(self):
		self.container = None

	def allocContainer(self, container):
		self.container = container
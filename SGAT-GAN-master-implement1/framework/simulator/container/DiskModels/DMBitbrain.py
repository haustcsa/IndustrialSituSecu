from .DM import *

class DMBitbrain(DM):
	def __init__(self, constant_size, read_list, write_list):
		super().__init__()
		self.constant_size = constant_size
		self.read_list = read_list
		self.write_list = write_list

	def disk(self):  # 依据当前的环境间隔与开始时间，动态从 read_list 和 write_list 中选择合适的读写值并返回。
		read_list_count = (self.container.env.interval - self.container.startAt) % len(self.read_list)
		write_list_count = (self.container.env.interval - self.container.startAt) % len(self.write_list)
		return self.constant_size, self.read_list[read_list_count], self.write_list[write_list_count]
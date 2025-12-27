from simulator.host.Disk import *
from simulator.host.RAM import *
from simulator.host.Bandwidth import *
# Host 类用于模拟计算机主机或虚拟机的资源管理，主要包括处理能力（IPS）、内存（RAM）、磁盘（Disk）、带宽（Bandwidth）等。它提供了一系列方法来获取当前主机的资源使用情况，并计算剩余资源。
class Host():   # 此外，Host 类还与功率模型（powermodel）和环境（env）紧密集成，可以根据容器的运行情况动态调整功耗、计算资源等。
	# IPS = Million Instructions per second capacity 
	# RAM = Ram in MB capacity
	# Disk = Disk characteristics capacity
	# Bw = Bandwidth characteristics capacity
	def __init__(self, ID, IPS, RAM, Disk, Bw, Latency, Powermodel, Environment):
		self.id = ID
		self.ipsCap = IPS
		self.ramCap = RAM
		self.diskCap = Disk
		self.bwCap = Bw
		self.latency = Latency
		self.powermodel = Powermodel
		self.powermodel.allocHost(self)
		self.powermodel.host = self
		self.env = Environment

	def getPower(self):
		return self.powermodel.power()

	def getPowerFromIPS(self, ips):
		return self.powermodel.powerFromCPU(min(100, 100 * (ips / self.ipsCap)))
		
	def getCPU(self):
		ips = self.getApparentIPS()
		return 100 * (ips / self.ipsCap)

	def getBaseIPS(self):
		# Get base ips count as sum of min ips of all containers
		ips = 0
		containers = self.env.getContainersOfHost(self.id)
		for containerID in containers:
			ips += self.env.getContainerByID(containerID).getBaseIPS()
		# assert ips <= self.ipsCap
		return ips

	def getApparentIPS(self):
		# Give containers remaining IPS for faster execution
		ips = 0
		containers = self.env.getContainersOfHost(self.id)
		for containerID in containers:
			ips += self.env.getContainerByID(containerID).getApparentIPS()
		# assert int(ips) <= self.ipsCap
		return int(ips)

	def getIPSAvailable(self):
		# IPS available is ipsCap - baseIPS
		# When containers allocated, existing ips can be allocated to
		# the containers
		return self.ipsCap - self.getBaseIPS()

	def getCurrentRAM(self):
		size, read, write = 0, 0, 0
		containers = self.env.getContainersOfHost(self.id)
		for containerID in containers:
			s, r, w = self.env.getContainerByID(containerID).getRAM()
			size += s; read += r; write += w
		# assert size <= self.ramCap.size
		# assert read <= self.ramCap.read
		# assert write <= self.ramCap.write
		return size, read, write

	def getRAMAvailable(self):
		size, read, write = self.getCurrentRAM()
		return self.ramCap.size - size, self.ramCap.read - read, self.ramCap.write - write

	def getCurrentDisk(self):
		size, read, write = 0, 0, 0
		containers = self.env.getContainersOfHost(self.id)
		for containerID in containers:
			s, r, w = self.env.getContainerByID(containerID).getDisk()
			size += s; read += r; write += w
		assert size <= self.diskCap.size
		assert read <= self.diskCap.read
		assert write <= self.diskCap.write
		return size, read, write

	def getDiskAvailable(self):
		size, read, write = self.getCurrentDisk()
		return self.diskCap.size - size, self.diskCap.read - read, self.diskCap.write - write

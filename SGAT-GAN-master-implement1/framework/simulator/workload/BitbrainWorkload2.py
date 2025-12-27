from .Workload import *
from simulator.container.IPSModels.IPSMBitbrain import *
from simulator.container.RAMModels.RMBitbrain import *
from simulator.container.DiskModels.DMBitbrain import *
from random import gauss, randint
from os import path, makedirs, listdir, remove
import wget
from zipfile import ZipFile
import shutil
import pandas as pd
import warnings
warnings.simplefilter("ignore")
#  用于模拟和生成容器负载的工作负载类。它使用从 Bitbrain 数据集（包含 CPU、内存、磁盘等性能数据）中读取的信息，随机选择并生成符合条件的工作负载模型。
# Intel Pentium III gives 2054 MIPS at 600 MHz
# Source: https://archive.vn/20130205075133/http://www.tomshardware.com/charts/cpu-charts-2004/Sandra-CPU-Dhrystone,449.html

############################
# def calculate_ips_multiplier(cpu_frequency, cpu_cores):
#     # 根据实际硬件参数动态计算 ips_multiplier
#     base_ips = 2054.0  # 基准值
#     return base_ips / (cpu_cores * cpu_frequency)
# # 获取实际硬件参数
# cpu_frequency = 600  # MHz
# cpu_cores = 2
# ips_multiplier = calculate_ips_multiplier(cpu_frequency, cpu_cores)
##################################


import psutil
import time
try:
    from get_information import get_system_metrics as external_metrics
    USE_EXTERNAL_METRICS = True
except ImportError:
    USE_EXTERNAL_METRICS = False
##########原，未经过数据清洗
ips_multiplier = 2054.0 / (2 * 600)

class BWGD2(Workload):
	def __init__(self, meanNumContainers, sigmaNumContainers):
		super().__init__()
		self.mean = meanNumContainers
		self.sigma = sigmaNumContainers

		dataset_path = "D:/PythonCode/SGATGAN-master-implement/simulator/workload/datasets/bitbrain/"

		if not path.exists(dataset_path):
			raise FileNotFoundError(
				f"Dataset path does not exist: {'dataset_pathsimulator/workload/datasets/bitbrain'}")
		if not listdir(dataset_path):
			raise ValueError(f"No files found in dataset path: {'dataset_pathsimulator/workload/datasets/bitbrain'}")

		# dataset_path = 'simulator/workload/datasets/bitbrain/'
		# if not path.exists(dataset_path):
		# 	makedirs(dataset_path)
		# 	print('Downloading Bitbrain Dataset')
		# 	url = 'http://gwa.ewi.tudelft.nl/fileadmin/pds/trace-archives/grid-workloads-archive/datasets/gwa-t-12/rnd.zip'
		# 	filename = wget.download(url); zf = ZipFile(filename, 'r'); zf.extractall(dataset_path); zf.close()
		# 	for f in listdir(dataset_path+'rnd/2013-9/'): shutil.move(dataset_path+'rnd/2013-9/'+f, dataset_path+'rnd/')
		# 	shutil.rmtree(dataset_path+'rnd/2013-7'); shutil.rmtree(dataset_path+'rnd/2013-8')
		# 	shutil.rmtree(dataset_path+'rnd/2013-9'); remove(filename)
		self.dataset_path = dataset_path
		self.disk_sizes = [1, 2, 3]
		self.meanSLA, self.sigmaSLA = 20, 3
		self.possible_indices = []
		# for i in range(1, 500):
		# 	df = pd.read_csv(self.dataset_path+'rnd/'+str(i)+'.csv', sep=';\t')
		# 	if (ips_multiplier*df['CPU usage [MHZ]']).to_list()[10] < 3000 and (ips_multiplier*df['CPU usage [MHZ]']).to_list()[10] > 500:
		# 		self.possible_indices.append(i)
		for filename in listdir(self.dataset_path):
			if filename.endswith('.csv'):
			# if filename.endswith('.txt'):
				file_path = path.join(self.dataset_path, filename)  #####加一行
				df = pd.read_csv(self.dataset_path + filename, sep=';\t')
				if 'CPU usage [MHZ]' in df and len(df) > 10:  # Ensure necessary columns and sufficient data
					cpu_usage = (ips_multiplier * df['CPU usage [MHZ]']).iloc[10]  # 通过计算每个 CSV 文件中第 10 行的数据来筛选有效的文件
					if 500 < cpu_usage < 3000:
						self.possible_indices.append(filename)

		if not self.possible_indices:
			raise ValueError("No valid CSV files found in the dataset.")


	def generateNewContainers(self, interval):   # 从筛选后的 .csv 文件中随机选择一个文件，读取文件中的数据，并使用这些数据为每个容器生成负载模型
		workloadlist = []
		for i in range(max(1,int(gauss(self.mean, self.sigma)))):
			CreationID = self.creation_id
			# index = self.possible_indices[randint(0,len(self.possible_indices)-1)]
			# df = pd.read_csv(self.dataset_path+'rnd/'+str(index)+'.csv', sep=';\t')
			index = randint(0, len(self.possible_indices) - 1)  ####
			filename = self.possible_indices[index]  ####
			print(f"Selected index: {index}, Filename: {filename}")  # 调试打印
			file_path = path.join(self.dataset_path, filename)  ####
			df = pd.read_csv(file_path, sep=';\t', engine='python')  ####

			sla = gauss(self.meanSLA, self.sigmaSLA)
			IPSModel = IPSMBitbrain((ips_multiplier*df['CPU usage [MHZ]']).to_list(), (ips_multiplier*df['CPU capacity provisioned [MHZ]']).to_list()[0], int(1.2*sla), interval + sla)
			RAMModel = RMBitbrain((df['Memory usage [KB]']/4000).to_list(), (df['Network received throughput [KB/s]']/1000).to_list(), (df['Network transmitted throughput [KB/s]']/1000).to_list())
			disk_size  = self.disk_sizes[index % len(self.disk_sizes)]
			DiskModel = DMBitbrain(disk_size, (df['Disk read throughput [KB/s]']/4000).to_list(), (df['Disk write throughput [KB/s]']/12000).to_list())
			workloadlist.append((CreationID, interval, IPSModel, RAMModel, DiskModel))
			self.creation_id += 1
		self.createdContainers += workloadlist
		self.deployedContainers += [False] * len(workloadlist)
		return self.getUndeployedContainers()

	# def generateNewContainers(self, interval):
	# 	workloadlist = []
	# 	print(f"\n[DEBUG] 生成新容器，当前creation_id: {self.creation_id}")
	#
	# 	# 获取实时系统指标
	# 	realtime_metrics = self.get_system_metrics()
	# 	df = pd.DataFrame([realtime_metrics])
	#
	# 	# 确定要生成的容器数量
	# 	num_containers = max(1, int(gauss(self.mean, self.sigma)))
	#
	# 	for i in range(num_containers):
	# 		CreationID = self.creation_id
	#
	# 		# 生成SLA
	# 		sla = gauss(self.meanSLA, self.sigmaSLA)
	#
	# 		# 创建各资源模型
	# 		IPSModel = IPSMBitbrain(
	# 			(ips_multiplier * df['CPU usage [MHZ]']).to_list(),
	# 			(ips_multiplier * df['CPU capacity provisioned [MHZ]']).to_list()[0],
	# 			int(1.2 * sla),
	# 			interval + sla
	# 		)
	# 		RAMModel = RMBitbrain(
	# 			(df['Memory usage [KB]'] / 4000).to_list(),
	# 			(df['Network received throughput [KB/s]'] / 1000).to_list(),
	# 			(df['Network transmitted throughput [KB/s]'] / 1000).to_list()
	# 		)
	# 		disk_size = self.disk_sizes[i % len(self.disk_sizes)]
	# 		DiskModel = DMBitbrain(
	# 			disk_size,
	# 			(df['Disk read throughput [KB/s]'] / 4000).to_list(),
	# 			(df['Disk write throughput [KB/s]'] / 12000).to_list()
	# 		)
	#
	# 		workloadlist.append((CreationID, interval, IPSModel, RAMModel, DiskModel))
	# 		self.creation_id += 1
	#
	# 	print(f"[DEBUG] 生成 {len(workloadlist)} 个容器，示例: {workloadlist[0][0] if workloadlist else '无'}")
	#
	# 	self.createdContainers += workloadlist
	# 	self.deployedContainers += [False] * len(workloadlist)
	#
	# 	return {
	# 		'containers': self.getUndeployedContainers(),
	# 		'fault_probs': [(cid, 0.5) for cid in range(len(workloadlist))]  # 模拟故障概率
	# 	}
	#
	# def get_system_metrics(self):
	# 	"""智能选择采集方式"""
	# 	if USE_EXTERNAL_METRICS:
	# 		try:
	# 			metrics = external_metrics()
	# 			# 验证字段完整性
	# 			assert all(k in metrics for k in [
	# 				"CPU usage [MHZ]", "Memory usage [KB]",
	# 				"Disk read throughput [KB/s]"
	# 			]), "指标字段不完整"
	# 			return metrics
	# 		except Exception as e:
	# 			print(f"[Fallback] 外部采集失败: {e}")
	#
	# 	# 回退实现
	# 	return {
	# 		"Timestamp [ms]": int(time.time() * 1000),
	# 		"CPU usage [MHZ]": psutil.cpu_percent() * 2500,  # 假设2.5GHz
	# 		"Memory usage [KB]": psutil.virtual_memory().used // 1024,
	# 		"Disk read throughput [KB/s]": 0.0,
	# 		"Network received throughput [KB/s]": 0.0
	# 	}

##########  经过数据清洗
# ips_multiplier = 2054.0 / (2 * 600)
#
#
# class BWGD2(Workload):
# 	def __init__(self, meanNumContainers, sigmaNumContainers):
# 		super().__init__()
# 		self.mean = meanNumContainers
# 		self.sigma = sigmaNumContainers
#
# 		dataset_path = "D:/PythonCode/SGATGAN-master/simulator/workload/datasets/bitbrain/"
#
# 		if not path.exists(dataset_path):
# 			raise FileNotFoundError(
# 				f"Dataset path does not exist: {'dataset_pathsimulator/workload/datasets/bitbrain'}")
# 		if not listdir(dataset_path):
# 			raise ValueError(f"No files found in dataset path: {'dataset_pathsimulator/workload/datasets/bitbrain'}")
#
# 		self.dataset_path = dataset_path
# 		self.disk_sizes = [1, 2, 3]
# 		self.meanSLA, self.sigmaSLA = 20, 3
# 		self.possible_indices = []
#
# 		for filename in listdir(self.dataset_path):
# 			if filename.endswith('.csv'):
# 				file_path = path.join(self.dataset_path, filename)
# 				try:
# 					df = pd.read_csv(file_path, sep=';', engine='python', on_bad_lines='skip')  # 跳过格式错误的行
# 					df.columns = df.columns.str.strip()  # 去除列名中的空格和制表符
# 					# print(f"File: {filename}, Columns: {df.columns}")  # 打印列名
#
# 					if 'CPU usage [MHZ]' in df and len(df) > 10:
# 						cpu_usage = df['CPU usage [MHZ]'].iloc[10]
# 						# print(f"File: {filename}, CPU usage: {cpu_usage}")  # 打印 CPU 使用率的值
#
# 						if -2 < cpu_usage < 2:  # 根据实际范围调整条件
# 							self.possible_indices.append(filename)
# 				except Exception as e:
# 					print(f"Error reading file {filename}: {e}")
#
# 		if not self.possible_indices:
# 			raise ValueError("No valid CSV files found in the dataset.")
#
# 	def generateNewContainers(self, interval):
# 		workloadlist = []
# 		for i in range(max(1, int(gauss(self.mean, self.sigma)))):
# 			CreationID = self.creation_id
# 			index = randint(0, len(self.possible_indices) - 1)
# 			filename = self.possible_indices[index]
# 			print(f"Selected index: {index}, Filename: {filename}")  # 调试打印
# 			file_path = path.join(self.dataset_path, filename)
# 			df = pd.read_csv(file_path, sep=';')  # 使用分号作为分隔符
#
# 			sla = gauss(self.meanSLA, self.sigmaSLA)
#
# 			# 由于数据已经标准化，直接使用标准化后的值
# 			IPSModel = IPSMBitbrain(
# 				df['CPU usage [MHZ]'].to_list(),
# 				df['CPU capacity provisioned [MHZ]'].to_list()[0],
# 				int(1.2 * sla),
# 				interval + sla
# 			)
# 			RAMModel = RMBitbrain(
# 				(df['Memory usage [KB]'] / 4000).to_list(),
# 				(df['Network received throughput [KB/s]'] / 1000).to_list(),
# 				(df['Network transmitted throughput [KB/s]'] / 1000).to_list()
# 			)
# 			disk_size = self.disk_sizes[index % len(self.disk_sizes)]
# 			DiskModel = DMBitbrain(
# 				disk_size,
# 				(df['Disk read throughput [KB/s]'] / 4000).to_list(),
# 				(df['Disk write throughput [KB/s]'] / 12000).to_list()
# 			)
# 			workloadlist.append((CreationID, interval, IPSModel, RAMModel, DiskModel))
# 			self.creation_id += 1
#
# 		self.createdContainers += workloadlist
# 		self.deployedContainers += [False] * len(workloadlist)
# 		return self.getUndeployedContainers()
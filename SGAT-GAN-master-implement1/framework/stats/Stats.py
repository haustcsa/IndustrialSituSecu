import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scheduler.GOBI import GOBIScheduler


#plt.style.use(['science'])
plt.style.use([r'D:\PythonCode\SimpleNet-master\venv\Lib\site-packages\scienceplots\styles\science.mplstyle'])

plt.rcParams["text.usetex"] = False
# 在调度系统中收集和保存不同的统计信息。它通过与模拟环境和调度器的交互来跟踪和记录多个指标
class Stats():
	def __init__(self, Environment, WorkloadModel, Datacenter, Scheduler): # 初始化了各类数据结构，设定环境、工作负载模型、数据中心和调度器，并准备了存储统计信息的数组
		self.env = Environment
		self.env.stats = self
		self.workload = WorkloadModel
		self.datacenter = Datacenter
		self.scheduler = Scheduler
		self.simulated_scheduler = GOBIScheduler('energy_latency_'+str(self.datacenter.num_hosts))
		self.simulated_scheduler.env = self.env
		self.time_series = np.zeros((1,3 * len(self.env.hostlist))) # 3 dims: cpu, ram-size, disk-size
		self.schedule_series = np.zeros((1, len(self.env.containerlist), len(self.env.hostlist)))

		# #### 初始化统计指标
		# self.true_positives = 0
		# self.false_positives = 0
		# self.true_negatives = 0
		# self.false_negatives = 0
		# self.precision = 0.0
		# self.recall = 0.0
		# self.f1_score = 0.0
		# # 初始化 anomaly_score
		# self.anomaly_score = -float('inf')  ######## 或者设置为合适的初始值，例如 None
		# self.class_score = []  # 例如，用列表存储不同类别的分数
		# self.loss = float('inf')  # 初始化 loss 为 0.0
		# self.aloss = float('inf')  # 初始化 loss 为 0.0
		# self.tloss = float('inf')  # 初始化 loss 为 0.0

		self.initStats()

	def initStats(self):	
		self.hostinfo = []
		self.workloadinfo = []
		self.activecontainerinfo = []
		self.allcontainerinfo = []
		self.metrics = []
		self.schedulerinfo = []

		# ##### 在这里初始化或重置统计相关的逻辑
		# self.true_positives = 0
		# self.false_positives = 0
		# self.true_negatives = 0
		# self.false_negatives = 0
		# self.precision = 0.0
		# self.recall = 0.0
		# self.f1_score = 0.0
		# self.anomaly_score = -float('inf')
		# self.class_score = []
		# self.loss = float('inf')
		# self.aloss = float('inf')
		# self.tloss = float('inf')

	def saveHostInfo(self):
		hostinfo = dict()
		hostinfo['interval'] = self.env.interval
		hostinfo['cpu'] = [host.getCPU() for host in self.env.hostlist]
		hostinfo['numcontainers'] = [len(self.env.getContainersOfHost(i)) for i,host in enumerate(self.env.hostlist)]
		hostinfo['power'] = [host.getPower() for host in self.env.hostlist]
		hostinfo['baseips'] = [host.getBaseIPS() for host in self.env.hostlist]
		hostinfo['ipsavailable'] = [host.getIPSAvailable() for host in self.env.hostlist]
		hostinfo['ipscap'] = [host.ipsCap for host in self.env.hostlist]
		hostinfo['apparentips'] = [host.getApparentIPS() for host in self.env.hostlist]
		hostinfo['ram'] = [host.getCurrentRAM() for host in self.env.hostlist]
		hostinfo['ramavailable'] = [host.getRAMAvailable() for host in self.env.hostlist]
		hostinfo['disk'] = [host.getCurrentDisk() for host in self.env.hostlist]
		hostinfo['diskavailable'] = [host.getDiskAvailable() for host in self.env.hostlist]
		cpulist, ramlist, disklist = hostinfo['cpu'], [i[0] for i in hostinfo['ram']], [i[0] for i in hostinfo['disk']]
		datapoint = np.concatenate([[cpulist[i], ramlist[i], disklist[i]] for i in range(len(cpulist))]).reshape(1, -1)
		self.time_series = np.append(self.time_series, datapoint, axis=0)
		datapoint = np.array([self.env.scheduler.result_cache])
		self.schedule_series = np.append(self.schedule_series, datapoint, axis=0)
		self.hostinfo.append(hostinfo)


		# ###### 检查每个字段是否返回正确的值
		# cpulist = hostinfo['cpu']
		# # ramlist = [i[0] if isinstance(i, (list, np.ndarray)) else i for i in hostinfo['ram']]
		# # disklist = [i[0] if isinstance(i, (list, np.ndarray)) else i for i in hostinfo['disk']]
		# # # 生成 datapoint，确保形状正确
		# # datapoint = np.array([[cpulist[i], ramlist[i], disklist[i]] for i in range(len(cpulist))]).flatten().reshape(1,-1)
		# cpulist = np.array(cpulist, dtype=np.float64)
		# ramlist = np.array([i[0] if isinstance(i, (list, np.ndarray)) else i for i in hostinfo['ram']],dtype=np.float64)
		# disklist = np.array([i[0] if isinstance(i, (list, np.ndarray)) else i for i in hostinfo['disk']],dtype=np.float64)
		# # 确保数组形状一致
		# print("CPU List Shape:", cpulist.shape)
		# print("RAM List Shape:", ramlist.shape)
		# print("Disk List Shape:", disklist.shape)
		# # 构建 datapoint
		# datapoint = np.concatenate((cpulist.reshape(-1, 1),
		# 							ramlist.reshape(-1, 1),
		# 							disklist.reshape(-1, 1)), axis=1).reshape(1, -1)
		# # 检查 time_series 初始化
		# if self.time_series.size == 0:
		# 	self.time_series = np.empty((0, datapoint.shape[1]), dtype=np.float64)
		# print("Datapoint Shape:", datapoint.shape)
		# print("Time Series Shape:", self.time_series.shape)
		# # 检查 self.time_series 的形状并更新
		# if self.time_series.shape[1:] != datapoint.shape[1:]:
		# 	raise ValueError("Shape mismatch between `datapoint` and `time_series`.")
		# self.time_series = np.append(self.time_series, datapoint, axis=0)
		# # 检查 scheduler.result_cache
		# datapoint = np.expand_dims(self.env.scheduler.result_cache, axis=0)
		# self.schedule_series = np.append(self.schedule_series, datapoint, axis=0)
		# self.hostinfo.append(hostinfo)



	def saveWorkloadInfo(self, deployed, migrations):
		workloadinfo = dict()
		workloadinfo['interval'] = self.env.interval
		workloadinfo['totalcontainers'] = len(self.workload.createdContainers)
		if self.workloadinfo != []:
			workloadinfo['newcontainers'] = workloadinfo['totalcontainers'] - self.workloadinfo[-1]['totalcontainers'] 
		else:
			workloadinfo['newcontainers'] = workloadinfo['totalcontainers']
		workloadinfo['deployed'] = len(deployed)
		workloadinfo['migrations'] = len(migrations)
		workloadinfo['inqueue'] = len(self.workload.getUndeployedContainers())
		self.workloadinfo.append(workloadinfo)

	def saveContainerInfo(self):
		containerinfo = dict()
		containerinfo['interval'] = self.env.interval
		containerinfo['activecontainers'] = self.env.getNumActiveContainers()
		containerinfo['ips'] = [(c.getBaseIPS() if c else 0) for c in self.env.containerlist]
		containerinfo['apparentips'] = [(c.getApparentIPS() if c else 0) for c in self.env.containerlist]
		containerinfo['ram'] = [(c.getRAM() if c else 0) for c in self.env.containerlist]
		containerinfo['disk'] = [(c.getDisk() if c else 0) for c in self.env.containerlist]
		containerinfo['creationids'] = [(c.creationID if c else -1) for c in self.env.containerlist]
		containerinfo['hostalloc'] = [(c.getHostID() if c else -1) for c in self.env.containerlist]
		containerinfo['active'] = [(c.active if c else False) for c in self.env.containerlist]
		self.activecontainerinfo.append(containerinfo)

	def saveAllContainerInfo(self):
		containerinfo = dict()
		allCreatedContainers = [self.env.getContainerByCID(cid) for cid in list(np.where(self.workload.deployedContainers)[0])]
		containerinfo['interval'] = self.env.interval
		if self.datacenter.__class__.__name__ == 'Datacenter':
			containerinfo['application'] = [self.env.getContainerByCID(cid).application for cid in list(np.where(self.workload.deployedContainers)[0])]
		containerinfo['ips'] = [(c.getBaseIPS() if c.active else 0) for c in allCreatedContainers]
		containerinfo['create'] = [(c.createAt) for c in allCreatedContainers]
		containerinfo['start'] = [(c.startAt) for c in allCreatedContainers]
		containerinfo['destroy'] = [(c.destroyAt) for c in allCreatedContainers]
		containerinfo['apparentips'] = [(c.getApparentIPS() if c.active else 0) for c in allCreatedContainers]
		containerinfo['ram'] = [(c.getRAM() if c.active else 0) for c in allCreatedContainers]
		containerinfo['disk'] = [(c.getDisk() if c.active else 0) for c in allCreatedContainers]
		containerinfo['hostalloc'] = [(c.getHostID() if c.active else -1) for c in allCreatedContainers]
		containerinfo['active'] = [(c.active) for c in allCreatedContainers]
		self.allcontainerinfo.append(containerinfo)

	def saveMetrics(self, destroyed, migrations):
		import numpy as np

		# 初始化总迁移次数（如果尚未初始化）
		if not hasattr(self, 'total_migrations'):
			self.total_migrations = 0

		metrics = dict()
		metrics['interval'] = self.env.interval  # 当前的时间间隔
		metrics['numdestroyed'] = len(destroyed)  # 在当前时间间隔内被销毁的容器数量
		metrics['nummigrations'] = len(migrations)  # 在当前时间间隔内发生迁移的容器数量
		metrics['energy'] = [host.getPower() * self.env.intervaltime for host in self.env.hostlist]  # 每个主机在当前时间间隔内的能耗
		metrics['energytotalinterval'] = np.sum(metrics['energy'])  # 当前时间间隔内的总能耗
		metrics['energypercontainerinterval'] = np.sum(metrics['energy']) / self.env.getNumActiveContainers()  # 当前时间间隔内每个容器的平均能耗

		############################################# 打印计算结果
		print("能耗能耗能耗能耗能耗能耗能耗能耗能耗能耗能耗能耗")
		# print("Energy per host: ", metrics['energy'])  # 打印每个主机的能耗
		# print("Total energy for the interval: ", metrics['energytotalinterval'])  # 打印总能耗
		print("Energy per container during the interval: ", metrics['energypercontainerinterval'])  # 打印每个容器的能耗
		metrics['average_interval_energy'] = np.sum(metrics['energy']) / self.env.getNumActiveContainers()
		metrics['average_host_energy'] = np.mean(metrics['energy'])
		print(f"Average Host Energy: {metrics['average_host_energy']}")
		print(f"Average Interval Energy per Container: {metrics['average_interval_energy']}")

		metrics['responsetime'] = [c.totalExecTime + c.totalMigrationTime for c in destroyed]  # 每个被销毁容器的响应时间
		metrics['avgresponsetime'] = np.average(metrics['responsetime']) if len(
			destroyed) > 0 else 0  # 当前时间间隔内所有被销毁容器的平均响应时间
		metrics['migrationtime'] = [c.totalMigrationTime for c in destroyed]  # 每个被销毁容器的迁移时间
		metrics['avgmigrationtime'] = np.average(metrics['migrationtime']) if len(
			destroyed) > 0 else 0  # 当前时间间隔内所有被销毁容器的平均迁移时间
		metrics['slaviolations'] = len(np.where([c.destroyAt > c.sla for c in destroyed]))  # 当前时间间隔内违反 SLA（服务级别协议）的容器数量
		metrics['slaviolationspercentage'] = metrics['slaviolations'] * 100.0 / len(destroyed) if len(
			destroyed) > 0 else 0  # 当前时间间隔内违反 SLA 的容器百分比
		metrics['waittime'] = [c.startAt - c.createAt for c in destroyed]  # 每个被销毁容器的等待时间

		# 累加当前时间间隔的迁移次数
		self.total_migrations += metrics['nummigrations']

		# 创建或打开文件进行写入
		file_path = r"D:\PythonCode\SGATGAN-master-implement\stats\results.txt"

		##################################### 打开文件并写入结果
		with open(file_path, 'a') as file:
			file.write(f"\n\n")
			# 写入基础信息
			file.write(f"Interval: {metrics['interval']}\n")
			file.write(f"Number of destroyed: {metrics['numdestroyed']}\n")
			file.write(f"Number of migrations: {metrics['nummigrations']}\n")



			# file.write("Energy per host (in units of power):\n")
			# for energy in metrics['energy']:
			# 	file.write(f"{energy:.6f} units\n")  # 保持原始能耗值，不做单位换算
			# 总能耗（保持原始单位）
			file.write(f"Total energy for the interval (in units): {metrics['energytotalinterval']:.6f} units\n")
			# 每个容器的能耗（保持原始单位）
			file.write(
				f"Energy per container during the interval (in units): {metrics['energypercontainerinterval']:.6f} units\n")
			# 输出新的两个能耗指标（保持原始单位）
			file.write(f"Average interval energy (in units): {metrics['average_interval_energy']:.6f} units\n")
			file.write(f"Average host energy (in units): {metrics['average_host_energy']:.6f} units\n")

			# 能耗部分（将能耗转换为千瓦时）
			# file.write("Energy per host (in kWh):\n")
			# for energy in metrics['energy']:
			# 	file.write(f"{energy / 3_600_000:.6f} kWh\n")
			# # 总能耗（转换为千瓦时）
			# file.write(
			# 	f"Total energy for the interval (in kWh): {metrics['energytotalinterval'] / 3_600_000:.6f} kWh\n")
			# # 每个容器的能耗（转换为千瓦时）
			# file.write(
			# 	f"Energy per container during the interval (in kWh): {metrics['energypercontainerinterval'] / 3_600_000:.6f} kWh\n")
			# # 输出新的两个能耗指标
			# file.write(f"Average interval energy (in kWh): {metrics['average_interval_energy']:.6f} kWh\n")
			# file.write(f"Average host energy (in kWh): {metrics['average_host_energy']:.6f} kWh\n")
			# 响应时间部分
			file.write(f"Average response time: {metrics['avgresponsetime']:.6f} seconds\n")
			# 迁移时间部分
			file.write(f"Average migration time: {metrics['avgmigrationtime']:.6f} seconds\n")
			# SLA违规情况
			file.write(f"SLA Violations: {metrics['slaviolations']}\n")
			file.write(f"SLA Violations Percentage: {metrics['slaviolationspercentage']:.2f}%\n")
			# 等待时间部分
			file.write(f"Average wait time: {np.average(metrics['waittime']):.6f} seconds\n")
			# 写入总迁移次数
			file.write(f"Total migrations across all intervals: {self.total_migrations}\n")
		print("Results written to file.")
		########################################

		self.metrics.append(metrics)

	def saveSchedulerInfo(self, selectedcontainers, decision, schedulingtime):
		schedulerinfo = dict()
		schedulerinfo['interval'] = self.env.interval
		schedulerinfo['selection'] = selectedcontainers
		schedulerinfo['decision'] = decision
		schedulerinfo['schedule'] = [(c.id, c.getHostID()) if c else (None, None) for c in self.env.containerlist]
		schedulerinfo['schedulingtime'] = schedulingtime
		if self.datacenter.__class__.__name__ == 'Datacenter':
			schedulerinfo['migrationTime'] = self.env.intervalAllocTimings[-1]
		self.schedulerinfo.append(schedulerinfo)

	def saveStats(self, deployed, migrations, destroyed, selectedcontainers, decision, schedulingtime):	
		self.saveHostInfo()
		self.saveWorkloadInfo(deployed, migrations)
		self.saveContainerInfo()
		self.saveAllContainerInfo()
		self.saveMetrics(destroyed, migrations)
		self.saveSchedulerInfo(selectedcontainers, decision, schedulingtime)

	def runSimulationGOBI(self):
		host_alloc = []; container_alloc = [-1] * len(self.env.hostlist)
		for i in range(len(self.env.hostlist)):
			host_alloc.append([])
		for c in self.env.containerlist:
			if c and c.getHostID() != -1: 
				host_alloc[c.getHostID()].append(c.id) 
				container_alloc[c.id] = c.getHostID()
		selected = self.simulated_scheduler.selection()
		decision = self.simulated_scheduler.filter_placement(self.simulated_scheduler.placement(selected))
		for cid, hid in decision:
			if self.env.getPlacementPossible(cid, hid) and container_alloc[cid] != -1:
				host_alloc[container_alloc[cid]].remove(cid)
				host_alloc[hid].append(cid)
		energytotalinterval_pred = 0
		for hid, cids in enumerate(host_alloc):
			ips = 0
			for cid in cids: ips += self.env.containerlist[cid].getApparentIPS()
			energytotalinterval_pred += self.env.hostlist[hid].getPowerFromIPS(ips)
		return energytotalinterval_pred*self.env.intervaltime, max(0, np.mean([metric_d['avgresponsetime'] for metric_d in self.metrics[-5:]]))

	def runSimulation(self, schedule_data):
		host_alloc = []; container_alloc = [-1] * len(self.env.hostlist)
		for i in range(len(self.env.hostlist)): host_alloc.append([])
		for c in self.env.containerlist:
			if c and c.getHostID() != -1: 
				host_alloc[c.getHostID()].append(c.id) 
				container_alloc[c.id] = c.getHostID()
		decision = []
		for cid in np.concatenate(host_alloc):
			cid = int(cid)
			one_hot = schedule_data[cid].tolist()
			new_host = one_hot.index(max(one_hot))
			if container_alloc[cid] != new_host: decision.append((cid, new_host))
		decision = self.simulated_scheduler.filter_placement(decision)
		for cid, hid in decision:
			if self.env.getPlacementPossible(cid, hid) and container_alloc[cid] != -1:
				host_alloc[container_alloc[cid]].remove(cid)
				host_alloc[hid].append(cid)
		energytotalinterval_pred = 0
		for hid, cids in enumerate(host_alloc):
			ips = 0
			for cid in cids: ips += self.env.containerlist[cid].getApparentIPS()
			energytotalinterval_pred += self.env.hostlist[hid].getPowerFromIPS(ips)
		return energytotalinterval_pred*self.env.intervaltime, max(0, np.mean([metric_d['avgresponsetime'] for metric_d in self.metrics[-5:]]))

	########################################################################################################

	def generateGraphsWithInterval(self, dirname, listinfo, obj, metric, metric2=None):
		fig, axes = plt.subplots(len(listinfo[0][metric]), 1, sharex=True,figsize=(4, 0.5*len(listinfo[0][metric])))
		title = obj + '_' + metric + '_with_interval' 
		totalIntervals = len(listinfo)
		x = list(range(totalIntervals))
		metric_with_interval = []; metric2_with_interval = []
		ylimit = 0; ylimit2 = 0
		for hostID in range(len(listinfo[0][metric])):
			metric_with_interval.append([listinfo[interval][metric][hostID] for interval in range(totalIntervals)])
			ylimit = max(ylimit, max(metric_with_interval[-1]))
			if metric2:
				metric2_with_interval.append([listinfo[interval][metric2][hostID] for interval in range(totalIntervals)])
				ylimit2 = max(ylimit2, max(metric2_with_interval[-1]))
		for hostID in range(len(listinfo[0][metric])):
			axes[hostID].set_ylim(0, max(ylimit, ylimit2))
			axes[hostID].plot(x, metric_with_interval[hostID])
			if metric2:
				axes[hostID].plot(x, metric2_with_interval[hostID])
			axes[hostID].set_ylabel(obj[0].capitalize()+" "+str(hostID))
			axes[hostID].grid(b=True, which='both', color='#eeeeee', linestyle='-')
		plt.tight_layout(pad=0)
		plt.savefig(dirname + '/' + title + '.pdf')

	def generateMetricsWithInterval(self, dirname):
		fig, axes = plt.subplots(9, 1, sharex=True, figsize=(4, 5))
		x = list(range(len(self.metrics)))
		res = {}
		for i,metric in enumerate(['numdestroyed', 'nummigrations', 'energytotalinterval', 'avgresponsetime',\
			 'avgmigrationtime', 'slaviolations', 'slaviolationspercentage', 'waittime', 'energypercontainerinterval']):
			metric_with_interval = [self.metrics[i][metric] for i in range(len(self.metrics))] if metric != 'waittime' else \
				[sum(self.metrics[i][metric]) for i in range(len(self.metrics))]
			axes[i].plot(x, metric_with_interval)
			axes[i].set_ylabel(metric, fontsize=5)
			axes[i].grid(b=True, which='both', color='#eeeeee', linestyle='-')
			res[metric] = sum(metric_with_interval)
			print("Summation ", metric, " = ", res[metric])
		print('Average energy (sum energy interval / sum numdestroyed) = ', res['energytotalinterval']/res['numdestroyed'])
		plt.tight_layout(pad=0)
		plt.savefig(dirname + '/' + 'Metrics' + '.pdf')

	def generateWorkloadWithInterval(self, dirname):
		fig, axes = plt.subplots(5, 1, sharex=True, figsize=(4, 5))
		x = list(range(len(self.workloadinfo)))
		for i,metric in enumerate(['totalcontainers', 'newcontainers', 'deployed', 'migrations', 'inqueue']):
			metric_with_interval = [self.workloadinfo[i][metric] for i in range(len(self.workloadinfo))]
			axes[i].plot(x, metric_with_interval)
			axes[i].set_ylabel(metric)
			axes[i].grid(b=True, which='both', color='#eeeeee', linestyle='-')
		plt.tight_layout(pad=0)
		plt.savefig(dirname + '/' + 'Workload' + '.pdf')

	########################################################################################################

	def generateCompleteDataset(self, dirname, data, name):
		title = name + '_with_interval' 
		metric_with_interval = []
		headers = list(data[0].keys())
		for datum in data:
			metric_with_interval.append([datum[value] for value in datum.keys()])
		df = pd.DataFrame(metric_with_interval, columns=headers)
		df.to_csv(dirname + '/' + title + '.csv', index=False)

	def generateTimeSeriesDataset(self, dirname):
		title = 'time_series'
		np.save(f'{dirname}/time_series.npy', self.time_series)
		np.save(f'{dirname}/schedule_series.npy', self.schedule_series)
		headers = np.concatenate([[f'cpu_{i}', f'ram_{i}', f'disk_{i}'] for i in range(len(self.env.hostlist))])
		df = pd.DataFrame(self.time_series, columns=headers)
		df.to_csv(dirname + '/' + title + '.csv', index=False)

	def generateDatasetWithInterval(self, dirname, metric, objfunc, metric2=None, objfunc2=None):
		title = metric + '_' + (metric2 + '_' if metric2 else "") + (objfunc + '_' if objfunc else "") + (objfunc2 + '_' if objfunc2 else "") + 'with_interval' 
		totalIntervals = len(self.hostinfo)
		metric_with_interval = []; metric2_with_interval = [] # metric1 is of host and metric2 is of containers
		host_alloc_with_interval = []; objfunc2_with_interval = []
		objfunc_with_interval = []
		for interval in range(totalIntervals-1):
			metric_with_interval.append([self.hostinfo[interval][metric][hostID] for hostID in range(len(self.hostinfo[0][metric]))])
			host_alloc_with_interval.append([self.activecontainerinfo[interval]['hostalloc'][cID] for cID in range(len(self.activecontainerinfo[0]['hostalloc']))])
			objfunc_with_interval.append(self.metrics[interval+1][objfunc])
			if metric2:
				metric2_with_interval.append(self.activecontainerinfo[interval][metric2])
			if objfunc2:
				objfunc2_with_interval.append(self.metrics[interval+1][objfunc2])
		df = pd.DataFrame(metric_with_interval)
		if metric2: df = pd.concat([df, pd.DataFrame(metric2_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(host_alloc_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(objfunc_with_interval)], axis=1)
		if objfunc2: df = pd.concat([df, pd.DataFrame(objfunc2_with_interval)], axis=1)
		df.to_csv(dirname + '/' + title + '.csv' , header=False, index=False)

	def generateDatasetWithInterval2(self, dirname, metric, metric2, metric3, metric4, objfunc, objfunc2):
		title = metric + '_' + metric2 + '_'  + metric3 + '_'  + metric4 + '_'  +objfunc + '_' + objfunc2 + '_' + 'with_interval' 
		totalIntervals = len(self.hostinfo)
		metric_with_interval = []; metric2_with_interval = [] 
		metric3_with_interval = []; metric4_with_interval = []
		host_alloc_with_interval = []; objfunc2_with_interval = []
		objfunc_with_interval = []
		for interval in range(totalIntervals-1):
			metric_with_interval.append([self.hostinfo[interval][metric][hostID] for hostID in range(len(self.hostinfo[0][metric]))])
			host_alloc_with_interval.append([self.activecontainerinfo[interval]['hostalloc'][cID] for cID in range(len(self.activecontainerinfo[0]['hostalloc']))])
			objfunc_with_interval.append(self.metrics[interval+1][objfunc])
			metric2_with_interval.append(self.activecontainerinfo[interval][metric2])
			metric3_with_interval.append(self.metrics[interval][metric3])
			metric4_with_interval.append(self.metrics[interval][metric4])
			objfunc2_with_interval.append(self.metrics[interval+1][objfunc2])
		df = pd.DataFrame(metric_with_interval)
		df = pd.concat([df, pd.DataFrame(metric2_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(host_alloc_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(metric3_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(metric4_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(objfunc_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(objfunc2_with_interval)], axis=1)
		df.to_csv(dirname + '/' + title + '.csv' , header=False, index=False)

	def generateGraphs(self, dirname):
		# self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'cpu')
		# self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'numcontainers')
		# self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'power')
		# self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'baseips', 'apparentips')
		# self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'ipscap', 'apparentips')
		# self.generateGraphsWithInterval(dirname, self.activecontainerinfo, 'container', 'ips', 'apparentips')
		# self.generateGraphsWithInterval(dirname, self.activecontainerinfo, 'container', 'hostalloc')
		# self.generateMetricsWithInterval(dirname)
		# self.generateWorkloadWithInterval(dirname)
		self.generateTimeSeriesDataset(dirname)

	def generateDatasets(self, dirname):
		# self.generateDatasetWithInterval(dirname, 'cpu', objfunc='energytotalinterval')
		self.generateDatasetWithInterval(dirname, 'cpu', metric2='apparentips', objfunc='energytotalinterval', objfunc2='avgresponsetime')
		# self.generateDatasetWithInterval2(dirname, 'cpu', 'apparentips', 'energytotalinterval_pred', 'avgresponsetime_pred', objfunc='energytotalinterval', objfunc2='avgresponsetime')
		
	def generateCompleteDatasets(self, dirname):
		self.generateCompleteDataset(dirname, self.hostinfo, 'hostinfo')
		self.generateCompleteDataset(dirname, self.workloadinfo, 'workloadinfo')
		self.generateCompleteDataset(dirname, self.metrics, 'metrics')
		self.generateCompleteDataset(dirname, self.activecontainerinfo, 'activecontainerinfo')
		self.generateCompleteDataset(dirname, self.allcontainerinfo, 'allcontainerinfo')
		self.generateCompleteDataset(dirname, self.schedulerinfo, 'schedulerinfo')

	# def update_metrics(self, TP, FP, FN):
	# 	self.true_positives += TP
	# 	self.false_positives += FP
	# 	self.false_negatives += FN
	#
	# def save_epoch_data(self, epoch, TP, FP, FN):
	# 	self.time_series.append([epoch, TP, FP, FN])
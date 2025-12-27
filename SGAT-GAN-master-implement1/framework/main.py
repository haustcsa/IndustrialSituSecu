import os, sys, stat
import sys
import optparse
import logging as logger
import configparser
import pickle
import shutil
import sqlite3
import platform
from time import time
from subprocess import call
from os import system, rename
from tqdm import tqdm  ###########
# from recovery.SGATGANSrc.src.train import accuracy  ########
# from scheduler.BaGTI.train import backprop, accuracy
# Framework imports
from framework.Framework import *
from framework.database.Database import *
from framework.datacenter.Datacenter_Setup import *
from framework.datacenter.Datacenter import *
from framework.workload.DeFogWorkload import *

# Simulator imports
from simulator.Simulator import *
from simulator.environment.AzureFog import *
from simulator.environment.RPiEdge import *
from simulator.environment.BitbrainFog import *
from simulator.workload.BitbrainWorkload_GaussianDistribution import *
from simulator.workload.BitbrainWorkload2 import *

# Scheduler imports
from scheduler.IQR_MMT_Random import IQRMMTRScheduler
from scheduler.MAD_MMT_Random import MADMMTRScheduler
from scheduler.MAD_MC_Random import MADMCRScheduler
from scheduler.LR_MMT_Random import LRMMTRScheduler
from scheduler.Random_Random_FirstFit import RFScheduler
from scheduler.Random_Random_LeastFull import RLScheduler
from scheduler.RLR_MMT_Random import RLRMMTRScheduler
from scheduler.Threshold_MC_Random import TMCRScheduler
from scheduler.Random_Random_Random import RandomScheduler
from scheduler.HGP_LBFGS import HGPScheduler
from scheduler.GA import GAScheduler
from scheduler.GOBI import GOBIScheduler
from scheduler.GOBI2 import GOBI2Scheduler
from scheduler.DRL import DRLScheduler
from scheduler.DQL import DQLScheduler
from scheduler.POND import PONDScheduler
from scheduler.SOGOBI import SOGOBIScheduler
from scheduler.SOGOBI2 import SOGOBI2Scheduler
from scheduler.HGOBI import HGOBIScheduler
from scheduler.HGOBI2 import HGOBI2Scheduler
from scheduler.HSOGOBI import HSOGOBIScheduler
from scheduler.HSOGOBI2 import HSOGOBI2Scheduler

# Recovery imports
from recovery.Recovery import Recovery
from recovery.SGATGAN import SGATGANRecovery
from recovery.PCFT import PCFTRecovery
from recovery.DFTM import DFTMRecovery
from recovery.ECLB import ECLBRecovery
from recovery.CMODLB import CMODLBRecovery

from fault_detector import FaultDetector

# Auxiliary imports
from stats.Stats import *
from utils.Utils import *
from pdb import set_trace as bp
import torch
import matplotlib.pyplot as plt
plt.style.use('./science.mplstyle')  # 确保路径正确
# plt.style.use('seaborn')  # 或者 'ggplot'
import matplotlib as mpl
import warnings
import matplotlib.font_manager as fm
warnings.filterwarnings("ignore", message="findfont: Font family 'Maven Pro' not found.")
plt.rcParams['font.family'] = 'Arial'  # 或者其他已安装的字体
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{C:/Users/Lenovo/AppData/Local/MiKTeX/fonts/map/pdftex/pdftex.map}'


# 禁用 LaTeX 渲染
mpl.rcParams['text.usetex'] = False

usage = "usage: python main.py -e <environment> -m <mode> # empty environment run simulator"

parser = optparse.OptionParser(usage=usage)
parser.add_option("-e", "--environment", action="store", dest="env", default="", 
					help="Environment is AWS, Openstack, Azure, VLAN, Vagrant")
parser.add_option("-m", "--mode", action="store", dest="mode", default="0", 
					help="Mode is 0 (Create and destroy), 1 (Create), 2 (No op), 3 (Destroy)")
opts, args = parser.parse_args()
# 打印解析后的参数
print(f"Environment: {opts.env}")
print(f"Mode: {opts.mode}")
# Global constants


NUM_SIM_STEPS = 3
HOSTS = 16 if opts.env == '' else 16
CONTAINERS = HOSTS  # 定义容器的数量，默认等于主机的数量（HOSTS）。这可能意味着每个主机默认运行一个容器。
TOTAL_POWER = 1000  # 整个仿真环境的总功率预算。这是一个性能或能耗的上限约束。
ROUTER_BW = 10000   # 定义路由器的带宽，表示整个仿真环境的网络传输能力上限
INTERVAL_TIME = 300 # seconds  定义仿真时间步之间的间隔时间，单位为秒。在实际中可能用于模拟真实系统的时间流逝
NEW_CONTAINERS = 1
DB_NAME = ''
DB_HOST = ''
DB_PORT = 0   # 数据库端口
HOSTS_IP = []
logFile = 'COSCO.log'

if len(sys.argv) > 1:
	with open(logFile, 'w'): os.utime(logFile, None)

def initalizeEnvironment(environment, logger):
	print('初始环境加载')      ###################
	if environment != '':
		# Initialize the db
		print('111111')
		db = Database(DB_NAME, DB_HOST, DB_PORT)

	# Initialize simple fog datacenter
	''' Can be SimpleFog, BitbrainFog, AzureFog // Datacenter '''
	if environment != '':
		print('222222222')
		datacenter = Datacenter(HOSTS_IP, environment, 'Virtual')
	else:
		print('3333333333')
		datacenter = RPiEdgeDatacenter(HOSTS)

	# Initialize workload
	''' Can be SWSD, BWGD, BWGD2 // DFW '''
	if environment != '':
		print('444444444')
		workload = DFW(NEW_CONTAINERS, 1.5, db)
	else:
		print('55555555555555')
		workload = BWGD2(NEW_CONTAINERS, 1.5)
	print('66666666666')
	# Initialize scheduler
	''' Can be LRMMTR, RF, RL, RM, Random, RLRMMTR, TMCR, TMMR, TMMTR, GA, GOBI (arg = 'energy_latency_'+str(HOSTS)) '''
	scheduler = GOBIScheduler('energy_latency_'+str(HOSTS))
	print('777777777777')
	# Initialize recovery
	''' Can be SGATGANRecovery, PCFTRecovery, DFTMRecovery, ECLBRecovery, CMODLBRecovery '''
	recovery = SGATGANRecovery(HOSTS, environment, training = False)
	# recovery = SGATGANRecovery(HOSTS, environment, training=True)
	# Initialize Environment
	hostlist = datacenter.generateHosts()
	if environment != '':
		print('88888888')
		env = Framework(scheduler, recovery, CONTAINERS, INTERVAL_TIME, hostlist, db, environment, logger)
	else:
		print('9999999999')
		env = Simulator(TOTAL_POWER, ROUTER_BW, scheduler, recovery, CONTAINERS, INTERVAL_TIME, hostlist) # 该模拟器通过提供的资源（如主机、带宽、容器、调度器等），进行容器的分配、执行和迁移等一系列操作，并在每个步骤中更新容器的状态
		print(env)

	# Execute first step
	torch.compile()
	newcontainerinfos = workload.generateNewContainers(env.interval) # New containers info
	deployed = env.addContainersInit(newcontainerinfos) # Deploy new containers and get container IDs
	start = time()
	decision = scheduler.placement(deployed) # Decide placement using container ids
	schedulingTime = time() - start
	migrations = env.allocateInit(decision) # Schedule containers
	workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed)) # Update workload allocated using creation IDs
	print("Deployed containers' creation IDs:", env.getCreationIDs(migrations, deployed))
	print("Containers in host:", env.getContainersInHosts())
	print("Schedule:", env.getActiveContainerList())
	printDecisionAndMigrations(decision, migrations)
	print("初始化schedulingTime:", schedulingTime)      ########################
	# Initialize stats
	stats = Stats(env, workload, datacenter, scheduler)

	stats.saveStats(deployed, migrations, [], deployed, decision, schedulingTime)
	return datacenter, workload, scheduler, recovery, env, stats


# def stepSimulation(workload, scheduler, recovery, env, stats): # 更高层的控制函数，包括生成新容器、选择容器、决定容器位置、迁移容器，并输出/保存统计信息。
# #def stepSimulation(epoch, precision, recall, f1, anomaly_score, class_score, loss, aloss, tloss, workload, scheduler, recovery, env, stats):
# 	#global best_f1, best_pscore, best_rscore, best_ascore, best_cscore, best_loss, best_aloss, best_tloss  ##### 使用全局变量
# 	newcontainerinfos = workload.generateNewContainers(env.interval) # New containers info
# 	if opts.env != '': print(newcontainerinfos)
# 	deployed, destroyed = env.addContainers(newcontainerinfos) # Deploy new containers and get container IDs
# 	start = time()
# 	selected = scheduler.selection() # Select container IDs for migration
# 	decision = scheduler.filter_placement(scheduler.placement(selected+deployed)) # Decide placement for selected container ids
# 	schedulingTime = time() - start
# 	recovered_decision = recovery.run_model(stats.time_series, decision)
# 	migrations = env.simulationStep(recovered_decision) # Schedule containers
# 	workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed)) # Update workload deployed using creation IDs
#
# 	print("Deployed containers' creation IDs:", env.getCreationIDs(migrations, deployed))
# 	print("Deployed:", len(env.getCreationIDs(migrations, deployed)), "of", len(newcontainerinfos), [i[0] for i in newcontainerinfos])
# 	print("Destroyed:", len(destroyed), "of", env.getNumActiveContainers())
# 	print("Containers in host:", env.getContainersInHosts())
# 	print("Num active containers:", env.getNumActiveContainers())
# 	print("Host allocation:", [(c.getHostID() if c else -1)for c in env.containerlist])
# 	printDecisionAndMigrations(decision, migrations)
# 	print("模拟容器的创建、选择、调度、部署schedulingTime:", schedulingTime)     ########################
#
# 	stats.saveStats(deployed, migrations, destroyed, selected, decision, schedulingTime)



def stepSimulation(workload, scheduler, recovery, env, stats):
	"""更高层的控制函数，包括容器管理、调度和故障检测"""
	# 1. 生成新容器
	newcontainerinfos = workload.generateNewContainers(env.interval)
	if opts.env != '':
		print(newcontainerinfos)

	# 2. 部署/销毁容器
	deployed, destroyed = env.addContainers(newcontainerinfos)

	# 3. 容器调度决策
	start = time()
	selected = scheduler.selection()
	decision = scheduler.filter_placement(scheduler.placement(selected + deployed))
	schedulingTime = time() - start
	recovered_decision = recovery.run_model(stats.time_series, decision)

	# 4. 执行调度
	migrations = env.simulationStep(recovered_decision)
	workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))

	# 5. 故障检测（新增部分）
	try:
		print("\n正在初始化故障检测器...")
		detector = FaultDetector(num_hosts=HOSTS, interval=INTERVAL_TIME)
		print("故障检测器初始化完成，开始检测...")

		detection_result = detector.detect_faults()

		print("\n故障检测结果:")
		if not detection_result['fault_probs']:
			print("警告：未检测到任何容器故障信息")
		else:
			for cid, prob in detection_result['fault_probs']:
				print(f"容器 {cid}: 故障概率 {prob:.2%}")
			if detection_result['high_risk']:
				print(f"警告！高风险容器: {detection_result['high_risk']}")

		# 保存故障检测结果到统计系统
		if hasattr(stats, 'save_fault_data'):
			stats.save_fault_data(detection_result)
		else:
			print("警告: stats对象没有save_fault_data方法")

	except Exception as e:
		print(f"\n故障检测严重错误:")
		print(f"类型: {type(e).__name__}")
		print(f"信息: {str(e)}")
		import traceback
		traceback.print_exc()  # 打印完整堆栈跟踪

	# 6. 打印调试信息（原有部分）
	# print("Deployed containers' creation IDs:", env.getCreationIDs(migrations, deployed))
	# print("Deployed:", len(env.getCreationIDs(migrations, deployed)), "of", len(newcontainerinfos),
	# 	  [i[0] for i in newcontainerinfos])
	# print("Destroyed:", len(destroyed), "of", env.getNumActiveContainers())
	# print("Containers in host:", env.getContainersInHosts())
	# print("Num active containers:", env.getNumActiveContainers())
	# print("Host allocation:", [(c.getHostID() if c else -1) for c in env.containerlist])
	# printDecisionAndMigrations(decision, migrations)
	# print("模拟容器的创建、选择、调度、部署schedulingTime:", schedulingTime)

	# 7. 保存统计信息
	stats.saveStats(deployed, migrations, destroyed, selected, decision, schedulingTime)



def saveStats(stats, datacenter, workload, env, end=True):
	dirname = "logs/" + datacenter.__class__.__name__
	dirname += "_" + workload.__class__.__name__
	dirname += "_" + str(NUM_SIM_STEPS) 
	dirname += "_" + str(HOSTS)
	dirname += "_" + str(CONTAINERS)
	dirname += "_" + str(TOTAL_POWER)
	dirname += "_" + str(ROUTER_BW)
	dirname += "_" + str(INTERVAL_TIME)
	dirname += "_" + str(NEW_CONTAINERS)
	if not os.path.exists("logs"): os.mkdir("logs")
	if os.path.exists(dirname): shutil.rmtree(dirname, ignore_errors=True)
	os.mkdir(dirname)
	print('dirnamedirnamedirname',dirname)
	stats.generateDatasets(dirname)

	if 'Datacenter' in datacenter.__class__.__name__:
		print('nnnnnnnnnnnnnnnnnnnn')
		saved_env, saved_workload, saved_datacenter, saved_scheduler, saved_sim_scheduler = stats.env, stats.workload, stats.datacenter, stats.scheduler, stats.simulated_scheduler
		stats.env, stats.workload, stats.datacenter, stats.scheduler, stats.simulated_scheduler = None, None, None, None, None
		with open(dirname + '/' + dirname.split('/')[1] +'.pk', 'wb') as handle:
		    pickle.dump(stats, handle)
		stats.env, stats.workload, stats.datacenter, stats.scheduler, stats.simulated_scheduler = saved_env, saved_workload, saved_datacenter, saved_scheduler, saved_sim_scheduler
	if not end: return
	stats.generateGraphs(dirname)
	stats.generateCompleteDatasets(dirname)
	stats.env, stats.workload, stats.datacenter, stats.scheduler = None, None, None, None
	if 'Datacenter' in datacenter.__class__.__name__:
		stats.simulated_scheduler = None
		logger.getLogger().handlers.clear(); env.logger.getLogger().handlers.clear()
		if os.path.exists(dirname+'/'+logFile): os.remove(dirname+'/'+logFile)
		rename(logFile, dirname+'/'+logFile)
	with open(dirname + '/' + dirname.split('/')[1] +'.pk', 'wb') as handle:
	    pickle.dump(stats, handle)


if __name__ == '__main__':
	# plt.rcParams["text.usetex"] = True
	# plt.rcParams['text.latex.preamble'] = r'\usepackage{pdftex}'  ########
	# plt.rcParams["font.family"] = "Maven Pro"

	print('kaihsi1............')
	env, mode = opts.env, int(opts.mode)
	print('kaihsi222222............',opts.mode)
	if env != '':
		print('if111===============')
		# Convert all agent files to unix format
		unixify(['framework/agent/', 'framework/agent/scripts/'])

		# Start InfluxDB service
		print(color.HEADER+'InfluxDB service runs as a separate front-end window. Please minimize this window.'+color.ENDC)
		if 'Windows' in platform.system():
			os.startfile('D:/influxdb-1.8.3-1/influxd.exe')


		configFile = 'framework/config/' + opts.env + '_config.json'
		print('WindowsWindowsWindows')
		logger.basicConfig(filename=logFile, level=logger.DEBUG,
	                        format='%(asctime)s - %(levelname)s - %(message)s')
		print('WindowsWindowsWindows222222222')
		logger.debug("Creating enviornment in :{}".format(env))
		print('WindowsWindowsWindows333333')
		cfg = {}
		with open(configFile, "r") as f:
			cfg = json.load(f)
		print('WindowsWindowsWindows4444444444')
		DB_HOST = cfg['database']['ip']
		DB_PORT = cfg['database']['port']
		DB_NAME = 'COSCO'

		if env == 'Vagrant':
			print("Setting up VirtualBox environment using Vagrant")
			HOSTS_IP = setupVagrantEnvironment(configFile, mode)
			print(HOSTS_IP)
		elif env == 'VLAN':
			print("Setting up VLAN envir"
				  "onment using Ansible")
			HOSTS_IP = setupVLANEnvironment(configFile, mode)
			print('打印HOST',HOSTS_IP)
		# exit()

	datacenter, workload, scheduler, recovery, env, stats = initalizeEnvironment(env, logger)
	print('for111===============',env)
	for step in range(NUM_SIM_STEPS):
		print('for222===============',step)
		print(color.BOLD+("Simulation" if opts.env == '' else "Execution")+" Interval:", step, color.ENDC)
		stepSimulation(workload, scheduler, recovery, env, stats)
		if env != '' and step % 10 == 0: saveStats(stats, datacenter, workload, env, end = False)
	print('save111111111111')
	if opts.env != '':
		print('if222===============')
		# Destroy environment if required
		eval('destroy'+opts.env+'Environment(configFile, mode)')

		# Quit InfluxDB
		if 'Windows' in platform.system():
			print('if333===============')
			os.system('taskkill /f /im influxd.exe')
	print('save===============')



##########################################
# main.py
# import os
# import platform
# import json
# import logging as logger
# # from framework.utils import unixify, setupVagrantEnvironment, setupVLANEnvironment, initalizeEnvironment, stepSimulation, saveStats, destroyVagrantEnvironment, destroyVLANEnvironment
# # from framework.color import color
# import argparse
#
# # 解析命令行参数
# parser = argparse.ArgumentParser(description='Run simulation and analyze results.')
# # parser.add_argument('--env', type=str, required=True, help='Environment name (e.g., AWS, Openstack)')
# parser.add_argument('--env', type=str, default='', help='Environment name (e.g., AWS, Openstack)')
# parser.add_argument('--mode', type=int, default=0, help='Mode (e.g., 0 for create and destroy)')
# opts = parser.parse_args()
#
# # 日志文件路径
# logFile = 'logs/' + opts.env + '.log'
#
# if __name__ == '__main__':
#     print('Starting simulation...')
#     env, mode = opts.env, int(opts.mode)
#
#     if env != '':
#         # Convert all agent files to unix format
#         unixify(['framework/agent/', 'framework/agent/scripts/'])
#
#         # Start InfluxDB service
#         print(color.HEADER + 'InfluxDB service runs as a separate front-end window. Please minimize this window.' + color.ENDC)
#         if 'Windows' in platform.system():
#             os.startfile('D:/influxdb-1.8.3-1/influxd.exe')
#
#         # 加载配置文件
#         configFile = 'framework/config/' + opts.env + '_config.json'
#         logger.basicConfig(filename=logFile, level=logger.DEBUG,
#                            format='%(asctime)s - %(levelname)s - %(message)s')
#         logger.debug("Creating environment in: {}".format(env))
#
#         cfg = {}
#         with open(configFile, "r") as f:
#             cfg = json.load(f)
#
#         DB_HOST = cfg['database']['ip']
#         DB_PORT = cfg['database']['port']
#         DB_NAME = 'COSCO'
#
#         if env == 'Vagrant':
#             print("Setting up VirtualBox environment using Vagrant")
#             HOSTS_IP = setupVagrantEnvironment(configFile, mode)
#             print(HOSTS_IP)
#         elif env == 'VLAN':
#             print("Setting up VLAN environment using Ansible")
#             HOSTS_IP = setupVLANEnvironment(configFile, mode)
#             print('HOSTS IP:', HOSTS_IP)
#
#     # 初始化环境
#     datacenter, workload, scheduler, recovery, env, stats = initalizeEnvironment(env, logger)
#
#     # 运行仿真
#     for step in range(NUM_SIM_STEPS):
#         print(color.BOLD + ("Simulation" if opts.env == '' else "Execution") + " Interval:", step, color.ENDC)
#         stepSimulation(workload, scheduler, recovery, env, stats)
#         if env != '' and step % 10 == 0:
#             saveStats(stats, datacenter, workload, env, end=False)
#
#     # 销毁环境
#     if opts.env != '':
#         eval('destroy' + opts.env + 'Environment(configFile, mode)')
#
#         # 停止 InfluxDB 服务
#         if 'Windows' in platform.system():
#             os.system('taskkill /f /im influxd.exe')
#
#     # 调用 grapher.py 中的分析函数
#     from grapher import analyze_and_plot_results
#     analyze_and_plot_results(env)

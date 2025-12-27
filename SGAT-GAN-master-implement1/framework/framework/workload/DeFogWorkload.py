from .Workload import *
from datetime import datetime
from framework.database.Database import *
from random import gauss, choices
import random

class DFW(Workload):
    def __init__(self, num_workloads, std_dev, database):
        super().__init__()
        self.num_workloads = num_workloads
        self.std_dev = std_dev
        self.db = database
    # 这个方法通过高斯分布生成一定数量的工作负载，并为每个工作负载分配一个应用程序、SLA 和唯一的 ID。最后，这些工作负载被添加到 createdContainers 列表中，并标记为未部署。最终返回一个未部署的容器列表。
    def generateNewContainers(self, interval):
        workloadlist = []
        containers = []
        applications = ['shreshthtuli/yolo', 'shreshthtuli/pocketsphinx', 'shreshthtuli/aeneas']
        for i in range(max(1,int(gauss(self.num_workloads, self.std_dev)))):  # 高斯分布，论文里写的是泊松分布
            CreationID = self.creation_id
            SLA = np.random.randint(5,8) ## Update this based on intervals taken 服务水平协议SLA，范围5-7
            application = random.choices(applications, weights=[0.2, 0.4, 0.4])[0]
            workloadlist.append((CreationID, interval, SLA, application))
            self.creation_id += 1
        self.createdContainers += workloadlist
        self.deployedContainers += [False] * len(workloadlist)
        return self.getUndeployedContainers()
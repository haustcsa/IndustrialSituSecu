import json
import random
from enum import Enum
from ipaddress import IPv4Network
from typing import Dict, List, Type, Tuple
import inspect

from gym.utils.seeding import RandomNumberGenerator
import numpy as np
import os
from CybORG.Agents import SleepAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent
from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent

from CybORG.Shared import Scenario
from CybORG.Shared.RewardCalculator import EmptyRewardCalculator
from CybORG.Shared.BlueRewardMachine import BlueRewardMachine
from CybORG.Shared.Enums import Architecture, ProcessName, ProcessType, ProcessVersion
from CybORG.Shared.Scenario import ScenarioAgent
from CybORG.Shared.Scenarios.ScenarioGenerator import ScenarioGenerator
from CybORG.Simulator.Actions.AbstractActions import Monitor, DiscoverRemoteSystems, DiscoverNetworkServices, \
    ExploitRemoteService, PrivilegeEscalate, Impact, DegradeServices, AggressiveServiceDiscovery, \
    StealthServiceDiscovery, DiscoverDeception
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions import DecoyHarakaSMPT, DecoyApache, DecoyTomcat, DecoyVsftpd, DeployDecoy
from CybORG.Simulator.Actions.AbstractActions import Analyse, Remove, Restore
from CybORG.Simulator.Actions.Action import Sleep
from CybORG.Simulator.Actions.ConcreteActions import RedSessionCheck, Withdraw
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import AllowTrafficZone, BlockTrafficZone
from CybORG.Simulator.Actions.AbstractActions import Impact, DegradeServices, DiscoverDeception
from CybORG.Simulator.Actions.AbstractActions import DiscoverRemoteSystems, AggressiveServiceDiscovery, StealthServiceDiscovery, PrivilegeEscalate, Monitor
from CybORG.Shared.Session import RedAbstractSession, Session, VelociraptorServer
from CybORG.Simulator.Host import Host
from CybORG.Simulator.Interface import Interface
from CybORG.Simulator.Process import Process
from CybORG.Simulator.Service import Service
from CybORG.Simulator.Subnet import Subnet
from CybORG.Simulator.Actions.GreenActions import GreenAccessService, GreenLocalWork
from CybORG.Simulator.User import User

# 构建工业互联网场景

class SUBNET(str, Enum):
    """A class of class attributes that link subnet enums to the corresponding string subnet name.
    子网
    """
    ITLayer='it_layer_subnet'
    OTLayer='ot_layer_subnet'
    INTERNET = 'internet_subnet'





class ICSNetworkGenerator(ScenarioGenerator):
    """ 
    This class is used to generate scenarios designed for ICSNetwork.

    Attributes
    ----------
    background_image : str
        path to a background render image
    used_pids: List[int]
    blue_agent_class : BaseAgent
        class instance that inherits from BaseAgent to be used in scenario for blue agents
    red_agent_class : BaseAgent
        class instance that inherits from BaseAgent to be used in scenario for red agents
    green_agent_class : BaseAgent
        class instance that inherits from BaseAgent to be used in scenario for green agents
    steps : int
        number of steps that make up the episode

    The number of it layer device
    HMI_HOSTS : int
    Web_Server_HOSTS : int
    SCADA_HOSTS : int
    IT_DDECOY_HOSTS : int

    The number of it layer device
    PLC_HOSTS : int
    Industrial_ROBOT_HOSTS : int
    Sensors_HOSTS : int
    OT_DDECOY_HOSTS : int
    MAX_ADDON_SERVICES : int
        maximum number of add-on services generated in the dynamic scenario, set at 10
    MAX_BANDWIDTH : int
        maximum bandwidth of communications, set at 100
    MESSAGE_LENGTH : int
        message length of agent communications, set at 8
    """

    #IT层设备 10
    IT_SWITCH_HOST=1
    HMI_HOSTS=5
    Web_Server_HOSTS=2
    SCADA_HOSTS=2
    IT_DECOY_HOSTS=1

    #OT层设备 50
    OT_SWITCH_HOST=1
    PLC_HOSTS=10
    Industrial_ROBOT_HOSTS=15
    Sensors_HOSTS=20
    OT_DECOY_HOSTS=5

    MAX_HOSTS=65  #it+ot intetnet子网有1+1+1
    MAX_ADDON_SERVICES = 10
    #通信参数
    MAX_BANDWIDTH = 100
    MESSAGE_LENGTH = 8

    def __init__(
            self,
            blue_agent_class: Type[BaseAgent] = None,
            red_agent_class: Type[BaseAgent] = None,
            green_agent_class: Type[BaseAgent] = None,
            steps: int = 100
    ):
        """
        Parameters
        ----------
        blue_agent_class : BaseAgent, optional
            The type of agent for blue agents, by default None
        red_agent_class : BaseAgent, optional
            The type of agent for red agents, by default None
        green_agent_class : BaseAgent, optional
            The type of agent for green agents, by default None
        steps : int, optional
            The number of steps, by default 100
        """

        super().__init__()
        self.background_image = "img/blank.png"
        self.used_pids: List[int] = []
        self.blue_agent_class = blue_agent_class
        self.red_agent_class = red_agent_class
        self.green_agent_class = green_agent_class
        self.steps = steps
        file_path = os.path.abspath('F:/cage-challenge-4-main/CybORG/Simulator/Scenarios/cve_database.json')
        with open(file_path,'r',encoding='utf8') as f:
            self.cve_db=json.load(f)

    def _assign_cves(self,hostname:str, subnet: Subnet)-> List[dict]:
        layer = "IT" if "it_layer" in subnet.name.lower() else "OT"  # 从子网提取层级
        node_type=None
        if 'hmi' in hostname:
            node_type="HMI"
        elif 'web_server' in hostname:
            node_type="Web_Server"
        elif 'scada' in hostname:
            node_type="SCADA"
        elif 'it_layer_subnet_decoy' in hostname:
            node_type="IT_DECOY"
        elif 'plc' in hostname:
            node_type="PLC"
        elif 'robot' in hostname:
            node_type="Industrial_ROBOT"
        elif 'sensor' in hostname:
            node_type="Sensors"
        elif 'ot_layer_subnet_decoy' in hostname:
            node_type="OT_DECOY"
        #cve
        # print(f"[DEBUG] _assign_cves: hostname={hostname}, node_type={node_type}, layer={layer}")
        for cveinfo in self.cve_db:
            if cveinfo['node_type']==node_type and cveinfo['layer']== layer:
                return cveinfo['cves']
        return []

    def create_scenario(self, np_random: RandomNumberGenerator) -> Scenario:
        """
        This public function initiates the generation of a new Enterprise Scenario.

        This function calls a multitude of private functions to generate:

        - subnets
        - hosts
        - agents (red, green, blue)
        - mission phases
        - reward machines

        Finally, the outputs from all the private functions in this class are used to create an instance of the Scenario object - which is returned.

        Parameters
        ----------
        np_random : RandomNumberGenerator
            The RNG that will be used to make "random" decisions when creating scenarios.

        Returns
        -------
        scenario : Scenario
            The new icsnetwork scenario object
        """
        self.used_pids.clear()
        self.np_random = np_random
        subnets = self._generate_subnets()
        hosts = self._generate_hosts(subnets)
        agents: Dict[str, ScenarioAgent] = {}
        self._generate_blue_agents(subnets, agents)
        self._generate_green_agents(hosts, subnets, agents)
        self._generate_red_agents(subnets, agents)
        team_agents = self._generate_team_agents(agents)
        scenario = Scenario(
            agents=agents,
            team_calcs=None,
            team_agents=team_agents,
            hosts=hosts,
            subnets=subnets,
            mission_phases=self._generate_mission_phases(self.steps),
            allowed_subnets_per_mphase=self._set_allowed_subnets_per_mission_phase(),
            predeployed=False,
            max_bandwidth=self.MAX_BANDWIDTH
        )
        scenario.team_calc = self._generate_team_calcs()

        return scenario

    def _generate_subnets(self) -> Dict[str, Subnet]:
        """
        This function generates the specific subnets required by icsnetwork the scenario.

        Returns
        -------
        scenario_subnets : Dict[str, Subnet]
            A dictionary where the keys are the subnet names, and the values are the subnets
            themselves.
        """
        subnet_prefix = 24
        network = IPv4Network("192.168.0.0/16")
        network_subnets = list(network.subnets(new_prefix=subnet_prefix))

        # declare subnet NACLs 网络访问控制列表，Network Access Control List）
        subnet_nacls = {
            SUBNET.ITLayer: {
                SUBNET.OTLayer: {"in": "all", "out": "all"}, # 允许IT层与OT层之间的流量
                SUBNET.INTERNET: {"in": "all", "out": "all"} #允许IT层与互联网的流量
            },
            SUBNET.OTLayer: {
                SUBNET.ITLayer: {"in": "all", "out": "all"} #允许OT层与IT层之间的流量
            },
            SUBNET.INTERNET: {
                # SUBNET.OTLayer: {"in": "all", "out": "all"},
                SUBNET.ITLayer: {"in": "all", "out": "all"} # 互联网只与IT层通信
            }
        }
        # Create subnets in a list that can be iterated over 在列表中创建可迭代的子网
        scenario_subnets = {}
        for subnet_name in SUBNET:
            nacl = subnet_nacls[subnet_name]
            subnet = self._generate_subnet(subnet_name.value, nacl, network_subnets)
            scenario_subnets[subnet_name] = subnet
        return scenario_subnets

    def _generate_subnet(self, subnet_name: str, nacls: Dict[str, Dict[str, str]],ipv4_subnets: List[IPv4Network]) -> Subnet:
        """
        This function generates a new Scenario subnet. It has placeholder values for 'size' and
        'hosts' as we haven't generated the hosts yet.

        Parameters
        ----------
        subnet_name : str
            the label of the subnet
        nacls : Dict[str, Dict[str, str]]
            A dictionary where the keys are the other subnets the subnet being generated interacts
            with, and the values are another dictionary that specifies how information can flow.
        ipv4_subnets : List[IPv4Network]
            A list containing the remaining available IPv4 subnets.

        Returns
        -------
        Subnet
            A new Subnet object
        """
        selected_subnet_index = self.np_random.choice(len(ipv4_subnets))
        cidr = ipv4_subnets.pop(selected_subnet_index)
        size = len(list(cidr.hosts()))
        return Subnet(subnet_name, size, [], nacls, cidr, [])

    def _set_allowed_subnets_per_mission_phase(self) -> Dict[SUBNET, tuple]:
        """This static function returns the allowed_subnets according to readme for CC4.

        # (0) Pre-planning phase
        # (1) Mission A
        # (2) Mission B

        Returns
        -------
        comms_policy : Array[Array[Tuple(Subnet, Subnet)]]
            A list of pairs of subnets that are allowed to communicate with each other during the policy iteration
        """

        policy_1 = [
            (SUBNET.ITLayer, SUBNET.OTLayer),  # IT子网与OT子网之间通信
            (SUBNET.ITLayer, SUBNET.INTERNET),  # IT子网与互联网之间通信
        ]
        comms_policy = [policy_1]

        return comms_policy

    def _generate_hosts(self, subnets: Dict[str, Subnet]) -> Dict[str, Host]:
        """
        This function initiates the creation of all the hosts in the scenario.
        Since the hosts are tried to the subnets, the scenario's subnets are required as a parameter.
        生成主机
        Parameters
        ----------
        subnets : Dict[str, Subnet]
            A dictionary where the keys are the names of the subnets, and the values are the
            subnets themselves.

        Returns
        -------
        Dict[str, Host]
            A dictionary where the keys are the hostname, and the values are the hosts themselves.
        """
        host_list = []


        #遍历子网
        for subnet in subnets.values():
            ip_addresses = list(subnet.cidr.hosts())

            if subnet.name == "internet_subnet":
                hostname = "root_internet_host_0"
                subnet.hosts.append(hostname)
                selected_ip_address_index = self.np_random.choice(len(ip_addresses))
                ip_address = ip_addresses.pop(selected_ip_address_index)
                subnet.ip_addresses.append(ip_address)
                host_list.append(self._generate_linux_host(hostname, ip_address, subnet))

                # 生成一个路由器
                ip_addresses = list(subnet.cidr.hosts())
                hostname2 = f'internet_subnet_router'
                subnet.hosts.append(hostname2)
                selected_ip_address_index = self.np_random.choice(len(ip_addresses))
                ip_address = ip_addresses.pop(selected_ip_address_index)
                subnet.ip_addresses.append(ip_address)
                host_list.append(self._generate_linux_host(hostname2, ip_address, subnet))

                # 生成一个sdn控制器
                ip_addresses = list(subnet.cidr.hosts())
                hostname3 = f'internet_subnet_sdn_controller'
                selected_ip_address_index = self.np_random.choice(len(ip_addresses))
                ip_address = ip_addresses.pop(selected_ip_address_index)
                subnet.ip_addresses.append(ip_address)
                host_list.append(self._generate_linux_host(hostname3, ip_address, subnet))
                subnet.hosts.append(hostname3)


                subnet.size = 3 #设置子网的大小为 1，表示这个子网只有一个主机（即互联网主机）。
                continue


            if subnet.name == "it_layer_subnet":
                # IT子网

                # 为每个子网生成一个SDN交换机主机
                hostname = f'{subnet.name}_sdn_switch'
                subnet.hosts.append(hostname)
                selected_ip_address_index = self.np_random.choice(len(ip_addresses))
                ip_address = ip_addresses.pop(selected_ip_address_index)
                subnet.ip_addresses.append(ip_address)
                host_list.append(self._generate_linux_host(hostname, ip_address, subnet))

                # 生成HMI_HOSTS=6
                num_hmi_hosts = self.HMI_HOSTS
                for i in range(num_hmi_hosts):
                    hostname = f"{subnet.name}_hmi_host_{i}"
                    subnet.hosts.append(hostname)
                    selected_ip_address_index = self.np_random.choice(len(ip_addresses))
                    ip_address = ip_addresses.pop(selected_ip_address_index)
                    subnet.ip_addresses.append(ip_address)
                    host_list.append(self._generate_linux_host(hostname, ip_address, subnet))

                # 生成Web_Server_HOSTS=2
                num_web_server_hosts = self.Web_Server_HOSTS
                for i in range(num_web_server_hosts):
                    hostname = f"{subnet.name}_web_server_host_{i}"
                    ip_address = ip_addresses.pop()
                    subnet.ip_addresses.append(ip_address)
                    host_list.append(self._generate_linux_host(hostname, ip_address, subnet))
                    subnet.hosts.append(hostname)

                # 生成SCADA_HOSTS=2
                num_sdaca_hosts = self.SCADA_HOSTS
                for i in range(num_sdaca_hosts):
                    hostname = f"{subnet.name}_scada_host_{i}"
                    ip_address = ip_addresses.pop()
                    subnet.ip_addresses.append(ip_address)
                    host_list.append(self._generate_linux_host(hostname, ip_address, subnet))
                    subnet.hosts.append(hostname)

                # 生成IT_DECOY_HOSTS=1
                num_it_decoy_hosts = self.IT_DECOY_HOSTS
                for i in range(num_it_decoy_hosts):
                    hostname = f"{subnet.name}_decoy_host_{i}"
                    ip_address = ip_addresses.pop()
                    subnet.ip_addresses.append(ip_address)
                    host_list.append(self._generate_linux_host(hostname, ip_address, subnet))
                    subnet.hosts.append(hostname)

                subnet.size = num_hmi_hosts + num_web_server_hosts + num_sdaca_hosts + num_it_decoy_hosts + 1
                continue

            if subnet.name == "ot_layer_subnet":
                # OT子网
                # 为每个子网生成一个SDN交换机主机
                hostname = f'{subnet.name}_sdn_switch'
                subnet.hosts.append(hostname)
                selected_ip_address_index = self.np_random.choice(len(ip_addresses))
                ip_address = ip_addresses.pop(selected_ip_address_index)
                subnet.ip_addresses.append(ip_address)
                host_list.append(self._generate_linux_host(hostname, ip_address, subnet))

                # 生成PLC_HOSTS=15
                num_plc_hosts = self.PLC_HOSTS
                for i in range(num_plc_hosts):
                    hostname = f"{subnet.name}_plc_host_{i}"
                    subnet.hosts.append(hostname)
                    selected_ip_address_index = self.np_random.choice(len(ip_addresses))
                    ip_address = ip_addresses.pop(selected_ip_address_index)
                    subnet.ip_addresses.append(ip_address)
                    host_list.append(self._generate_linux_host(hostname, ip_address, subnet))

                # 生成Industrial_ROBOT_HOSTS=25
                num_industrial_robot_hosts = self.Industrial_ROBOT_HOSTS
                for i in range(num_industrial_robot_hosts):
                    hostname = f"{subnet.name}_industrial_robot_host_{i}"
                    ip_address = ip_addresses.pop()
                    subnet.ip_addresses.append(ip_address)
                    host_list.append(self._generate_linux_host(hostname, ip_address, subnet))
                    subnet.hosts.append(hostname)

                # 生成Sensors_HOSTS=60
                num_sensors_hosts = self.Sensors_HOSTS
                for i in range(num_sensors_hosts):
                    hostname = f"{subnet.name}_sensor_host_{i}"
                    ip_address = ip_addresses.pop()
                    subnet.ip_addresses.append(ip_address)
                    host_list.append(self._generate_linux_host(hostname, ip_address, subnet))
                    subnet.hosts.append(hostname)

                # 生成OT_DECOY_HOSTS=5
                num_ot_decoy_hosts = self.OT_DECOY_HOSTS
                for i in range(num_ot_decoy_hosts):
                    hostname = f"{subnet.name}_decoy_host_{i}"
                    ip_address = ip_addresses.pop()
                    subnet.ip_addresses.append(ip_address)
                    host_list.append(self._generate_linux_host(hostname, ip_address, subnet))
                    subnet.hosts.append(hostname)

                subnet.size = num_plc_hosts + num_industrial_robot_hosts + num_sensors_hosts + num_ot_decoy_hosts + 1

                continue

        # Convert list into a dictionary and return it
        return {host.hostname: host for host in host_list}

    def _generate_data_links(self, hostname: str, subnet):
        """_summary_
                Parameters
                ----------
                hostname : str
                    The name of the host whose parent is to be defined.
                subnet : _type_
                    The subnet that host belongs to.

                Returns
                -------
                List[str]
                    The parent data link
                """
        if hostname == "root_internet_host_0":
            data_links = [
                "internet_subnet_router",
            ]
        elif hostname == "internet_subnet_router":
            data_links = [
                "root_internet_host_0",
                "internet_subnet_sdn_controller",
                "it_layer_subnet_sdn_switch",
                "ot_layer_subnet_sdn_switch",
                          ]
        elif hostname == "internet_subnet_sdn_controller":
            data_links = [
                "internet_subnet_router"
                          ]
        elif "_switch" in hostname:
            if 'it' or 'ot' in hostname:
                data_links = ["internet_subnet_router"]
            else:
                raise ValueError(f"Unexpected switch {hostname} in subnet {subnet}")
        else:
            data_links = [f"{subnet.name}_sdn_switch"]
        return data_links

    def _between_subnet_links(self, hostname: str):
        """Additional info about other hosts that red gains when it get root controll of the host.

        Parameters
        ----------
        hostname : str
            the name of the host.

        Returns
        -------
        links : Dict[str, List[str]]
            hosts that have (directional) links to eachother
        """

        num1=random.randint(0,4)
        num2=random.randint(5,9)
        num3=random.randint(10,14)
        num4=random.randint(0,4)
        num5=random.randint(5,9)
        num6=random.randint(10,14)
        num7=random.randint(15,19)
        links = {
            "root_internet_host_0": [
                "it_layer_subnet_web_server_host_0",
                "it_layer_subnet_web_server_host_1",
                "it_layer_subnet_decoy_host_0",
                ],
            "it_layer_subnet_web_server_host_0": [
                "root_internet_host_0"
            ],
            "it_layer_subnet_web_server_host_1": [
                "root_internet_host_0"
            ],
            "it_layer_subnet_sdaca_host_0": [
                f"ot_layer_subnet_plc_host_{num1}",
                f"ot_layer_subnet_plc_host_{num2}",
                f"ot_layer_subnet_industrial_robot_host_{num1}",
                f"ot_layer_subnet_industrial_robot_host_{num2}",
                f"ot_layer_subnet_industrial_robot_host_{num3}",
                f"ot_layer_subnet_sensor_host_{num1}",
                f"ot_layer_subnet_sensor_host_{num2}",
                f"ot_layer_subnet_sensor_host_{num3}",
                f"ot_layer_subnet_sensor_host_{num7}",
            ],
            "it_layer_subnet_sdaca_host_1": [
                f"ot_layer_subnet_plc_host_{num2}",
                f"ot_layer_subnet_plc_host_{num5}",
                f"ot_layer_subnet_industrial_robot_host_{num4}",
                f"ot_layer_subnet_industrial_robot_host_{num5}",
                f"ot_layer_subnet_industrial_robot_host_{num6}",
                f"ot_layer_subnet_sensor_host_{num4}",
                f"ot_layer_subnet_sensor_host_{num5}",
                f"ot_layer_subnet_sensor_host_{num6}",
                f"ot_layer_subnet_sensor_host_{num7}",
            ],
            "it_layer_subnet_sdaca_host_2": [
                f"ot_layer_subnet_plc_host_{num1}",
                f"ot_layer_subnet_plc_host_{num4}",
                f"ot_layer_subnet_industrial_robot_host_{num1}",
                f"ot_layer_subnet_industrial_robot_host_{num5}",
                f"ot_layer_subnet_industrial_robot_host_{num3}",
                f"ot_layer_subnet_sensor_host_{num1}",
                f"ot_layer_subnet_sensor_host_{num5}",
                f"ot_layer_subnet_sensor_host_{num3}",
                f"ot_layer_subnet_sensor_host_{num7}",
            ],
            f"ot_layer_subnet_plc_host_{num1}": [
                "it_layer_subnet_sdaca_host_0",
                "it_layer_subnet_sdaca_host_2"
            ],
            f"ot_layer_subnet_plc_host_{num2}": [
                "it_layer_subnet_sdaca_host_0",
                "it_layer_subnet_sdaca_host_1"
            ],
            f"ot_layer_subnet_plc_host_{num4}": [
                "it_layer_subnet_sdaca_host_2",
            ],
            f"ot_layer_subnet_plc_host_{num5}": [
                "it_layer_subnet_sdaca_host_1",
            ],
            f"ot_layer_subnet_industrial_robot_host_{num1}": [
                "it_layer_subnet_sdaca_host_0",
                "it_layer_subnet_sdaca_host_2"
            ],
            f"ot_layer_subnet_industrial_robot_host_{num2}": [
                "it_layer_subnet_sdaca_host_0",
            ],
            f"ot_layer_subnet_industrial_robot_host_{num3}": [
                "it_layer_subnet_sdaca_host_0",
                "it_layer_subnet_sdaca_host_2"
            ],
            f"ot_layer_subnet_industrial_robot_host_{num4}": [
                "it_layer_subnet_sdaca_host_1",
            ],
            f"ot_layer_subnet_industrial_robot_host_{num5}": [
                "it_layer_subnet_sdaca_host_1",
                "it_layer_subnet_sdaca_host_2"
            ],
            f"ot_layer_subnet_industrial_robot_host_{num6}": [
                "it_layer_subnet_sdaca_host_1",
            ],
            f"ot_layer_subnet_sensor_host_{num1}": [
                "it_layer_subnet_sdaca_host_0",
                "it_layer_subnet_sdaca_host_2"
            ],
            f"ot_layer_subnet_sensor_host_{num2}": [
                "it_layer_subnet_sdaca_host_0",
            ],
            f"ot_layer_subnet_sensor_host_{num3}": [
                "it_layer_subnet_sdaca_host_0",
                "it_layer_subnet_sdaca_host_2"
            ],
            f"ot_layer_subnet_sensor_host_{num4}": [
                "it_layer_subnet_sdaca_host_1",
            ],
            f"ot_layer_subnet_sensor_host_{num5}": [
                "it_layer_subnet_sdaca_host_1",
                "it_layer_subnet_sdaca_host_2"
            ],
            f"ot_layer_subnet_sensor_host_{num6}": [
                "it_layer_subnet_sdaca_host_1",
            ],
            f"ot_layer_subnet_sensor_host_{num7}": [
                "it_layer_subnet_sdaca_host_0",
                "it_layer_subnet_sdaca_host_1",
                "it_layer_subnet_sdaca_host_2"
            ]
        }
        if not hostname in links:
            return None
        info = {}
        for host in links[hostname]:
            info[host] = {'Interfaces': 'ip_address'}
        return info

    def _generate_linux_host(self, hostname: str, ip_address: IPv4Network, subnet: Subnet) -> Host:

        cves=self._assign_cves(hostname,subnet)
        # 添加调试输出：打印分配的CVE信息
        # print(f"\n[DEBUG] 主机 {hostname} 的CVE信息: {cves}\n")

        linux_distro_options = [
            { "OSDistribution": "UBUNTU", "OSVersion": "22.04.2 LTS" },
            { "OSDistribution": "KALI", "OSVersion": "K2019_4" }
        ]
        system_info = { 'OSType': "LINUX", "Architecture": Architecture.x64,'CVE Info':cves }
        OSDistribution = self.np_random.choice(linux_distro_options)
        system_info.update(OSDistribution)
        interfaces = [Interface(
            name='eth0',
            ip_address=ip_address,
            subnet=subnet.cidr,
            interface_type='wired',
            data_links=self._generate_data_links(hostname, subnet),
            swarm=False
        )]
        root_user = User(groups=[{'GID': 0, 'Group Name': 'root'}], uid=0, username='root')
        user_group = {'GID': 1, 'Group Name': 'user'}
        user = User(groups=[user_group], uid=1000, username='user', bruteforceable=True)

        services = None
        processes = None
        respond_to_ping = True

        return Host(
            hostname=hostname,
            cve_info=cves,
            host_type="",
            processes=processes,
            system_info=system_info,
            interfaces=interfaces,
            info=self._between_subnet_links(hostname),
            users=[root_user, user],
            services=services,
            respond_to_ping=respond_to_ping,
            np_random=self.np_random,
        )

    def _generate_linux_host_services(self, hostname: str) -> Dict[str, Service]:
        """
        This function generates a dict of random services for a linux host.

        Parameters
        ----------
        hostname : str
            The name of the host to have services generated.

        Returns
        -------
        Dict[str, dict]
            A dictionary where the keys are the service names, and the values are the dictionaries
            containing the services themselves.
        """
        # Set up the mandatory services.
        services = { ProcessName.SSHD: Service(process=self._generate_pid()) }

        services[ProcessName.OTSERVICE] = Service(process=self._generate_pid())

        # Define what the options are for additional services
        addon_services_options = {
            ProcessName.APACHE2: Service(process=self._generate_pid()),
            ProcessName.MYSQLD: Service(process=self._generate_pid()),
            ProcessName.SMTP: Service(process=self._generate_pid()),
        }
        # Choose a random number of the optional services
        max_addon_services = min(len(addon_services_options), self.MAX_ADDON_SERVICES)
        num_addon_services = self.np_random.integers(0, max_addon_services, endpoint=True)
        for _ in range(num_addon_services):
            choice = self.np_random.choice(list(addon_services_options.keys()))
            services[choice] = addon_services_options.pop(choice)
        return services

    def _generate_pid(self) -> int:
        """
        Generates a dummy process ID number that is not already contained within the list of used
        process IDs.

        Returns
        -------
        int
            The new process ID.
        """
        while True:
            pid = self.np_random.integers(1000, 10000)  # generate a random 4-digit number
            if pid not in self.used_pids:  # check if the generated PID is not in the used_pids list
                self.used_pids.append(pid)
                return pid  # if not, return the generated PID

    def _generate_linux_host_processes(self, services: Dict[str, Service]) -> list[Process]:
        """
        Creates a set of randomised processes for a linux host based on its services.

        Parameters
        ----------
        services : dict
            A dict containing the services that were made for the host.

        Returns
        -------
        List[dict]
            A list containing dicts that represent the processes for the linux host.
        """
        processes = []
        prob_vuln_proc_occurs = 1.0

        local_processes = {
            ProcessName.SSHD: {'port': 22, 'type': ProcessType.SSH},
            ProcessName.APACHE2: {'port': 80, 'type': ProcessType.WEBSERVER},
            ProcessName.MYSQLD: {'port': 3390, 'type': ProcessType.MYSQL},
            ProcessName.SMTP: {'port': 25, 'type': ProcessType.SMTP},
            ProcessName.OTSERVICE: {'port': 1, 'type': ProcessType.UNKNOWN},
            "FTP": {'port': 21, 'type': ProcessType.FEMITTER}
        }

        for key, service in services.items():
            process = Process(
                process_name=key,
                pid=service.process,
                path='/ usr / sbin',
                username="user",
                open_ports=[{
                    "local_address": "0.0.0.0",
                    "local_port": local_processes[key]['port'],
                }],
                process_type=local_processes[key]['type']
            )

            if local_processes[key]['type'] == ProcessType.SMTP:
                process.version = ProcessVersion.HARAKA_2_8_9

            if prob_vuln_proc_occurs < self.np_random.random():
                if local_processes[key]['type'] == ProcessType.WEBSERVER:
                    process.properties = ['rfi']
                if local_processes[key]['type'] == ProcessType.SMTP:
                    process.version = ProcessVersion.HARAKA_2_7_0

            processes.append(process)
        return processes


    def _generate_blue_agents(self, subnets: Dict[str, Subnet], agents: Dict[str, ScenarioAgent]):
        """生成三个蓝色代理，分别位于IT、OT、Internet子网，其中Internet代理仅绑定SDN控制器

        ['blue_agent_it_0',
         'blue_agent_ot_1',
         'blue_agent_internet_2',
         'red_agent_0',
         'red_agent_1']

        """
        blue_actions = [AllowTrafficZone, BlockTrafficZone, Monitor, Analyse, Restore, Remove, DeployDecoy, Sleep]
        blue_agent_allowed_subnets = [
            [SUBNET.ITLayer.value],  # IT层子网
            [SUBNET.OTLayer.value],  # OT层子网
            [SUBNET.INTERNET.value],  # Internet子网
        ]

        for idx, allowed_subnet_group in enumerate(blue_agent_allowed_subnets):
            # 获取子网对象
            subnet_name = allowed_subnet_group[0]
            subnet = subnets[subnet_name]

            # 生成代理名称（唯一标识）
            agent_name = f"blue_agent_{idx}"  # 例如 blue_agent_IT_0

            # 确定允许的主机列表
            if subnet_name == SUBNET.INTERNET.value:
                # Internet子网代理仅允许在SDN控制器上
                allowed_hosts = ['internet_subnet_sdn_controller']
            else:
                allowed_hosts = subnet.hosts  # IT/OT子网允许所有主机

            # 初始化会话和OSINT信息
            sessions: List[Session] = []
            osint = {"Hosts": {}}

            # 配置OSINT（代理可见的主机信息）
            for host in allowed_hosts:
                osint["Hosts"][host] = {
                    'Interfaces': 'All',
                    'System info': 'All',
                    'User info': 'All'
                }

            # 创建父会话（VelociraptorServer）
            parent_host = allowed_hosts[0]  # 第一个允许的主机作为父会话主机
            parent_session = VelociraptorServer(
                name=f"blue_session_{agent_name}_parent",
                username="ubuntu",
                session_type="VelociraptorServer",
                hostname=parent_host,
                pid=None,
                ident=None,
                agent=agent_name
            )
            sessions.append(parent_session)

            # 创建子会话（普通Session）
            for host in allowed_hosts:
                if host == parent_host:
                    continue  # 跳过父会话主机
                session = Session(
                    name=f"blue_session_{agent_name}_{host}",
                    username="ubuntu",
                    session_type="blue_session",
                    hostname=host,
                    pid=None,
                    ident=None,
                    agent=agent_name,
                    parent=parent_session.name  # 设置父会话
                )
                sessions.append(session)

            # 设置父会话的子会话数量
            parent_session.num_children = len(sessions) - 1

            # 确定代理类型（默认使用SleepAgent）
            agent_type = self.blue_agent_class(agent_name) if self.blue_agent_class else SleepAgent(agent_name)

            # 定义默认动作（监控）
            default_actions = (Monitor, {'session': 0, 'agent': agent_name})

            # 将代理添加到agents字典
            agents[agent_name] = ScenarioAgent(
                agent_name=agent_name,
                team="Blue",
                starting_sessions=sessions,
                actions=blue_actions,
                osint=osint,
                allowed_subnets=allowed_subnet_group,
                agent_type=agent_type,
                active=True,
                default_actions=default_actions
            )

    def _generate_green_agents(self, hosts: Dict[str, Host], subnets: Dict[str, Subnet], agents: Dict[str, ScenarioAgent]):
        """
        Populates the agents dict with green agents. There is a green agents for every host in the
        scenario.

        Parameters
        ----------
        hosts : Dict[str, Host]
            A dict containing all of the hosts of the scenario.
        subnets : Dict[str, Subnet]
            A dict containing all the subnets of the scenario.
        agents : Dict[str, ScenarioAgent]
            A dict containing all of the agents of the scenario (so far.)
        """
        green_actions = [GreenAccessService, GreenLocalWork, Sleep]
        green_agent_count = 0
        for subnet in subnets.values():
            for hostname in subnet.hosts:
                if "user" not in hostname: continue
                # Set-up OSINT based on the subnet the starting host is in.
                osint = {"Hosts": {}}
                for host in subnet.hosts:
                    osint["Hosts"][host] = {
                        'Interfaces': 'All', 'System info': 'All', 'User info': 'All'
                    }
                agent_name = f"green_agent_{green_agent_count}"
                green_agent_count += 1
                session = Session(
                    name=f"green_session_{green_agent_count}",
                    username="ubuntu",
                    session_type="green_session",
                    hostname=hostname,
                    pid=None,
                    ident=None,
                    agent=None
                )
                agent_type = None
                default_actions = (Sleep, {})
                if self.green_agent_class:
                    if self.green_agent_class == EnterpriseGreenAgent:
                        host_ip = hosts[hostname].interfaces[0].ip_address
                        agent_type = self.green_agent_class(name=agent_name, np_random=self.np_random, own_ip=host_ip)
                    elif self.green_agent_class == SleepAgent:
                        green_actions = [Sleep]
                    else:
                        agent_type = self.green_agent_class(agent_name)
                agents[agent_name] = ScenarioAgent(
                    agent_name, "Green", [session], green_actions, osint, [subnet.name], agent_type, True,
                    default_actions
                )


    def _generate_red_agents(self, subnets: Dict[str, Subnet], agents: Dict[str, ScenarioAgent]):
        """生成红色代理，分布在IT、OT、互联网子网，仅互联网代理初始激活"""
        red_actions = [
            DiscoverRemoteSystems, AggressiveServiceDiscovery, StealthServiceDiscovery,
            ExploitRemoteService, PrivilegeEscalate, DegradeServices, DiscoverDeception,
            Impact, Withdraw, Sleep
        ]

        # 允许的子网：IT层、OT层、互联网
        red_agent_allowed_subnets = [
            [SUBNET.ITLayer.value],
            [SUBNET.OTLayer.value],
            [SUBNET.INTERNET.value]
        ]

        for allowed_subnet_group in red_agent_allowed_subnets:
            # 获取子网名称（如 "it_layer_subnet"）
            subnet_name = allowed_subnet_group[0]
            subnet = subnets[subnet_name]

            # 生成代理名称（如 "red_agent_internet_subnet_0"）
            agent_name = f"red_agent_{len(agents)}"

            # 确定允许的起始主机（排除路由器）
            allowed_starting_hosts = [h for h in subnet.hosts if 'router' not in h]

            # 随机选择起始主机
            starting_host = self.np_random.choice(allowed_starting_hosts)

            # 初始化OSINT信息（仅可见起始主机）
            osint = {"Hosts": {starting_host: {'Interfaces': 'All', 'System info': 'All', 'User info': 'All'}}}

            # 初始化会话列表
            sess_list = []
            active = False

            # 仅互联网子网的代理初始激活
            if subnet_name == SUBNET.INTERNET.value:
                # 使用RedAbstractSession作为攻击入口
                session = RedAbstractSession(
                    name=f"red_session_{agent_name}",
                    username="root",  # 假设已获得root权限
                    session_type="RedAbstractSession",
                    hostname=starting_host,
                    pid=self._generate_pid(),
                    ident=None,
                    agent=agent_name
                )
                sess_list.append(session)
                active = True
            else:
                # 其他子网代理初始非激活
                session = Session(
                    name=f"red_session_{agent_name}",
                    username="user",
                    session_type="red_session",
                    hostname=starting_host,
                    pid=self._generate_pid(),
                    ident=None,
                    agent=agent_name
                )
                sess_list.append(session)

            # 确定代理类型（互联网代理使用主动攻击策略）
            if subnet_name == SUBNET.INTERNET.value:
                agent_type = FiniteStateRedAgent(
                    name=agent_name,
                    np_random=self.np_random,
                    agent_subnets=[subnet.cidr]
                )
            else:
                agent_type = SleepAgent(agent_name)  # 其他代理初始休眠

            # 定义默认动作（红队检查会话）
            default_actions = (RedSessionCheck, {'session': 0, 'agent': agent_name})

            # 将代理添加到字典
            agents[agent_name] = ScenarioAgent(
                agent_name=agent_name,
                team="Red",
                starting_sessions=sess_list,
                actions=red_actions,
                osint=osint,
                allowed_subnets=allowed_subnet_group,
                agent_type=agent_type,
                active=active,  # 只有互联网代理初始激活
                default_actions=default_actions
            )


    def _generate_team_calcs(self) -> dict:
        """
        Returns
        -------
        team_calcs : Dict[str, Dict[str, BlueRewardMachine]]
            A dictionary of reward calculator instances for each agent type
        """
        team_calcs = {
            "Blue": { 'BlueRewardMachine': BlueRewardMachine("Blue") },
            "Red": { 'None': EmptyRewardCalculator("Red") },
            "Green": { 'None': EmptyRewardCalculator("Green") }
        }
        return team_calcs

    def _generate_team_agents(self, agents: Dict[str, ScenarioAgent]) -> Dict[str, List[str]]:
        """
        Creates a dict where the keys are the different teams, and the values are lists of the
        names of agents that belong to those teams.

        Parameters
        ----------
        agents : Dict[str, ScenarioAgent]
            A dict that contains all the agents of the scenario.

        Returns
        -------
        Dict[str, List[str]]
            _description_
        """
        team_agents = {}
        for team in ["Blue", "Red", "Green"]:
            team_agents[team] = [agent for agent in agents.keys() if team.lower() in agent]
        return team_agents


    def _generate_mission_phases(self, steps) -> Tuple[int, int, int]:
        quotient, remainder = divmod(steps, 3)
        if remainder == 2:
           return (quotient+1, quotient+1, quotient)
        if remainder == 1:
            return (quotient+1, quotient, quotient)
        return (quotient, quotient, quotient)

    def determine_done(self, env_controller) -> bool:
        """ Determines when the episode ends

        Returns
        -------
        Boolean
            T/F value for if episode is to end
        """
        return env_controller.step_count >= (self.steps-1)

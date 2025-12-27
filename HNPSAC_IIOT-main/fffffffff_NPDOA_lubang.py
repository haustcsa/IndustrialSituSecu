import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import time
from scipy.spatial import ConvexHull

# 固定随机种子
np.random.seed(21)
torch.manual_seed(21)

# ========================= 基础配置 =========================
n_servers = 20
n_devices = 250
network_speed = np.random.uniform(100, 900, n_devices)
server_cost = np.random.uniform(0.01, 0.1, n_servers)
server_speed = np.random.uniform(50, 200, n_servers)
server_ram = np.random.uniform(2, 8, n_devices)
data_size = np.random.uniform(50, 150, n_devices)
completion_requirement = np.random.uniform(20, 40, n_devices)
ram_requirement = np.random.uniform(1, 2, n_devices)
# m, n = 1e-2, 10
# m, n = 1, 1
m, n = 10, 1e-2


def generate_data():
    return np.random.uniform(1e6, 1e8, n_devices)


def calculate_total_cost(pos, weight_m=m, weight_n=n):
    total_cost = 0
    total_latency = 0
    penalty = 1000

    for j in range(n_devices):
        s = int(np.clip(pos[j], 0, n_servers - 1))
        if ram_requirement[j] <= server_ram[s]:
            trans = data_size[j] / network_speed[j]
            proc = completion_requirement[j] / server_speed[s]
            t = trans + proc
            cost = server_cost[s] * t
            total_cost += cost
            total_latency += t
        else:
            total_cost += penalty
            total_latency += penalty

    return weight_m * total_cost + weight_n * total_latency


def calculate_raw_values(pos):
    total_cost = 0
    total_latency = 0
    penalty = 1000

    for j in range(n_devices):
        s = int(np.clip(pos[j], 0, n_servers - 1))
        if ram_requirement[j] <= server_ram[s]:
            trans = data_size[j] / network_speed[j]
            proc = completion_requirement[j] / server_speed[s]
            t = trans + proc
            cost = server_cost[s] * t
            total_cost += cost
            total_latency += t
        else:
            total_cost += penalty
            total_latency += penalty

    return total_cost, total_latency


# ========================= 改进的NPDOA实现 =========================
class AdaptiveNPDOA_Optimizer:
    def __init__(self, n_population, dim, lb, ub, max_fe, a=0.3, l=1.2, d=0.8):
        self.n_population = n_population
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.max_fe = max_fe
        self.initial_a = a
        self.initial_l = l
        self.initial_d = d
        self.fe = 0
        self.population = None
        self.fitness = None

    def get_adaptive_parameters(self):
        """根据进化进度自适应调整参数"""
        progress = self.fe / self.max_fe

        # 早期：强探索，后期：强开发
        a = self.initial_a * (1 - 0.5 * progress)  # 吸引子比例逐渐减小
        l = self.initial_l * (1 + 0.5 * progress)  # 学习率逐渐增大
        d = self.initial_d * (1 - 0.7 * progress)  # 扰动强度逐渐减小

        return max(a, 0.1), max(l, 0.5), max(d, 0.1)

    def initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.n_population, self.dim))

    def attractor_trend_strategy(self, population, fitness):
        a, l, d = self.get_adaptive_parameters()
        n_attractors = max(1, int(a * self.n_population))

        sorted_indices = np.argsort(fitness)
        attractors = population[sorted_indices[:n_attractors]]

        # 计算种群多样性
        diversity = np.mean(np.std(population, axis=0))

        new_population = np.copy(population)
        for i in range(self.n_population):
            if i not in sorted_indices[:n_attractors]:
                k = np.random.randint(0, n_attractors)
                r1 = np.random.rand()

                # 根据多样性调整扰动强度
                adaptive_std = (self.ub - self.lb) * max(0.1, (1 - diversity / np.mean(self.ub - self.lb)))
                w = np.random.normal(0, adaptive_std, self.dim)

                new_population[i] = l * (r1 ** 2) * (attractors[k] - population[i]) + w

        return np.clip(new_population, self.lb, self.ub)

    def coupling_disturbance_strategy(self, population):
        _, _, d = self.get_adaptive_parameters()
        n = self.n_population
        new_population = np.copy(population)

        for i in range(n):
            r2 = np.random.uniform(0, 0.5)
            x_add = r2 * np.sum(population, axis=0) / np.sum(population)

            r3 = np.random.uniform(0, 0.5)
            x_diff = r3 * np.sum(population[i] - population, axis=0) / np.sum(population)

            x_couple = d * (x_add + x_diff)
            new_population[i] += x_couple

        return np.clip(new_population, self.lb, self.ub)

    def information_projection_strategy(self, population, attractor_population, couple_population):
        new_population = np.copy(population)
        n = self.n_population
        dim = self.dim

        for i in range(n):
            C_attract = np.random.choice([0, 1], size=dim)
            R_attract = np.random.uniform(0, 1, dim)
            X_C_attract = C_attract * R_attract * attractor_population[i]

            C_couple = np.random.choice([0, 1], size=dim)
            R_couple = np.random.uniform(0, 1, dim)
            r4 = np.random.uniform(0, 1)
            couple_factor = r4 * (1 - self.fe / self.max_fe)
            X_C_couple = C_couple * R_couple * couple_population[i] * couple_factor

            new_population[i] = population[i] + X_C_attract + X_C_couple

        return np.clip(new_population, self.lb, self.ub)

    def optimize(self, objective_func):
        self.population = self.initialize_population()
        self.fitness = np.array([objective_func(ind) for ind in self.population])
        self.fe += self.n_population

        best_fitness = np.min(self.fitness)
        best_solution = self.population[np.argmin(self.fitness)]

        fitness_history = [best_fitness]

        while self.fe < self.max_fe:
            # 应用问题特定启发式
            self.population = self.problem_specific_heuristics(self.population)

            attractor_population = self.attractor_trend_strategy(self.population, self.fitness)
            couple_population = self.coupling_disturbance_strategy(self.population)
            new_population = self.information_projection_strategy(self.population, attractor_population,
                                                                  couple_population)

            new_fitness = np.array([objective_func(ind) for ind in new_population])
            self.fe += self.n_population

            improve_mask = new_fitness < self.fitness
            self.population[improve_mask] = new_population[improve_mask]
            self.fitness[improve_mask] = new_fitness[improve_mask]

            current_best = np.min(self.fitness)
            if current_best < best_fitness:
                best_fitness = current_best
                best_solution = self.population[np.argmin(self.fitness)]

            fitness_history.append(best_fitness)

            if self.fe >= self.max_fe:
                break

        return best_fitness, best_solution, fitness_history

    def problem_specific_heuristics(self, population):
        """针对服务器分配问题的启发式操作"""
        improved_population = np.copy(population)

        for i in range(len(population)):
            individual = population[i].copy()

            # 启发式1: 修复不可行解（RAM约束）
            for j in range(n_devices):
                server_idx = int(np.clip(individual[j], 0, n_servers - 1))
                if ram_requirement[j] > server_ram[server_idx]:
                    # 找到满足RAM约束的服务器
                    feasible_servers = [s for s in range(n_servers)
                                        if ram_requirement[j] <= server_ram[s]]
                    if feasible_servers:
                        individual[j] = np.random.choice(feasible_servers)

            # 启发式2: 负载均衡
            server_loads = np.zeros(n_servers)
            for j in range(n_devices):
                server_idx = int(np.clip(individual[j], 0, n_servers - 1))
                server_loads[server_idx] += data_size[j] / server_speed[server_idx]

            # 如果某个服务器负载过高，迁移部分设备
            overload_threshold = np.mean(server_loads) * 1.5
            overloaded_servers = np.where(server_loads > overload_threshold)[0]

            for server in overloaded_servers:
                devices_on_server = [j for j in range(n_devices)
                                     if int(individual[j]) == server]
                if devices_on_server:
                    # 迁移部分设备到负载较低的服务器
                    n_to_migrate = max(1, len(devices_on_server) // 3)
                    devices_to_migrate = np.random.choice(devices_on_server,
                                                          n_to_migrate, replace=False)

                    for device in devices_to_migrate:
                        low_load_servers = np.where(server_loads < overload_threshold)[0]
                        if len(low_load_servers) > 0:
                            new_server = np.random.choice(low_load_servers)
                            individual[device] = new_server
                            server_loads[server] -= data_size[device] / server_speed[server]
                            server_loads[new_server] += data_size[device] / server_speed[new_server]

            improved_population[i] = individual

        return improved_population


class MultiPopulationNPDOA:
    def __init__(self, n_populations, n_population, dim, lb, ub, max_fe):
        self.n_populations = n_populations
        self.optimizers = []

        for i in range(n_populations):
            # 不同种群使用不同的参数配置
            a = 0.2 + 0.2 * i / n_populations
            l = 0.8 + 0.8 * i / n_populations
            d = 0.5 + 0.5 * i / n_populations

            optimizer = AdaptiveNPDOA_Optimizer(n_population, dim, lb, ub, max_fe, a, l, d)
            self.optimizers.append(optimizer)

    def optimize(self, objective_func, migration_interval=100):
        best_solutions = []
        best_fitnesses = []

        # 初始化各种群
        for optimizer in self.optimizers:
            population = optimizer.initialize_population()
            fitness = np.array([objective_func(ind) for ind in population])
            optimizer.fe += optimizer.n_population

            best_idx = np.argmin(fitness)
            best_solutions.append(population[best_idx])
            best_fitnesses.append(fitness[best_idx])

            optimizer.population = population
            optimizer.fitness = fitness

        global_best_fitness = min(best_fitnesses)
        global_best_solution = best_solutions[np.argmin(best_fitnesses)]
        fitness_history = [global_best_fitness]

        while self.optimizers[0].fe < self.optimizers[0].max_fe:
            for i, optimizer in enumerate(self.optimizers):
                if optimizer.fe >= optimizer.max_fe:
                    continue

                # 定期进行种群间迁移
                if optimizer.fe % migration_interval == 0 and len(self.optimizers) > 1:
                    self.migrate_between_populations()

                # 执行NPDOA步骤
                attractor_population = optimizer.attractor_trend_strategy(optimizer.population, optimizer.fitness)
                couple_population = optimizer.coupling_disturbance_strategy(optimizer.population)
                new_population = optimizer.information_projection_strategy(
                    optimizer.population, attractor_population, couple_population)

                new_fitness = np.array([objective_func(ind) for ind in new_population])
                optimizer.fe += optimizer.n_population

                improve_mask = new_fitness < optimizer.fitness
                optimizer.population[improve_mask] = new_population[improve_mask]
                optimizer.fitness[improve_mask] = new_fitness[improve_mask]

                current_best_idx = np.argmin(optimizer.fitness)
                if optimizer.fitness[current_best_idx] < global_best_fitness:
                    global_best_fitness = optimizer.fitness[current_best_idx]
                    global_best_solution = optimizer.population[current_best_idx].copy()

                fitness_history.append(global_best_fitness)

        return global_best_fitness, global_best_solution, fitness_history

    def migrate_between_populations(self):
        """种群间个体迁移"""
        migration_rate = 0.1

        for i in range(self.n_populations):
            if np.random.rand() < migration_rate:
                # 选择迁出个体
                source_idx = np.random.randint(0, self.n_populations)
                target_idx = np.random.randint(0, self.n_populations)

                if source_idx != target_idx:
                    # 迁移最优个体
                    best_idx_source = np.argmin(self.optimizers[source_idx].fitness)
                    worst_idx_target = np.argmax(self.optimizers[target_idx].fitness)

                    self.optimizers[target_idx].population[worst_idx_target] = \
                        self.optimizers[source_idx].population[best_idx_source].copy()
                    self.optimizers[target_idx].fitness[worst_idx_target] = \
                        self.optimizers[source_idx].fitness[best_idx_source]


class HybridLocalSearch:
    def __init__(self, n_servers, n_devices):
        self.n_servers = n_servers
        self.n_devices = n_devices

    def variable_neighborhood_search(self, solution, objective_func, max_iter=100):
        current_solution = solution.copy()
        current_cost = objective_func(current_solution)

        for iteration in range(max_iter):
            # 多种邻域结构
            neighborhoods = [
                self.swap_neighborhood,
                self.shift_neighborhood,
                self.multi_swap_neighborhood
            ]

            improved = False
            for neighborhood_func in neighborhoods:
                new_solution = neighborhood_func(current_solution)
                new_cost = objective_func(new_solution)

                if new_cost < current_cost:
                    current_solution = new_solution
                    current_cost = new_cost
                    improved = True
                    break

            if not improved:
                break

        return current_solution, current_cost

    def swap_neighborhood(self, solution):
        """交换两个设备的服务器"""
        new_solution = solution.copy()
        i, j = np.random.choice(self.n_devices, 2, replace=False)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution

    def shift_neighborhood(self, solution):
        """将一个设备移动到其他服务器"""
        new_solution = solution.copy()
        device_idx = np.random.randint(0, self.n_devices)
        new_server = np.random.randint(0, self.n_servers)
        new_solution[device_idx] = new_server
        return new_solution

    def multi_swap_neighborhood(self, solution):
        """同时交换多个设备"""
        new_solution = solution.copy()
        n_swaps = max(2, self.n_devices // 20)  # 交换约5%的设备
        devices_to_swap = np.random.choice(self.n_devices, n_swaps * 2, replace=False)

        for i in range(0, len(devices_to_swap), 2):
            idx1, idx2 = devices_to_swap[i], devices_to_swap[i + 1]
            new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]

        return new_solution


# ========================= SAC实现（保持不变） =========================
class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        state, action, reward, next_state = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), np.stack(action), np.stack(reward), np.stack(next_state)

    def __len__(self):
        return len(self.buffer)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        ###### 确保维度是整数
        if isinstance(state_dim, tuple):
            state_dim = state_dim[0]
        if isinstance(action_dim, tuple):
            action_dim = action_dim[0]


        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x))
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        ##### 确保维度是整数
        if isinstance(state_dim, tuple):
            state_dim = state_dim[0]
        if isinstance(action_dim, tuple):
            action_dim = action_dim[0]


        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class SAC:
    def __init__(self, state_dim, action_dim):
        ############ 确保维度是整数
        if isinstance(state_dim, (int, float)):
            self.state_dim = (int(state_dim),)
        else:
            self.state_dim = tuple(state_dim)

        if isinstance(action_dim, (int, float)):
            self.action_dim = (int(action_dim),)
        else:
            self.action_dim = tuple(action_dim)

        print(f"SAC initialized with state_dim: {self.state_dim}, action_dim: {self.action_dim}")  # 调试



        self.gamma = 0.99
        self.tau = 5e-3
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        self.target_entropy = -self.action_dim[0] # self.target_entropy = -action_dim
        self.alpha_optim = optim.Adam([self.log_alpha], lr=3e-4)

        self.actor = GaussianPolicy(self.state_dim[0], self.action_dim[0])  # 修改这里
        self.critic1 = QNetwork(self.state_dim[0], self.action_dim[0])  # 修改这里
        self.critic2 = QNetwork(self.state_dim[0], self.action_dim[0])  # 修改这里
        self.target_critic1 = QNetwork(self.state_dim[0], self.action_dim[0])  # 修改这里
        self.target_critic2 = QNetwork(self.state_dim[0], self.action_dim[0])  # 修改这里

        # self.actor = GaussianPolicy(state_dim, action_dim)
        # self.critic1 = QNetwork(state_dim, action_dim)
        # self.critic2 = QNetwork(state_dim, action_dim)
        # self.target_critic1 = QNetwork(state_dim, action_dim)
        # self.target_critic2 = QNetwork(state_dim, action_dim)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.replay_buffer = ReplayBuffer()
        self.batch_size = 256

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if evaluate:
            with torch.no_grad():
                action, _ = self.actor.sample(state)
        else:
            action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)

        self.alpha = self.log_alpha.exp()

        # Critic损失
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_next = self.target_critic1(next_state, next_action)
            q2_next = self.target_critic2(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = reward + self.gamma * q_next

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # Actor损失
        pi, log_prob = self.actor.sample(state)
        q1_pi = self.critic1(state, pi)
        q2_pi = self.critic2(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_prob - min_q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Alpha损失
        alpha_loss = - (self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # 目标网络软更新
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# ========================= 增强的函数 =========================
def enhanced_neural_population_dynamic_optimization(n_population, dimensions, bounds, max_fe, weight_m, weight_n):
    """改进版的NPDOA算法"""

    def weighted_objective(pos):
        return calculate_total_cost(pos, weight_m, weight_n)

    # 使用多种群协同进化
    multi_pop_optimizer = MultiPopulationNPDOA(
        n_populations=3,  # 3个种群
        n_population=n_population // 3,
        dim=dimensions,
        lb=bounds[0],
        ub=bounds[1],
        max_fe=max_fe
    )

    best_cost, best_pos, fitness_history = multi_pop_optimizer.optimize(weighted_objective)

    # 混合局部搜索
    local_searcher = HybridLocalSearch(n_servers, n_devices)
    best_pos, best_cost = local_searcher.variable_neighborhood_search(best_pos, weighted_objective, max_iter=200)

    return best_cost, best_pos, fitness_history


def enhanced_sac_training(initial_state, weight_m, weight_n, steps=8000):
    """增强的SAC训练"""
    state_dim = n_devices
    action_dim = n_devices


    ##### 修复：确保维度是元组
    state_dim = (int(state_dim),)
    action_dim = (int(action_dim),)


    sac_agent = SAC(state_dim, action_dim)

    state = initial_state
    best_cost = float('inf')
    best_action = None

    for step in range(steps):
        action = sac_agent.select_action(state)
        action_scaled = np.clip(action, 0, 1) * (n_servers - 1)
        next_state = action_scaled / (n_servers - 1)

        cost = calculate_total_cost(action_scaled, weight_m, weight_n)
        reward = -cost / 1000

        sac_agent.replay_buffer.push(state, action, reward, next_state)
        state = next_state
        sac_agent.update()

        if cost < best_cost:
            best_cost = cost
            best_action = action_scaled.copy()

    return best_action, best_cost


def find_pareto_front(costs, latencies):
    """改进的帕累托前沿计算"""
    points = np.column_stack((costs, latencies))
    pareto_front = []

    for i, point in enumerate(points):
        dominated = False
        for j, other_point in enumerate(points):
            if i == j:
                continue
            # 检查是否被其他点支配
            if (other_point[0] <= point[0] and other_point[1] <= point[1] and
                    (other_point[0] < point[0] or other_point[1] < point[1])):
                dominated = True
                break
        if not dominated:
            pareto_front.append(point)

    pareto_front = np.array(pareto_front)
    # 按成本排序
    if len(pareto_front) > 0:
        pareto_front = pareto_front[pareto_front[:, 0].argsort()]

    return pareto_front


def local_search_optimization(initial_pos, iterations=1000):
    """局部搜索优化"""
    current_pos = initial_pos.copy()
    current_cost = calculate_total_cost(current_pos)
    best_pos = current_pos.copy()
    best_cost = current_cost

    print(f"Local search initial cost: {current_cost:.2f}")

    for i in range(iterations):
        new_pos = current_pos.copy()

        # 随机调整一个设备
        device_idx = np.random.randint(0, n_devices)
        new_server = np.random.randint(0, n_servers)
        new_pos[device_idx] = new_server

        new_cost = calculate_total_cost(new_pos)

        if new_cost < current_cost:
            current_pos = new_pos.copy()
            current_cost = new_cost

            if new_cost < best_cost:
                best_pos = new_pos.copy()
                best_cost = new_cost

    print(f"Local search completed. Best cost: {best_cost:.2f}")
    return best_pos, best_cost

# ========================= 【新增函数】参数扰动鲁棒性测试 =========================
# def test_parameter_robustness(best_solution, noise_levels=[0.1, 0.2, 0.3], n_trials=5):
#     """
#     测试参数扰动鲁棒性：
#     - 对 network_speed 和 data_size 加入 [-noise, +noise] 的均匀噪声
#     - 不重新优化，仅评估 best_solution 的性能退化
#     - 输出：平均成本上升百分比
#     """
#     print("\n" + "="*60)
#     print("开始参数扰动鲁棒性测试 (Parameter Perturbation Robustness)")
#     print("="*60)
#
#     original_cost, original_latency = calculate_raw_values(best_solution)
#     print(f"原始环境最优成本: {original_cost:.2f}, 延迟: {original_latency:.2f}s")
#
#     results = []
#     for noise in noise_levels:
#         cost_ratios = []
#         for trial in range(n_trials):
#             # 备份原始参数
#             backup_speed = network_speed.copy()
#             backup_data = data_size.copy()
#
#             # 添加扰动
#             factor = np.random.uniform(1 - noise, 1 + noise, size=n_devices)
#             network_speed[:] = network_speed * factor
#             data_size[:] = data_size * np.random.uniform(1 - noise, 1 + noise, size=n_devices)
#
#             # 评估原解在新环境下的性能
#             noisy_cost, noisy_latency = calculate_raw_values(best_solution)
#
#             # 恢复环境
#             network_speed[:] = backup_speed
#             data_size[:] = backup_data
#
#             cost_ratio = noisy_cost / original_cost
#             cost_ratios.append(cost_ratio)
#
#         avg_ratio = np.mean(cost_ratios)
#         std_ratio = np.std(cost_ratios)
#         results.append((noise, avg_ratio, std_ratio))
#         print(f"噪声 ±{noise*100:4.1f}% → 平均成本上升: { (avg_ratio-1)*100:6.2f}% ± {std_ratio*100:5.2f}%")
#
#     # 绘制鲁棒性曲线
#     plt.figure(figsize=(8, 5))
#     noises, ratios, stds = zip(*results)
#     plt.errorbar([n*100 for n in noises], [(r-1)*100 for r in ratios],
#                  yerr=[s*100 for s in stds], fmt='o-', capsize=5, color='red', label='Cost Increase')
#     plt.axhline(y=15, color='orange', linestyle='--', label='15% 阈值')
#     plt.axhline(y=30, color='red', linestyle='--', label='30% 阈值')
#     plt.xlabel('扰动幅度 (%)')
#     plt.ylabel('成本上升 (%)')
#     plt.title('参数扰动鲁棒性测试\n(网络速度 & 数据量 ±扰动)')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('parameter_robustness_test.png', dpi=300, bbox_inches='tight')
#     plt.show()
#
#     # 总结
#     robust = all(r <= 1.15 for _, r, _ in results)  # 所有噪声下成本上升 < 15%
#     print(f"\n鲁棒性判定：{'通过' if robust else '未通过'} (成本上升 < 15%)")
#     return results, robust

def test_parameter_robustness(best_solution, noise_levels=[0.1, 0.2, 0.3], n_trials=5):
    """
    Test parameter perturbation robustness:
    - Add uniform noise in range [-noise, +noise] to network_speed and data_size
    - No re-optimization, only evaluate performance degradation of best_solution
    - Output: Average cost increase percentage
    """
    print("\n" + "="*60)
    print("Starting Parameter Perturbation Robustness Test")
    print("="*60)

    original_cost, original_latency = calculate_raw_values(best_solution)
    print(f"Original optimal cost: {original_cost:.2f}, Latency: {original_latency:.2f}s")

    results = []
    for noise in noise_levels:
        cost_ratios = []
        for trial in range(n_trials):
            # Backup original parameters
            backup_speed = network_speed.copy()
            backup_data = data_size.copy()

            # Add perturbation
            factor = np.random.uniform(1 - noise, 1 + noise, size=n_devices)
            network_speed[:] = network_speed * factor
            data_size[:] = data_size * np.random.uniform(1 - noise, 1 + noise, size=n_devices)

            # Evaluate original solution performance in new environment
            noisy_cost, noisy_latency = calculate_raw_values(best_solution)

            # Restore environment
            network_speed[:] = backup_speed
            data_size[:] = backup_data

            cost_ratio = noisy_cost / original_cost
            cost_ratios.append(cost_ratio)

        avg_ratio = np.mean(cost_ratios)
        std_ratio = np.std(cost_ratios)
        results.append((noise, avg_ratio, std_ratio))
        print(f"Noise ±{noise*100:4.1f}% → Average cost increase: { (avg_ratio-1)*100:6.2f}% ± {std_ratio*100:5.2f}%")

    # Plot robustness curve
    plt.figure(figsize=(8, 5))
    noises, ratios, stds = zip(*results)
    plt.errorbar([n*100 for n in noises], [(r-1)*100 for r in ratios],
                 yerr=[s*100 for s in stds], fmt='o-', capsize=5, color='red', label='Cost Increase')
    plt.axhline(y=15, color='orange', linestyle='--', label='15% Threshold')
    plt.axhline(y=30, color='red', linestyle='--', label='30% Threshold')
    plt.xlabel('Perturbation Amplitude (%)')
    plt.ylabel('Cost Increase (%)')
    plt.title('Parameter Perturbation Robustness Test\n(Network Speed & Data Size ±Perturbation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('parameter_robustness_test.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Summary
    robust = all(r <= 1.15 for _, r, _ in results)  # Cost increase < 15% for all noise levels
    print(f"\nRobustness Assessment: {'PASS' if robust else 'FAIL'} (Cost Increase < 15%)")
    return results, robust

# ========================= 主要的测试函数 =========================
def test_weight_combinations_enhanced():
    """增强的权重组合测试函数"""
    weight_combinations = [
        (10, 0.01),
        (1, 1),
        (0.01, 10)
    ]
    colors = ['blue', 'green', 'red']

    plt.figure(figsize=(18, 6))
    all_costs = []
    all_latencies = []

    for idx, (weight_m, weight_n) in enumerate(weight_combinations, 1):
        print(f"Testing weight combination {idx}: m={weight_m}, n={weight_n}")

        max_retries = 3
        best_pareto_size = 0
        best_results = None

        for retry in range(max_retries):
            print(f"  Attempt {retry + 1}/{max_retries}")# print(f"  尝试 {retry + 1}/{max_retries}")

            # 使用改进的NPDOA
            npdoa_cost, npdoa_pos, fitness_history = enhanced_neural_population_dynamic_optimization(
                n_population=180,         ###################### 改
                dimensions=n_devices,
                bounds=(np.zeros(n_devices), (n_servers - 1) * np.ones(n_devices)),
                max_fe=3000,
                weight_m=weight_m,
                weight_n=weight_n
            )

            print(f"AdapNPDOA completed, best cost: {npdoa_cost:.2f}")

            # 使用增强的SAC
            initial_state = npdoa_pos / (n_servers - 1)
            best_action, best_cost = enhanced_sac_training(
                initial_state, weight_m, weight_n, steps=8000
            )

            # 局部搜索优化
            best_action, best_cost = local_search_optimization(best_action, iterations=1000)

            # 生成测试样本
            test_positions = []
            for _ in range(50):
                noise = np.random.normal(0, 0.3, n_devices)
                noisy_pos = best_action + noise
                test_positions.append(np.clip(noisy_pos, 0, n_servers - 1))
            test_positions = np.array(test_positions)

            costs, latencies = [], []
            for pos in test_positions:
                cost_val, latency_val = calculate_raw_values(pos)
                costs.append(cost_val)
                latencies.append(latency_val)

            costs, latencies = np.array(costs), np.array(latencies)

            # 检查帕累托前沿质量
            pareto_front = find_pareto_front(costs, latencies)
            pareto_size = len(pareto_front)

            print(f"  Found {pareto_size} non-dominated solutions")# print(f"    找到 {pareto_size} 个非支配解")

            # 保留最好的结果
            if pareto_size > best_pareto_size:
                best_pareto_size = pareto_size
                best_results = (costs, latencies, best_action, pareto_front)

            # 如果找到足够多的非支配解，提前结束重试
            if pareto_size >= 5:
                print(f"  Found enough non-dominated solutions, ending retries early")# print(f"    找到足够的非支配解，提前结束重试")
                break

        # 使用最佳结果
        costs, latencies, best_action, pareto_front = best_results

        all_costs.extend(costs)
        all_latencies.extend(latencies)

        plt.subplot(1, 3, idx)
        plt.scatter(costs, latencies, alpha=0.6, c=colors[idx - 1],
                    label=f'Solutions (m={weight_m}, n={weight_n})')

        # 计算并绘制帕累托前沿
        if len(pareto_front) > 1:
            try:
                hull = ConvexHull(pareto_front)
                pareto_sorted = pareto_front[hull.vertices]
                pareto_sorted = pareto_sorted[pareto_sorted[:, 0].argsort()]
                plt.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 'r-',
                         linewidth=3, label='Pareto Front')
            except:
                pareto_sorted = pareto_front[pareto_front[:, 0].argsort()]
                plt.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 'r-',
                         linewidth=3, label='Pareto Front')

        plt.xlabel('Cost (evaluated with m=1)')
        plt.ylabel('Latency (evaluated with n=1)')
        plt.title(f'Optimized with m={weight_m}, n={weight_n}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('enhanced_npdoa_sac_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制所有点的总体帕累托前沿
    plt.figure(figsize=(10, 8))
    plt.scatter(all_costs, all_latencies, alpha=0.5, c='blue', label='All Solutions')

    overall_pareto = find_pareto_front(np.array(all_costs), np.array(all_latencies))
    if len(overall_pareto) > 1:
        overall_pareto = overall_pareto[overall_pareto[:, 0].argsort()]
        plt.plot(overall_pareto[:, 0], overall_pareto[:, 1], 'r-',
                 linewidth=3, label='Overall Pareto Front')

    plt.xlabel('Cost (evaluated with m=1)')
    plt.ylabel('Latency (evaluated with n=1)')
    plt.title(f'Optimized with m={m}, n={n}')
    plt.legend()
    plt.grid(True)
    plt.savefig('overall_pareto_front.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_npdoa_sac(runs=1, total_steps=20_000):
    """主运行函数"""
    print(f"Run 1/1")

    # 使用改进的NPDOA
    npdoa_cost, npdoa_pos, fitness_history = enhanced_neural_population_dynamic_optimization(
        n_population=180,                 ###################### 改
        dimensions=n_devices,
        bounds=(np.zeros(n_devices), (n_servers - 1) * np.ones(n_devices)),
        max_fe=4500,
        weight_m=m,
        weight_n=n
    )

    print(f"AdapNPDOA completed, best cost: {npdoa_cost:.2f}")

    # SAC阶段
    state_dim = n_devices
    action_dim = n_devices


    ########## 确保维度是整数元组
    state_dim = (int(state_dim),)  # 转换为元组 (250,)
    action_dim = (int(action_dim),)  # 转换为元组 (250,)
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")  # 调试信息


    sac_agent = SAC(state_dim, action_dim)

    state = npdoa_pos / (n_servers - 1)
    best_cost = npdoa_cost
    best_action = npdoa_pos.copy()

    steps = 0
    cost_history = []

    checkpoint_interval = total_steps // 10
    checkpoint_costs = []

    while steps < total_steps:
        action = sac_agent.select_action(state)
        action_scaled = np.clip(action, 0, 1) * (n_servers - 1)

        next_state = action_scaled / (n_servers - 1)
        cost = calculate_total_cost(action_scaled)
        reward = -cost / 1000

        sac_agent.replay_buffer.push(state, action, reward, next_state)
        state = next_state
        sac_agent.update()

        if cost < best_cost:
            best_cost = cost
            best_action = action_scaled.copy()

        steps += 1

        if steps % checkpoint_interval == 0:
            checkpoint_costs.append(best_cost)
            print(f"Step {steps}, Best Cost: {best_cost:.2f}")

        if steps % 1000 == 0:
            cost_history.append(best_cost)

    # 局部搜索优化
    print("Starting local search optimization...")
    best_action, final_best_cost = local_search_optimization(best_action, iterations=2000)

    # ========================= 【新增调用】参数扰动鲁棒性测试 =========================
    print("\nPerforming parameter perturbation robustness test...")
    robustness_results, is_robust = test_parameter_robustness(best_action, noise_levels=[0.1, 0.2, 0.3], n_trials=5)
    # ============================================================================

    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(cost_history)) * 1000, cost_history)
    plt.xlabel('Training Steps')
    plt.ylabel('Best Cost')
    plt.title('SAC Training Process (with Improved AdapNPDOA initialization)')
    plt.grid(True)
    plt.savefig('training_process.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Training completed. Testing different weight combinations...")
    test_weight_combinations_enhanced()

    return final_best_cost


if __name__ == "__main__":
    final_best_cost = run_npdoa_sac(runs=1)  # 修改：接收两个返回值
    print(f"Final Best Cost: {final_best_cost:.2f}")



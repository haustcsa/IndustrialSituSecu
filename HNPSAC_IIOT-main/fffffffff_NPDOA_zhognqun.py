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
server_ram = np.random.uniform(2, 8, n_servers)
data_size = np.random.uniform(50, 150, n_devices)
completion_requirement = np.random.uniform(20, 40, n_devices)
ram_requirement = np.random.uniform(1, 2, n_devices)
m, n = 10, 1e-2
# m, n = 1, 1
# m, n = 1e-2, 10


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
        a = self.initial_a * (1 - 0.5 * progress)
        l = self.initial_l * (1 + 0.5 * progress)
        d = self.initial_d * (1 - 0.7 * progress)
        return max(a, 0.1), max(l, 0.5), max(d, 0.1)

    def initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.n_population, self.dim))

    def attractor_trend_strategy(self, population, fitness):
        a, l, d = self.get_adaptive_parameters()
        n_attractors = max(1, int(a * self.n_population))

        sorted_indices = np.argsort(fitness)
        attractors = population[sorted_indices[:n_attractors]]

        diversity = np.mean(np.std(population, axis=0))
        adaptive_std = (self.ub - self.lb) * max(0.1, (1 - diversity / np.mean(self.ub - self.lb)))

        new_population = np.copy(population)
        for i in range(self.n_population):
            if i not in sorted_indices[:n_attractors]:
                k = np.random.randint(0, n_attractors)
                r1 = np.random.rand()
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

    def problem_specific_heuristics(self, population):
        """针对服务器分配问题的启发式操作"""
        improved_population = np.copy(population)

        for i in range(len(population)):
            individual = population[i].copy()

            # 启发式1: 修复不可行解（RAM约束）
            for j in range(n_devices):
                server_idx = int(np.clip(individual[j], 0, n_servers - 1))
                if ram_requirement[j] > server_ram[server_idx]:
                    feasible_servers = [s for s in range(n_servers) if ram_requirement[j] <= server_ram[s]]
                    if feasible_servers:
                        individual[j] = np.random.choice(feasible_servers)

            # 启发式2: 负载均衡
            server_loads = np.zeros(n_servers)
            for j in range(n_devices):
                server_idx = int(np.clip(individual[j], 0, n_servers - 1))
                server_loads[server_idx] += data_size[j] / server_speed[server_idx]

            overload_threshold = np.mean(server_loads) * 1.5
            overloaded_servers = np.where(server_loads > overload_threshold)[0]

            for server in overloaded_servers:
                devices_on_server = [j for j in range(n_devices) if int(individual[j]) == server]
                if devices_on_server:
                    n_to_migrate = max(1, len(devices_on_server) // 3)
                    devices_to_migrate = np.random.choice(devices_on_server, n_to_migrate, replace=False)

                    for device in devices_to_migrate:
                        low_load_servers = np.where(server_loads < overload_threshold)[0]
                        if len(low_load_servers) > 0:
                            new_server = np.random.choice(low_load_servers)
                            individual[device] = new_server
                            server_loads[server] -= data_size[device] / server_speed[server]
                            server_loads[new_server] += data_size[device] / server_speed[new_server]

            improved_population[i] = individual

        return improved_population

    def optimize(self, objective_func):
        self.population = self.initialize_population()
        self.fitness = np.array([objective_func(ind) for ind in self.population])
        self.fe += self.n_population

        best_fitness = np.min(self.fitness)
        best_solution = self.population[np.argmin(self.fitness)]
        fitness_history = [best_fitness]

        while self.fe < self.max_fe:
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


class MultiPopulationNPDOA:
    def __init__(self, n_populations, n_population, dim, lb, ub, max_fe):
        self.n_populations = n_populations
        self.optimizers = []

        for i in range(n_populations):
            a = 0.2 + 0.2 * i / n_populations
            l = 0.8 + 0.8 * i / n_populations
            d = 0.5 + 0.5 * i / n_populations
            optimizer = AdaptiveNPDOA_Optimizer(n_population, dim, lb, ub, max_fe, a, l, d)
            self.optimizers.append(optimizer)

    def optimize(self, objective_func, migration_interval=100):
        best_solutions = []
        best_fitnesses = []

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

                if optimizer.fe % migration_interval == 0 and len(self.optimizers) > 1:
                    self.migrate_between_populations()

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
        migration_rate = 0.1
        for i in range(self.n_populations):
            if np.random.rand() < migration_rate:
                source_idx = np.random.randint(0, self.n_populations)
                target_idx = np.random.randint(0, self.n_populations)
                if source_idx != target_idx:
                    best_idx_source = np.argmin(self.optimizers[source_idx].fitness)
                    worst_idx_target = np.argmax(self.optimizers[target_idx].fitness)
                    self.optimizers[target_idx].population[worst_idx_target] = self.optimizers[source_idx].population[
                        best_idx_source].copy()
                    self.optimizers[target_idx].fitness[worst_idx_target] = self.optimizers[source_idx].fitness[
                        best_idx_source]


class HybridLocalSearch:
    def __init__(self, n_servers, n_devices):
        self.n_servers = n_servers
        self.n_devices = n_devices

    def variable_neighborhood_search(self, solution, objective_func, max_iter=100):
        current_solution = solution.copy()
        current_cost = objective_func(current_solution)

        for iteration in range(max_iter):
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
        new_solution = solution.copy()
        i, j = np.random.choice(self.n_devices, 2, replace=False)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution

    def shift_neighborhood(self, solution):
        new_solution = solution.copy()
        device_idx = np.random.randint(0, self.n_devices)
        new_server = np.random.randint(0, self.n_servers)
        new_solution[device_idx] = new_server
        return new_solution

    def multi_swap_neighborhood(self, solution):
        new_solution = solution.copy()
        n_swaps = max(2, self.n_devices // 20)
        devices_to_swap = np.random.choice(self.n_devices, n_swaps * 2, replace=False)
        for i in range(0, len(devices_to_swap), 2):
            idx1, idx2 = devices_to_swap[i], devices_to_swap[i + 1]
            new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
        return new_solution


# ========================= SAC实现 =========================
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
        self.gamma = 0.99
        self.tau = 5e-3
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        self.target_entropy = -action_dim
        self.alpha_optim = optim.Adam([self.log_alpha], lr=3e-4)

        self.actor = GaussianPolicy(state_dim, action_dim)
        self.critic1 = QNetwork(state_dim, action_dim)
        self.critic2 = QNetwork(state_dim, action_dim)
        self.target_critic1 = QNetwork(state_dim, action_dim)
        self.target_critic2 = QNetwork(state_dim, action_dim)

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

        pi, log_prob = self.actor.sample(state)
        q1_pi = self.critic1(state, pi)
        q2_pi = self.critic2(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_prob - min_q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = - (self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# ========================= 改进的NPDOA+SAC运行函数 =========================
def enhanced_neural_population_dynamic_optimization(n_population, dimensions, bounds, max_fe, weight_m, weight_n):
    """改进版的NPDOA算法"""

    def weighted_objective(pos):
        return calculate_total_cost(pos, weight_m, weight_n)

    multi_pop_optimizer = MultiPopulationNPDOA(
        n_populations=3,
        n_population=n_population // 3,
        dim=dimensions,
        lb=bounds[0],
        ub=bounds[1],
        max_fe=max_fe
    )

    best_cost, best_pos, fitness_history = multi_pop_optimizer.optimize(weighted_objective)

    local_searcher = HybridLocalSearch(n_servers, n_devices)
    best_pos, best_cost = local_searcher.variable_neighborhood_search(best_pos, weighted_objective, max_iter=200)

    return best_cost, best_pos, fitness_history


def enhanced_sac_training(initial_state, weight_m, weight_n, steps=8000):  ####5000
    """增强的SAC训练"""
    state_dim = n_devices
    action_dim = n_devices
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


def local_search_optimization(initial_pos, iterations=1000):
    """局部搜索优化"""
    current_pos = initial_pos.copy()
    current_cost = calculate_total_cost(current_pos)
    best_pos = current_pos.copy()
    best_cost = current_cost

    for i in range(iterations):
        new_pos = current_pos.copy()
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

    return best_pos, best_cost


def run_enhanced_npdoa_sac(population_size=50, total_steps=8000): ### total_steps=100000
    """运行改进的NPDOA-SAC算法，记录计算时间和性能"""
    computation_times = []  # 记录计算时间 [NPDOA时间, SAC时间, 局部搜索时间, 总时间]
    cost_history = []  # 记录训练过程中的成本变化

    print(f"\nRunning Enhanced NPDOA-SAC with population size: {population_size}")

    # NPDOA阶段 - 记录时间
    npdoa_start_time = time.time()
    npdoa_cost, npdoa_pos, npdoa_history = enhanced_neural_population_dynamic_optimization(
        n_population=population_size,
        dimensions=n_devices,
        bounds=(np.zeros(n_devices), (n_servers - 1) * np.ones(n_devices)),
        max_fe=4500,
        weight_m=m,
        weight_n=n
    )
    npdoa_time = time.time() - npdoa_start_time
    computation_times.append(npdoa_time)
    print(f"Enhanced NPDOA completed in {npdoa_time:.2f} seconds, Best Cost: {npdoa_cost:.2f}")

    # SAC阶段
    state_dim = n_devices
    action_dim = n_devices
    sac_agent = SAC(state_dim, action_dim)

    state = npdoa_pos / (n_servers - 1)
    best_cost = npdoa_cost
    best_action = npdoa_pos.copy()

    steps = 0
    checkpoint_interval = total_steps // 10

    # SAC训练 - 记录时间
    sac_start_time = time.time()

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

        if steps % 1000 == 0:
            cost_history.append(best_cost)

        if steps % checkpoint_interval == 0:
            print(f"Step {steps}, Best Cost: {best_cost:.2f}")

    sac_time = time.time() - sac_start_time
    computation_times.append(sac_time)

    # 局部搜索阶段 - 记录时间
    local_search_start_time = time.time()
    best_action, best_cost = local_search_optimization(best_action, iterations=2000)
    local_search_time = time.time() - local_search_start_time
    computation_times.append(local_search_time)

    total_time = npdoa_time + sac_time + local_search_time
    computation_times.append(total_time)

    print(f"SAC completed in {sac_time:.2f} seconds")
    print(f"Local search completed in {local_search_time:.2f} seconds")
    print(f"Total computation time: {total_time:.2f} seconds")
    print(f"Final Best Cost: {best_cost:.2f}")

    return best_cost, computation_times, cost_history, npdoa_history


# ========================= 参数敏感性分析 =========================
def enhanced_parameter_sensitivity_analysis():
    """分析种群大小对改进的NPDOA-SAC算法性能的影响 - 每个尺寸重复3次"""

    # 设置全局字体大小和字体
    plt.rcParams.update({
        'font.size': 17,  # 设置较大的基础字体
        'font.family': 'serif',
        'font.serif': 'Times New Roman'
    })



    population_sizes = [80, 100, 120, 150, 180, 200]  # 测试不同的种群大小
    num_repeats = 3  # 每个尺寸重复3次
    results = {}

    for size in population_sizes:
        print(f"\n{'=' * 60}")
        print(f"Testing population size: {size}, Repeats: {num_repeats}")
        print(f"{'=' * 60}")

        size_results = {
            'best_costs': [],
            'npdoa_times': [],
            'sac_times': [],
            'local_search_times': [],
            'total_times': [],
            'cost_histories': [],
            'npdoa_histories': []
        }

        for repeat in range(num_repeats):
            print(f"\nRepeat {repeat + 1}/{num_repeats} for population size {size}")

            # 使用不同的随机种子确保每次运行不同
            current_seed = 21 + (size * 100) + repeat
            np.random.seed(current_seed)
            torch.manual_seed(current_seed)

            best_cost, computation_times, cost_history, npdoa_history = run_enhanced_npdoa_sac(
                population_size=size,
                total_steps=8000  # SAC训练步数total_steps=100000
            )

            # 收集每次运行的结果
            size_results['best_costs'].append(best_cost)
            size_results['npdoa_times'].append(computation_times[0])
            size_results['sac_times'].append(computation_times[1])
            size_results['local_search_times'].append(computation_times[2])
            size_results['total_times'].append(computation_times[3])
            size_results['cost_histories'].append(cost_history)
            size_results['npdoa_histories'].append(npdoa_history)

            print(f"Repeat {repeat + 1} - Best Cost: {best_cost:.2f}, Total Time: {computation_times[3]:.2f}s")

        # 计算统计信息
        results[size] = {
            'best_cost_mean': np.mean(size_results['best_costs']),
            'best_cost_std': np.std(size_results['best_costs']),
            'best_cost_min': np.min(size_results['best_costs']),
            'best_cost_max': np.max(size_results['best_costs']),
            'npdoa_time_mean': np.mean(size_results['npdoa_times']),
            'sac_time_mean': np.mean(size_results['sac_times']),
            'local_search_time_mean': np.mean(size_results['local_search_times']),
            'total_time_mean': np.mean(size_results['total_times']),
            'all_best_costs': size_results['best_costs'],
            'all_total_times': size_results['total_times'],
            # 使用第一次运行的收敛历史进行可视化（保持一致性）
            'cost_history': size_results['cost_histories'][0],
            'npdoa_history': size_results['npdoa_histories'][0]
        }

        print(f"\nPopulation Size {size} Summary:")
        print(f"  Best Cost: {results[size]['best_cost_mean']:.2f} ± {results[size]['best_cost_std']:.2f}")
        print(f"  Range: [{results[size]['best_cost_min']:.2f}, {results[size]['best_cost_max']:.2f}]")
        print(f"  Average Time: {results[size]['total_time_mean']:.2f}s")

    # 绘制结果 - 修改图表使用平均值和标准差
    plt.figure(figsize=(18, 12))

    # 1. 种群大小对最终成本的影响（带误差棒）
    plt.subplot(2, 3, 1)
    sizes = list(results.keys())
    mean_costs = [results[size]['best_cost_mean'] for size in sizes]
    std_costs = [results[size]['best_cost_std'] for size in sizes]
    min_costs = [results[size]['best_cost_min'] for size in sizes]

    plt.errorbar(sizes, mean_costs, yerr=std_costs, fmt='o-',
                 linewidth=3, markersize=10, capsize=5, capthick=2,
                 color='blue', label='Mean Cost ± Std')
    plt.plot(sizes, min_costs, 's--', linewidth=2, markersize=6,
             color='red', label='Min Cost')
    plt.xlabel('Population Size')
    plt.ylabel('Best Cost')
    plt.title('Effect of Population Size on Solution Quality\n(Enhanced NPDOA-SAC, 3 repeats)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. 种群大小对计算时间的影响
    plt.subplot(2, 3, 2)
    npdoa_times = [results[size]['npdoa_time_mean'] for size in sizes]
    sac_times = [results[size]['sac_time_mean'] for size in sizes]
    local_times = [results[size]['local_search_time_mean'] for size in sizes]
    total_times = [results[size]['total_time_mean'] for size in sizes]

    plt.plot(sizes, npdoa_times, 'o-', label='NPDOA Time', linewidth=2, markersize=8, color='orange')
    plt.plot(sizes, sac_times, 'o-', label='SAC Time', linewidth=2, markersize=8, color='green')
    plt.plot(sizes, local_times, 'o-', label='Local Search Time', linewidth=2, markersize=8, color='purple')
    plt.plot(sizes, total_times, 'o-', label='Total Time', linewidth=3, markersize=10, color='red')
    plt.xlabel('Population Size')
    plt.ylabel('Time (seconds)')
    plt.title('Effect of Population Size on Computation Time\n(Enhanced NPDOA-SAC, 3 repeats)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. 不同种群大小的SAC收敛曲线（使用第一次运行的数据）
    plt.subplot(2, 3, 3)
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    for i, size in enumerate(sizes):
        cost_history = results[size]['cost_history']
        plt.plot(np.arange(len(cost_history)) * 1000, cost_history,
                 label=f'Pop Size: {size}', linewidth=2, color=colors[i])
    plt.xlabel('SAC Training Steps')
    plt.ylabel('Best Cost')
    plt.title('SAC Convergence for Different Population Sizes\n(Enhanced NPDOA-SAC)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. 不同种群大小的NPDOA收敛曲线（使用第一次运行的数据）
    plt.subplot(2, 3, 4)
    for i, size in enumerate(sizes):
        npdoa_history = results[size]['npdoa_history']
        plt.plot(np.arange(len(npdoa_history)) * (2000 // len(npdoa_history)), npdoa_history,
                 label=f'Pop Size: {size}', linewidth=2, color=colors[i])
    plt.xlabel('NPDOA Function Evaluations')
    plt.ylabel('Best Cost')
    plt.title('NPDOA Convergence for Different Population Sizes\n(Enhanced NPDOA-SAC)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. 成本-时间权衡（显示所有重复运行的点）
    plt.subplot(2, 3, 5)
    for i, size in enumerate(sizes):
        all_costs = results[size]['all_best_costs']
        all_times = results[size]['all_total_times']
        plt.scatter(all_times, all_costs, s=80, color=colors[i], alpha=0.6, label=f'Size {size}')

        # 标注平均值点
        mean_time = results[size]['total_time_mean']
        mean_cost = results[size]['best_cost_mean']
        plt.scatter(mean_time, mean_cost, s=150, color=colors[i], marker='*', edgecolors='black')

    plt.xlabel('Total Computation Time (seconds)')
    plt.ylabel('Best Cost')
    plt.title('Cost-Time Trade-off (All Repeats)\nEnhanced NPDOA-SAC')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. 各阶段时间占比
    plt.subplot(2, 3, 6)
    time_components = {
        'NPDOA': npdoa_times,
        'SAC': sac_times,
        'Local Search': local_times
    }
    bottom = np.zeros(len(sizes))
    for component, times in time_components.items():
        plt.bar(sizes, times, bottom=bottom, label=component, alpha=0.8)
        bottom += times
    plt.xlabel('Population Size')
    plt.ylabel('Time (seconds)')
    plt.title('Time Distribution by Component\n(Enhanced NPDOA-SAC, 3 repeats)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('enhanced_npdoa_sac_population_size_sensitivity_3repeats.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印详细结果（包含统计信息）
    print("\n" + "=" * 100)
    print("DETAILED RESULTS FOR ENHANCED NPDOA-SAC (3 REPEATS)")
    print("=" * 100)
    print("Population Size | Mean Cost ± Std   | Min Cost  | Max Cost  | Mean Time | All Costs")
    print("-" * 100)
    for size in sizes:
        result = results[size]
        costs_str = ", ".join([f"{c:.1f}" for c in result['all_best_costs']])
        print(f"{size:15} | {result['best_cost_mean']:6.2f} ± {result['best_cost_std']:4.2f} | "
              f"{result['best_cost_min']:8.2f} | {result['best_cost_max']:8.2f} | "
              f"{result['total_time_mean']:8.2f}s | [{costs_str}]")

    # 确定最佳种群大小（综合考虑成本和时间）
    normalized_costs = (mean_costs - np.min(mean_costs)) / (np.max(mean_costs) - np.min(mean_costs))
    normalized_times = (total_times - np.min(total_times)) / (np.max(total_times) - np.min(total_times))

    # 综合评分（成本权重0.7，时间权重0.3）
    combined_scores = 0.7 * normalized_costs + 0.3 * normalized_times
    best_idx = np.argmin(combined_scores)
    best_size = sizes[best_idx]

    print(f"\nOPTIMAL POPULATION SIZE ANALYSIS (3 REPEATS):")
    print(f"Based on combined cost-time optimization (70% cost, 30% time):")
    print(f"Optimal population size: {best_size}")
    print(f"Mean cost: {results[best_size]['best_cost_mean']:.2f} ± {results[best_size]['best_cost_std']:.2f}")
    print(f"Mean time: {results[best_size]['total_time_mean']:.2f} seconds")
    print(f"Cost range: [{results[best_size]['best_cost_min']:.2f}, {results[best_size]['best_cost_max']:.2f}]")

    # 绘制综合评分图
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(sizes, normalized_costs, 'o-', label='Normalized Cost', linewidth=2, markersize=8)
    plt.plot(sizes, normalized_times, 'o-', label='Normalized Time', linewidth=2, markersize=8)
    plt.plot(sizes, combined_scores, 'o-', label='Combined Score', linewidth=3, markersize=10, color='red')
    plt.axvline(x=best_size, color='red', linestyle='--', alpha=0.7, label=f'Optimal Size: {best_size}')
    plt.xlabel('Population Size')
    plt.ylabel('Normalized Score')
    plt.title('Optimal Population Size Selection\n(Enhanced NPDOA-SAC, 3 repeats)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加稳定性分析
    plt.subplot(2, 1, 2)
    std_values = [results[size]['best_cost_std'] for size in sizes]
    plt.bar(sizes, std_values, color='orange', alpha=0.7)
    plt.xlabel('Population Size')
    plt.ylabel('Cost Standard Deviation')
    plt.title('Solution Stability Analysis\n(Lower values indicate more stable performance)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimal_population_size_selection_3repeats.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results, best_size


# ========================= 主函数 =========================
if __name__ == "__main__":
    print("Starting enhanced parameter sensitivity analysis for Improved NPDOA-SAC...")
    sensitivity_results, optimal_size = enhanced_parameter_sensitivity_analysis()

    print(f"\n{'=' * 60}")
    print(f"FINAL RECOMMENDATION: Use population size {optimal_size} for optimal performance")
    print(f"{'=' * 60}")
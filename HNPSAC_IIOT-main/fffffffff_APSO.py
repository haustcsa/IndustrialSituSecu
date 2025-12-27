import numpy as np
import pyswarms as ps
import time
import matplotlib.pyplot as plt
from pyswarms.backend.swarms import Swarm
from pyswarms.backend.topology import Star

# Set random seed for consistency
np.random.seed(21)

# Number of servers and devices
n_servers = 20
n_devices = 250

# Simulation parameters (consistent)
# network_speed = np.random.uniform(60, 900, n_devices)  # Mbps
# server_cost = np.random.uniform(0.02, 0.06, n_servers)  # $/second
# server_speed = np.random.uniform(10, 200, n_servers)  # MI/second
network_speed = np.random.uniform(100, 900, n_devices)
server_cost = np.random.uniform(0.01, 0.1, n_servers)
server_speed = np.random.uniform(50, 200, n_servers)
server_ram = np.random.uniform(2, 8, n_servers)  # GB
data_size = np.random.uniform(50, 150, n_devices)  # MB
completion_requirement = np.random.uniform(20, 40, n_devices)  # MI
ram_requirement = np.random.uniform(1, 2, n_devices)  # GB

# Weights for the cost and latency components
# m = 1  # cost weight
# n = 1  # latency weight

m = 1e-2
n = 10

# m = 10
# n = 1e-2

# Generate device workloads
def generate_data():
    device_workloads = np.random.uniform(1e6, 1e8, n_devices)
    return device_workloads


# def calculate_total_cost(device_workloads, positions, data_size, network_speed,
#                          completion_requirement, server_speed, server_cost,
#                          server_ram, ram_requirement, m, n):
#     total_costs = np.zeros(positions.shape[0])
#     total_latencies = np.zeros(positions.shape[0])  # New: store latencies
#
#     for i in range(positions.shape[0]):
#         particle = positions[i]
#         total_cost = 0
#         total_latency = 0  # New: calculate total latency
#
#         for j in range(n_devices):
#             server_index = int(np.clip(particle[j], 0, n_servers - 1))
#
#             transfer_time = data_size[j] / network_speed[j]
#             processing_time = completion_requirement[j] / server_speed[server_index]
#             total_time = transfer_time + processing_time
#
#             if ram_requirement[j] <= server_ram[server_index]:
#                 current_cost = server_cost[server_index] * total_time
#                 total_cost += m * current_cost
#                 total_latency += total_time  # New: accumulate latency
#
#         total_costs[i] = total_cost
#         total_latencies[i] = total_latency * n  # Apply latency weight
#
#     return total_costs, total_latencies  # Now returns two values

def calculate_total_cost(device_workloads, positions, data_size, network_speed,
                         completion_requirement, server_speed, server_cost,
                         server_ram, ram_requirement, m, n):
    total_costs = np.zeros(positions.shape[0])
    total_latencies = np.zeros(positions.shape[0])
    penalty = 1000  # 添加惩罚值

    for i in range(positions.shape[0]):
        particle = positions[i]
        total_cost = 0
        total_latency = 0

        for j in range(n_devices):
            server_index = int(np.clip(particle[j], 0, n_servers - 1))

            if ram_requirement[j] <= server_ram[server_index]:
                # 正常计算
                transfer_time = data_size[j] / network_speed[j]
                processing_time = completion_requirement[j] / server_speed[server_index]
                total_time = transfer_time + processing_time
                current_cost = server_cost[server_index] * total_time
                total_cost += m * current_cost
                total_latency += total_time
            else:
                # 添加惩罚
                total_cost += penalty
                total_latency += penalty

        total_costs[i] = total_cost
        total_latencies[i] = total_latency * n

    return total_costs, total_latencies

def pso_objective_function(positions, device_workloads):
    costs, latencies = calculate_total_cost(device_workloads, positions, data_size, network_speed,
                                            completion_requirement, server_speed, server_cost,
                                            server_ram, ram_requirement, m, n)
    return costs + latencies  # Keep as is, return weighted sum


def evolutionary_state_estimation(swarm):
    position_std = np.mean(np.std(swarm.position, axis=0))
    max_std = np.sqrt(np.sum((swarm.options['bounds'][1] - swarm.options['bounds'][0]) ** 2))
    norm_diversity = position_std / max_std

    fitness = swarm.pbest_cost
    best_fitness = np.min(fitness)
    worst_fitness = np.max(fitness)
    if worst_fitness - best_fitness == 0:
        norm_fitness = 0
    else:
        norm_fitness = (fitness.mean() - best_fitness) / (worst_fitness - best_fitness)

    f = norm_fitness
    return f


def adaptive_parameters(swarm, f, delta):
    delta = delta * 0.5
    omega = 1 / (1 + 1.5 * np.exp(-2.6 * f))
    omega = np.clip(omega, 0.4, 0.9)
    swarm.options['w'] = omega

    c1 = swarm.options.get('c1', 2.0)
    c2 = swarm.options.get('c2', 2.0)

    if f < 0.2:
        state = 'convergence'
    elif f < 0.4:
        state = 'exploitation'
    elif f < 0.6:
        state = 'exploration'
    else:
        state = 'jumping_out'

    if state == 'exploration':
        c1 = c1 + delta
        c2 = c2 - delta
    elif state == 'exploitation':
        c1 = c1 + 0.5 * delta
        c2 = c2 - 0.5 * delta
    elif state == 'convergence':
        c1 = c1 + 0.5 * delta
        c2 = c2 + 0.5 * delta
    elif state == 'jumping_out':
        c1 = c1 - delta
        c2 = c2 + delta
        elitist_learning(swarm)

    c1 = np.clip(c1, 1.5, 2.5)
    c2 = np.clip(c2, 1.5, 2.5)

    c_sum = c1 + c2
    if c_sum > 4.0:
        c1 = c1 / c_sum * 4.0
        c2 = c2 / c_sum * 4.0
    elif c_sum < 3.0:
        c1 = c1 / c_sum * 3.0
        c2 = c2 / c_sum * 3.0

    swarm.options['c1'] = c1
    swarm.options['c2'] = c2


def elitist_learning(swarm):
    learning_rate = 0.1
    perturbation = np.random.normal(0, learning_rate, swarm.dimensions)
    swarm.best_pos += perturbation


def adaptive_pso(n_particles, dimensions, options, bounds, iters, objective_func):
    swarm = Swarm(position=np.random.uniform(bounds[0], bounds[1], (n_particles, dimensions)),
                  velocity=np.zeros((n_particles, dimensions)),
                  options=options)
    swarm.options['bounds'] = bounds

    swarm.pbest_pos = np.copy(swarm.position)
    swarm.pbest_cost = np.full(n_particles, np.inf)

    swarm.best_pos = np.zeros(dimensions)
    swarm.best_cost = np.inf

    delta_min, delta_max = 0.05, 0.1

    for i in range(iters):
        swarm.current_cost = objective_func(swarm.position)

        mask = swarm.current_cost < swarm.pbest_cost
        swarm.pbest_pos[mask] = swarm.position[mask]
        swarm.pbest_cost[mask] = swarm.current_cost[mask]

        best_cost_arg = np.argmin(swarm.pbest_cost)
        if swarm.pbest_cost[best_cost_arg] < swarm.best_cost:
            swarm.best_cost = swarm.pbest_cost[best_cost_arg]
            swarm.best_pos = swarm.pbest_pos[best_cost_arg].copy()

        f = evolutionary_state_estimation(swarm)
        delta = np.random.uniform(delta_min, delta_max)
        adaptive_parameters(swarm, f, delta)

        r1 = np.random.rand(n_particles, dimensions)
        r2 = np.random.rand(n_particles, dimensions)
        cognitive = swarm.options['c1'] * r1 * (swarm.pbest_pos - swarm.position)
        social = swarm.options['c2'] * r2 * (swarm.best_pos - swarm.position)
        swarm.velocity = swarm.options['w'] * swarm.velocity + cognitive + social

        swarm.position += swarm.velocity
        swarm.position = np.clip(swarm.position, bounds[0], bounds[1])

    return swarm.best_cost, swarm.best_pos


def test_apso(device_workloads, iterations):
    runtimes = []
    best_costs = []
    all_costs = []
    all_latencies = []

    for _ in range(iterations):
        start_time = time.time()
        n_particles = 150       ###############################改
        dimensions = n_devices
        options = {'c1': 2.0, 'c2': 2.0, 'w': 0.9}
        bounds = (np.zeros(n_devices), (n_servers - 1) * np.ones(n_devices))

        def objective_wrapper(pos):
            costs, latencies = calculate_total_cost(device_workloads, pos, data_size, network_speed,
                                                    completion_requirement, server_speed, server_cost,
                                                    server_ram, ram_requirement, m, n)
            all_costs.extend(costs)
            all_latencies.extend(latencies)
            return costs + latencies

        # 将 iters=50 改为 iters=20 来匹配 3000 次评估
        # 因为 20 × 150 = 3000
        best_cost, _ = adaptive_pso(n_particles, dimensions, options, bounds, iters=20,  ###############################改iters=17（总数3000）
                                    objective_func=objective_wrapper)
        end_time = time.time()
        runtimes.append(end_time - start_time)
        best_costs.append(best_cost)

    return best_costs, all_costs, all_latencies


# def find_pareto_front(costs, latencies):
#     pareto_front = []
#     for i in range(len(costs)):
#         dominated = False
#         for j in range(len(costs)):
#             if (costs[j] < costs[i] and latencies[j] <= latencies[i]) or \
#                     (costs[j] <= costs[i] and latencies[j] < latencies[i]):
#                 dominated = True
#                 break
#         if not dominated:
#             pareto_front.append((costs[i], latencies[i]))
#     return np.array(pareto_front)

def find_pareto_front(costs, latencies):
    """改进的帕累托前沿查找函数"""
    points = np.column_stack((costs, latencies))
    pareto_front = []

    for i in range(len(points)):
        dominated = False
        for j in range(len(points)):
            if i == j:
                continue
            # 检查是否被支配
            if (points[j, 0] <= points[i, 0] and points[j, 1] <= points[i, 1]) and \
                    (points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1]):
                dominated = True
                break
        if not dominated:
            pareto_front.append(points[i])

    pareto_front = np.array(pareto_front)

    # 按成本排序并确保帕累托前沿是单调的
    if len(pareto_front) > 0:
        pareto_front = pareto_front[pareto_front[:, 0].argsort()]

        # 确保帕累托前沿是单调递减的（对于延迟）
        filtered_front = [pareto_front[0]]
        min_latency = pareto_front[0, 1]

        for i in range(1, len(pareto_front)):
            if pareto_front[i, 1] < min_latency:
                filtered_front.append(pareto_front[i])
                min_latency = pareto_front[i, 1]

        pareto_front = np.array(filtered_front)

    return pareto_front


# Main execution
device_workloads = generate_data()
iterations = 1    ##### 改为1
best_costs, all_costs, all_latencies = test_apso(device_workloads, iterations)

# Calculate average best cost
average_best_cost = np.mean(best_costs)

# Plotting the results
plt.figure(figsize=(18, 6))

# Plot 1: Best cost per run
plt.subplot(1, 3, 1)
plt.plot(range(1, iterations + 1), best_costs, marker='o', linestyle='-', color='b', label='Best Cost')
plt.axhline(y=average_best_cost, color='r', linestyle='--', label='Average Best Cost')
plt.text(iterations, average_best_cost + 0.5, f'Average: {average_best_cost:.2f}', color='r',
         verticalalignment='bottom', horizontalalignment='left', fontsize=10)
plt.xlabel('Run Number')
plt.ylabel('Best Cost')
plt.title('Best Cost in Each APSO Run')
plt.xticks(range(1, iterations + 1))
plt.legend()
plt.grid(True)

# Plot 2: Cost vs. Latency scatter plot
plt.subplot(1, 3, 2)
plt.scatter(all_costs, all_latencies, alpha=0.5, c='blue')
plt.xlabel('Cost ($)')
plt.ylabel('Latency (seconds)')
plt.title('Cost vs. Latency Scatter Plot')
plt.grid(True)

# Plot 3: Pareto front
plt.subplot(1, 3, 3)
pareto_front = find_pareto_front(all_costs, all_latencies)

if len(pareto_front) >= 2:
    pareto_front = pareto_front[pareto_front[:, 0].argsort()]
    plt.plot(pareto_front[:, 0], pareto_front[:, 1], 'r-', lw=2, label='Pareto Front')
elif len(pareto_front) == 1:
    # 只有一个帕累托点，显示为单个点
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='red', s=100,
                marker='*', label='Pareto Point')
else:
    # 没有帕累托点，显示最佳成本和解
    if len(all_costs) > 0:
        min_cost_idx = np.argmin(all_costs)
        min_latency_idx = np.argmin(all_latencies)
        extreme_points = np.array([
            [all_costs[min_cost_idx], all_latencies[min_cost_idx]],
            [all_costs[min_latency_idx], all_latencies[min_latency_idx]]
        ])
        extreme_points = extreme_points[extreme_points[:, 0].argsort()]
        plt.plot(extreme_points[:, 0], extreme_points[:, 1], 'r--', lw=2,
                label='Extreme Solutions')

plt.scatter(all_costs, all_latencies, alpha=0.3, c='blue', label='All Solutions')
plt.xlabel('Cost ($)')
plt.ylabel('Latency (seconds)')
plt.title('Pareto Front')
plt.legend()
plt.grid(True)
# plt.subplot(1, 3, 3)
# pareto_front = find_pareto_front(all_costs, all_latencies)
# pareto_front = pareto_front[pareto_front[:, 0].argsort()]
#
# plt.scatter(all_costs, all_latencies, alpha=0.3, c='blue', label='All Solutions')
# plt.plot(pareto_front[:, 0], pareto_front[:, 1], 'r-', lw=2, label='Pareto Front')
# plt.xlabel('Cost ($)')
# plt.ylabel('Latency (seconds)')
# plt.title('Pareto Front')
# plt.legend()
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()


def test_weight_combinations(device_workloads, iterations=5):
    # 定义颜色方案
    colors = ['blue', 'green', 'red']  # 对应三种权重组合

    weight_combinations = [
        (10, 0.01),  # 强成本导向（蓝色）
        (1, 1),  # 平衡权重（绿色）
        (0.01, 10)  # 强延迟导向（红色）
    ]

    plt.figure(figsize=(15, 5))

    for idx, (weight_m, weight_n) in enumerate(weight_combinations, 1):
        # 修改目标函数
        def evaluate_unified(pos):
            if pos.ndim == 1:
                pos = pos.reshape(1, -1)
            costs, latencies = calculate_total_cost(
                device_workloads, pos, data_size, network_speed,
                completion_requirement, server_speed, server_cost,
                server_ram, ram_requirement, 1, 1  # 统一使用权重1,1
            )
            return costs + latencies

        # 运行PSO
        options = {'c1': 2.0, 'c2': 2.0, 'w': 0.9}
        bounds = (np.zeros(n_devices), (n_servers - 1) * np.ones(n_devices))
        _, best_pos = adaptive_pso(150, n_devices, options, bounds, 20, evaluate_unified) #################改180、17（总数3000）

        # 生成测试解集
        test_positions = np.clip(
            best_pos + np.random.normal(0, 0.5, (200, n_devices)),
            0, n_servers - 1
        )

        # 计算可视化数据（统一权重1,1）
        costs, latencies = calculate_total_cost(
            device_workloads, test_positions, data_size, network_speed,
            completion_requirement, server_speed, server_cost,
            server_ram, ram_requirement, 1, 1
        )

        # 绘图（添加颜色参数）
        plt.subplot(1, 3, idx)
        plt.scatter(
            costs, latencies,
            alpha=0.5,
            c=colors[idx - 1],  # 使用预定义颜色
            label=f'All Solutions (m={weight_m}, n={weight_n})'
        )

        # Pareto前沿（保持红色）
        pareto_front = find_pareto_front(costs, latencies)
        plt.plot(pareto_front[:, 0], pareto_front[:, 1], 'r-', label='Pareto Front')

        plt.xlabel('Cost (evaluated with m=1)')
        plt.ylabel('Latency (evaluated with n=1)')
        plt.title(f'Optimized with m={weight_m}, n={weight_n}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

# 使用方式
test_weight_combinations(device_workloads)


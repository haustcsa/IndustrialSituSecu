
import numpy as np
import pyswarms as ps
import time
import matplotlib.pyplot as plt

# Set random seed for consistency
np.random.seed(21)

# Number of servers and devices
n_servers = 20
n_devices = 250

# Simulation parameters (consistent)
network_speed = np.random.uniform(100, 900, n_devices)
server_cost = np.random.uniform(0.01, 0.1, n_servers)
server_speed = np.random.uniform(50, 200, n_servers)
server_ram = np.random.uniform(2, 8, n_servers)  # GB
data_size = np.random.uniform(50, 150, n_devices)  # MB
completion_requirement = np.random.uniform(20, 40, n_devices)  # MI
ram_requirement = np.random.uniform(1, 2, n_devices)  # GB

# Weights for the cost and latency components
m = 10  # cost weight
n = 1e-2  # latency weight
# m = 1
# n = 1
# m = 1e-2
# n = 10


# Function to generate consistent workloads and server capacities
def generate_data():
    device_workloads = np.random.uniform(1e6, 1e8, n_devices)
    return device_workloads


def find_pareto_front(costs, latencies):
    """找出Pareto前沿"""
    pareto_front = []
    for i in range(len(costs)):
        dominated = False
        for j in range(len(costs)):
            if (costs[j] < costs[i] and latencies[j] <= latencies[i]) or \
                    (costs[j] <= costs[i] and latencies[j] < latencies[i]):
                dominated = True
                break
        if not dominated:
            pareto_front.append((costs[i], latencies[i]))
    return np.array(pareto_front)


def calculate_raw_values(positions, device_workloads):
    """计算原始成本和延迟（不加权），添加惩罚项"""
    raw_costs = np.zeros(positions.shape[0])
    raw_latencies = np.zeros(positions.shape[0])
    penalty = 1000  # 惩罚项

    for i in range(positions.shape[0]):
        particle = positions[i]
        total_cost = 0
        total_latency = 0

        for j in range(n_devices):
            server_index = int(particle[j]) % n_servers

            # 检查RAM约束
            if ram_requirement[j] <= server_ram[server_index]:
                # 正常计算
                transfer_time = data_size[j] / network_speed[j]
                processing_time = completion_requirement[j] / server_speed[server_index]
                total_time = transfer_time + processing_time
                current_cost = server_cost[server_index] * total_time
                total_cost += current_cost
                total_latency += total_time
            else:
                # 添加惩罚
                total_cost += penalty
                total_latency += penalty

        raw_costs[i] = total_cost
        raw_latencies[i] = total_latency

    return raw_costs, raw_latencies


def custom_objective(positions, device_workloads, m, n):
    """带权重的目标函数，包含惩罚项"""
    raw_costs, raw_latencies = calculate_raw_values(positions, device_workloads)
    return m * raw_costs + n * raw_latencies


def run_pso_with_weights(device_workloads, m, n):
    """使用特定权重运行PSO并返回100个解（基于最佳解的变异）"""
    optimizer = ps.single.GlobalBestPSO(
        n_particles=180,  # 保持20个粒子
        dimensions=n_devices,
        options={'c1': 2, 'c2': 2, 'w': 0.9},
        bounds=(np.zeros(n_devices), (n_servers - 1) * np.ones(n_devices))
    )

    # 运行优化（使用自定义权重）
    cost, best_pos = optimizer.optimize(
        lambda x: custom_objective(x, device_workloads, m, n),
        iters=17
    )

    # 围绕最佳解生成100个变异解
    # 使用高斯噪声围绕最佳位置生成变异解
    noise_std = 0.5  # 噪声标准差
    num_variations = 300  # 生成100个变异解

    # 生成变异解
    variations = np.clip(
        best_pos + np.random.normal(0, noise_std, (num_variations, n_devices)),
        0, n_servers - 1
    )

    return variations


# 主程序
device_workloads = generate_data()

# 定义权重组合
weight_combinations = [
    (10, 0.01),  # 强成本导向
    (1, 1),  # 平衡权重
    (0.01, 10)  # 强延迟导向
]

colors = ['blue', 'green', 'red']

plt.figure(figsize=(18, 6))

for idx, (m_val, n_val) in enumerate(weight_combinations, 1):
    # 为每组权重运行PSO并获取100个解
    positions = run_pso_with_weights(device_workloads, m_val, n_val)

    # 用统一标准(m=1, n=1权重)评估这些解
    costs, latencies = calculate_raw_values(positions, device_workloads)

    # 应用统一权重标准
    unified_costs = 1 * costs  # m=1
    unified_latencies = 1 * latencies  # n=1

    # 绘制结果
    plt.subplot(1, 3, idx)
    plt.scatter(
        unified_costs, unified_latencies,
        alpha=0.5,
        label='All Solutions',
        c=colors[idx - 1]
    )

    # 计算Pareto前沿
    pareto_front = find_pareto_front(unified_costs, unified_latencies)
    if len(pareto_front) > 0:
        pareto_front = pareto_front[pareto_front[:, 0].argsort()]
        plt.plot(
            pareto_front[:, 0], pareto_front[:, 1],
            'r-',
            label='Pareto Front'
        )

    plt.xlabel('Cost (evaluated with m=1)')
    plt.ylabel('Latency (evaluated with n=1)')
    plt.title(f'Optimized with m={m_val}, n={n_val}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()



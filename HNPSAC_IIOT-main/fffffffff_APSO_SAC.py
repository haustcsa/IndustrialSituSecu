import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import deque

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
# m = 1e-2    ####### 最佳粒子数150
# n = 10
m = 10  ####### 最佳粒子数180
n = 1e-2
# m = 1    ####### 最佳粒子数150
# n = 1


def calculate_total_cost(pos):
    total_cost = 0
    total_latency = 0
    for j in range(n_devices):
        s = int(np.clip(pos[j], 0, n_servers - 1))
        if ram_requirement[j] <= server_ram[s]:
            trans = data_size[j] / network_speed[j]
            proc = completion_requirement[j] / server_speed[s]
            t = trans + proc
            cost = server_cost[s] * t
            total_cost += m * cost
            total_latency += n * t
    return total_cost, total_latency


# ========================= APSO 部分 =========================
from pyswarms.backend.swarms import Swarm


def evolutionary_state_estimation(swarm):
    std = np.mean(np.std(swarm.position, axis=0))
    max_std = np.sqrt(np.sum((swarm.options['bounds'][1] - swarm.options['bounds'][0]) ** 2))
    diversity = std / max_std
    fitness = swarm.pbest_cost
    best = np.min(fitness)
    worst = np.max(fitness)
    norm_fitness = 0 if worst == best else (fitness.mean() - best) / (worst - best)
    return norm_fitness


def adaptive_parameters(swarm, f, delta):
    omega = 1 / (1 + 1.5 * np.exp(-2.6 * f))
    omega = np.clip(omega, 0.4, 0.9)
    swarm.options['w'] = omega
    c1, c2 = swarm.options.get('c1', 2.0), swarm.options.get('c2', 2.0)
    if f < 0.2:
        c1 += 0.5 * delta
        c2 += 0.5 * delta
    elif f < 0.4:
        c1 += 0.5 * delta
        c2 -= 0.5 * delta
    elif f < 0.6:
        c1 += delta
        c2 -= delta
    else:
        c1 -= delta
        c2 += delta
        elitist_learning(swarm)
    c1 = np.clip(c1, 1.5, 2.5)
    c2 = np.clip(c2, 1.5, 2.5)
    if c1 + c2 > 4.0:
        c1 = c1 / (c1 + c2) * 4.0
        c2 = c2 / (c1 + c2) * 4.0
    elif c1 + c2 < 3.0:
        c1 = c1 / (c1 + c2) * 3.0
        c2 = c2 / (c1 + c2) * 3.0
    swarm.options['c1'] = c1
    swarm.options['c2'] = c2


def elitist_learning(swarm):
    perturbation = np.random.normal(0, 0.1, swarm.dimensions)
    swarm.best_pos += perturbation


def adaptive_pso(n_particles, dimensions, options, bounds, iters):
    def objective_func(pos):
        costs = []
        for p in pos:
            cost, _ = calculate_total_cost(p)
            costs.append(cost)
        return np.array(costs)

    swarm = Swarm(
        n_particles=n_particles,
        dimensions=dimensions,
        position=np.random.uniform(bounds[0], bounds[1], (n_particles, dimensions)),
        velocity=np.zeros((n_particles, dimensions)),
        options=options
    )
    swarm.options['bounds'] = bounds
    swarm.pbest_pos = np.copy(swarm.position)
    swarm.pbest_cost = np.full(n_particles, np.inf)
    swarm.best_pos = np.zeros(dimensions)
    swarm.best_cost = np.inf
    for i in range(iters):
        swarm.current_cost = objective_func(swarm.position)
        mask = swarm.current_cost < swarm.pbest_cost
        swarm.pbest_pos[mask] = swarm.position[mask]
        swarm.pbest_cost[mask] = swarm.current_cost[mask]
        best_idx = np.argmin(swarm.pbest_cost)
        if swarm.pbest_cost[best_idx] < swarm.best_cost:
            swarm.best_cost = swarm.pbest_cost[best_idx]
            swarm.best_pos = swarm.pbest_pos[best_idx].copy()
        f = evolutionary_state_estimation(swarm)
        adaptive_parameters(swarm, f, np.random.uniform(0.05, 0.1))
        r1 = np.random.rand(n_particles, dimensions)
        r2 = np.random.rand(n_particles, dimensions)
        cognitive = swarm.options['c1'] * r1 * (swarm.pbest_pos - swarm.position)
        social = swarm.options['c2'] * r2 * (swarm.best_pos - swarm.position)
        swarm.velocity = swarm.options['w'] * swarm.velocity + cognitive + social
        swarm.position += swarm.velocity
        swarm.position = np.clip(swarm.position, bounds[0], bounds[1])
    return swarm.best_cost, swarm.best_pos


# ========================= 真正的SAC实现 =========================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 重参数化技巧
        action = torch.tanh(x_t)
        # 计算log概率（考虑tanh变换的Jacobian）
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


def sac_finetune(init_pos, lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
                 batch_size=256, total_steps=100000, tau=5e-3, gamma=0.99):
    device = torch.device("cpu")

    # 初始化网络
    actor = Actor(n_devices, n_devices).to(device)
    critic_1 = Critic(n_devices, n_devices).to(device)
    critic_2 = Critic(n_devices, n_devices).to(device)
    target_critic_1 = Critic(n_devices, n_devices).to(device)
    target_critic_2 = Critic(n_devices, n_devices).to(device)

    target_critic_1.load_state_dict(critic_1.state_dict())
    target_critic_2.load_state_dict(critic_2.state_dict())

    # 温度参数α（SAC核心组件）
    target_entropy = -torch.prod(torch.tensor([n_devices])).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp()

    # 优化器（包含α优化器）
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=lr_critic)
    alpha_optimizer = optim.Adam([log_alpha], lr=lr_alpha)  # 温度参数学习率

    # 经验回放池
    replay_buffer = deque(maxlen=1000000)  # 1,000,000

    def add_to_buffer(state, action, reward, next_state, done):
        replay_buffer.append((state, action, reward, next_state, done))

    # 初始化状态
    state = torch.tensor(init_pos / (n_servers - 1), dtype=torch.float32).unsqueeze(0).to(device)
    best_cost, _ = calculate_total_cost(init_pos)
    best_action = init_pos.copy()

    step = 0
    cost_history = []

    print("开始SAC训练...")

    while step < total_steps:
        # 选择动作（使用随机策略）
        with torch.no_grad():
            action, _ = actor.sample(state)  # 使用sample方法而不是直接forward
            action_scaled = torch.clamp(action, 0, 1) * (n_servers - 1)

        # 计算奖励
        cost, latency = calculate_total_cost(action_scaled.cpu().numpy().flatten())
        reward = -cost / 1000  # 奖励缩放
        next_state = state  # 简单状态转移

        # 存储经验（修复dones数据类型）
        add_to_buffer(state, action_scaled, torch.tensor(reward, dtype=torch.float32),
                      next_state, torch.tensor(False, dtype=torch.float32))  # 改为float32

        # 更新网络
        if len(replay_buffer) >= batch_size:
            # 采样批次
            indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
            batch = [replay_buffer[i] for i in indices]
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.cat(states).to(device)
            actions = torch.cat(actions).to(device)
            rewards = torch.stack(rewards).unsqueeze(1).to(device)
            next_states = torch.cat(next_states).to(device)
            dones = torch.stack(dones).unsqueeze(1).to(device)

            # Critic更新（修复dones计算）
            with torch.no_grad():
                next_actions, next_log_probs = actor.sample(next_states)
                next_actions_scaled = torch.clamp(next_actions, 0, 1) * (n_servers - 1)
                target_q1 = target_critic_1(next_states, next_actions_scaled)
                target_q2 = target_critic_2(next_states, next_actions_scaled)
                target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs  # SAC核心：包含熵项
                # 修复：将dones转换为float进行计算
                target = rewards + gamma * target_q * (1 - dones.float())

            q1 = critic_1(states, actions)
            q2 = critic_2(states, actions)
            critic_loss = nn.MSELoss()(q1, target) + nn.MSELoss()(q2, target)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor更新（SAC风格）
            actions_pred, log_probs = actor.sample(states)
            actions_scaled_pred = torch.clamp(actions_pred, 0, 1) * (n_servers - 1)
            q1_pi = critic_1(states, actions_scaled_pred)
            q2_pi = critic_2(states, actions_scaled_pred)
            min_q_pi = torch.min(q1_pi, q2_pi)

            # SAC的Actor损失：Q值最小化 + 熵最大化
            actor_loss = (alpha.detach() * log_probs - min_q_pi).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # 温度参数α更新
            alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            alpha = log_alpha.exp()

            # 目标网络软更新
            for target_param, param in zip(target_critic_1.parameters(), critic_1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for target_param, param in zip(target_critic_2.parameters(), critic_2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # 更新最佳解
        if cost < best_cost:
            best_cost = cost
            best_action = action_scaled.cpu().numpy().flatten()

        step += 1

        if step % 10000 == 0:
            print(f"步骤 {step}/{total_steps}, 最佳成本: {best_cost:.2f}")
            cost_history.append(best_cost)

    print(f"SAC训练完成，最终最佳成本: {best_cost:.2f}")
    return best_cost, best_action

# ========================= 修改后的帕累托前沿函数 =========================
def find_pareto_front(costs, latencies):
    """改进的帕累托前沿计算，匹配第一段代码样式"""
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

def calculate_raw_values(pos):
    """计算原始成本和延迟（不使用权重），用于公平比较"""
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
            total_cost += cost  # 原始成本
            total_latency += t  # 原始延迟
        else:
            total_cost += penalty
            total_latency += penalty

    return total_cost, total_latency

def test_weight_combinations(iterations=1):
    """修改后的测试函数，确保评估一致性"""
    weight_combinations = [
        (10, 0.01, 'blue', 'Cost-oriented'),
        (1, 1, 'green', 'Balanced'),
        (0.01, 10, 'red', 'Latency-oriented')
    ]

    plt.figure(figsize=(18, 6))

    # 存储所有点用于总体帕累托前沿
    all_costs = []
    all_latencies = []

    for idx, (m_val, n_val, color, label) in enumerate(weight_combinations, 1):
        print(f"\n测试权重组合 {idx}: m={m_val}, n={n_val} ({label})")

        costs = []
        latencies = []
        current_best_cost = float('inf')
        current_best_solution = None

        for iter_num in range(iterations):
            print(f"  迭代 {iter_num + 1}/{iterations}")

            # 运行APSO - 使用当前权重组合
            apso_cost, apso_pos = adaptive_pso(
                n_particles=180,  ###############改
                dimensions=n_devices,
                options={'c1': 2.0, 'c2': 2.0, 'w': 0.9},
                bounds=(np.zeros(n_devices), (n_servers - 1) * np.ones(n_devices)),
                iters=17  # 150粒子 × 20迭代 = 3000次评估
            )

            # 运行SAC - 使用当前权重组合
            sac_cost, sac_pos = sac_finetune_enhanced(
                apso_pos,
                lr_actor=3e-4,
                lr_critic=3e-4,
                lr_alpha=3e-4,
                total_steps=8000
            )

            # 记录最佳解
            if sac_cost < current_best_cost:
                current_best_cost = sac_cost
                current_best_solution = sac_pos.copy()

            # 生成测试样本
            test_positions = []
            for _ in range(50):
                noise = np.random.normal(0, 0.3, n_devices)
                noisy_pos = sac_pos + noise
                test_positions.append(np.clip(noisy_pos, 0, n_servers - 1))
            test_positions = np.array(test_positions)

            # 关键修改：使用统一权重(m=1,n=1)评估所有解，确保公平比较
            for pos in test_positions:
                # 使用固定权重m=1,n=1计算原始成本和延迟
                cost_val, latency_val = calculate_raw_values(pos)
                costs.append(cost_val)
                latencies.append(latency_val)

        costs = np.array(costs)
        latencies = np.array(latencies)

        print(f"  生成 {len(costs)} 个样本点")
        print(f"  最佳加权成本: {current_best_cost:.2f}")

        # 收集所有点
        all_costs.extend(costs)
        all_latencies.extend(latencies)

        # 绘制子图
        plt.subplot(1, 3, idx)
        plt.scatter(costs, latencies, alpha=0.6, c=color, marker='o',
                    label=f'Solutions (m={m_val}, n={n_val})')

        # 计算并绘制帕累托前沿
        pareto_front = find_pareto_front(costs, latencies)
        print(f"  找到 {len(pareto_front)} 个帕累托前沿点")

        if len(pareto_front) > 1:
            try:
                from scipy.spatial import ConvexHull
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
        plt.title(f'Optimized with m={m_val}, n={n_val}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('enhanced_apso_sac_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制总体帕累托前沿
    plt.figure(figsize=(10, 8))
    plt.scatter(all_costs, all_latencies, alpha=0.5, c='blue', marker='o', label='All Solutions')

    overall_pareto = find_pareto_front(np.array(all_costs), np.array(all_latencies))
    print(f"\n总体帕累托前沿点数量: {len(overall_pareto)}")

    if len(overall_pareto) > 1:
        overall_pareto = overall_pareto[overall_pareto[:, 0].argsort()]
        plt.plot(overall_pareto[:, 0], overall_pareto[:, 1], 'r-',
                 linewidth=3, label='Overall Pareto Front')

    plt.xlabel('Cost (evaluated with m=1)')
    plt.ylabel('Latency (evaluated with n=1)')
    plt.title('Overall Pareto Front Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('overall_apso_sac_pareto.png', dpi=300, bbox_inches='tight')
    plt.show()


# ========================= 修改主运行函数以匹配样式 =========================
def run_apso_sac(runs=1):
    """修改后的主运行函数，匹配第一段代码的输出样式"""
    print("开始APSO+SAC优化...")
    best_costs = []
    cost_history = []

    for run in range(runs):
        print(f"\n运行 {run + 1}/{runs}")

        # APSO初始化
        apso_cost, apso_pos = adaptive_pso(
            n_particles=180,   ############改
            dimensions=n_devices,
            options={'c1': 2.0, 'c2': 2.0, 'w': 0.9},
            bounds=(np.zeros(n_devices), (n_servers - 1) * np.ones(n_devices)),
            iters=25   # 150粒子 × 30迭代 = 4500次评估
        )
        print(f"APSO初始成本: {apso_cost:.2f}")

        # SAC微调
        sac_cost, sac_pos = sac_finetune_enhanced(
            apso_pos,
            lr_actor=3e-4,
            lr_critic=3e-4,
            lr_alpha=3e-4,
            total_steps=20000
        )
        print(f"SAC优化后成本: {sac_cost:.2f}")

        best_costs.append(sac_cost)
        cost_history.append(sac_cost)

    # 绘制训练过程图 - 匹配第一段代码的样式
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(cost_history) + 1), cost_history, marker='o', linewidth=2)
    plt.xlabel('Run Number')
    plt.ylabel('Best Cost')
    plt.title('APSO + SAC Optimization Process')
    plt.grid(True)
    plt.savefig('apso_sac_training_process.png', dpi=300, bbox_inches='tight')
    plt.show()

    return best_costs


# ========================= 修改calculate_total_cost函数以支持权重参数 =========================
def calculate_total_cost(pos, weight_m=None, weight_n=None):
    """统一的成本计算函数，匹配第一段代码的逻辑"""
    # 使用传入的权重或默认的全局权重
    current_m = weight_m if weight_m is not None else m
    current_n = weight_n if weight_n is not None else n

    total_cost = 0
    total_latency = 0
    penalty = 1000  # 保持惩罚项，但不涉及权重

    for j in range(n_devices):
        s = int(np.clip(pos[j], 0, n_servers - 1))
        if ram_requirement[j] <= server_ram[s]:
            trans = data_size[j] / network_speed[j]
            proc = completion_requirement[j] / server_speed[s]
            t = trans + proc
            cost = server_cost[s] * t
            total_cost += cost  # 原始成本，不加权重
            total_latency += t  # 原始延迟，不加权重
        else:
            total_cost += penalty
            total_latency += penalty

    # 在最后统一应用权重，匹配第一段代码的公式
    weighted_total = current_m * total_cost + current_n * total_latency
    return weighted_total, total_latency  # 返回加权总和和原始延迟用于分析


# ========================= 添加局部搜索优化函数 =========================
def local_search_optimization(initial_pos, iterations=1000):
    """添加局部搜索优化，匹配第一段代码的功能"""
    current_pos = initial_pos.copy()
    current_cost, _ = calculate_total_cost(current_pos)
    best_pos = current_pos.copy()
    best_cost = current_cost

    print(f"Local search initial cost: {current_cost:.2f}")

    for i in range(iterations):
        new_pos = current_pos.copy()

        # 随机调整一个设备
        device_idx = np.random.randint(0, n_devices)
        new_server = np.random.randint(0, n_servers)
        new_pos[device_idx] = new_server

        new_cost, _ = calculate_total_cost(new_pos)

        if new_cost < current_cost:
            current_pos = new_pos.copy()
            current_cost = new_cost

            if new_cost < best_cost:
                best_pos = new_pos.copy()
                best_cost = new_cost

    print(f"Local search completed. Best cost: {best_cost:.2f}")
    return best_cost, best_pos


# ========================= 修改SAC训练函数以包含局部搜索 =========================
def sac_finetune_enhanced(init_pos, lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
                          batch_size=256, total_steps=20000, tau=5e-3, gamma=0.99):
    """增强版SAC训练，包含局部搜索"""
    # 先运行原始SAC训练
    sac_cost, sac_pos = sac_finetune(
        init_pos, lr_actor, lr_critic, lr_alpha, batch_size, total_steps, tau, gamma
    )

    # 添加局部搜索优化
    print("Starting local search optimization...")
    final_cost, final_pos = local_search_optimization(sac_pos, iterations=1000)

    return final_cost, final_pos


if __name__ == "__main__":
    # 运行主优化
    best_costs = run_apso_sac(runs=5)
    average_cost = np.mean(best_costs)

    print(f"\n平均最佳成本: {average_cost:.2f}")

    # 测试不同权重组合
    print("\n开始测试不同权重组合...")
    test_weight_combinations(iterations=2)
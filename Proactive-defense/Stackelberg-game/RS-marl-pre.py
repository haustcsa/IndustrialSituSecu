"""
RS-SQL: 基于角色切换随机–Stackelberg 的多智能体强化学习算法

- 状态 s: 聚合情境（攻击图节点）
- 阶段 1 (真实博弈): 攻击者为 Leader, 防御者为 Follower
- 阶段 2 (预测博弈): 防御者在预测态势下为 Leader, 预部署防御
- 预测命中 + 预部署有效: 攻击被提前拦截, s_{t+1} = s_t, 防御成本打折
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import random


# =========================
# 1. 攻击图 + 环境定义
# =========================

@dataclass
class Edge:
    """
    攻击图中的一条边:
    src --(vuln)--> dst
    vuln 这里只是一个标签, 方便扩展, 算法本身只用到 src/dst
    """
    src: int
    dst: int
    vuln: int


class ICSGameEnv:
    """
    工业网络攻防环境 (无状态分解版本):

    - 状态 s ∈ {0,1,...,S-1} 对应情境状态(攻击图节点/聚合态势)
    - 攻击动作 a:
        0..(E-1): 选择一条攻击边 Edge
        E:        no-op (不发起攻击)
    - 防御动作 d:
        0..(S-1): 阻断对应目标状态
        S:        no-op (不防御)

    INSSP 基础转移 P0(s'|s) 用于预测:
        ŝ = argmax_s' P0(s'|s)
    真正转移由 attack + defense + 预部署 共同决定。
    """

    def __init__(
        self,
        num_states: int,
        edges: List[Edge],
        initial_state: int,
        terminal_states: List[int],
        attack_cost: float = 10.0,
        defense_cost: float = 10.0,
        reward_success: float = 100.0,
        predeploy_discount: float = 0.3,
    ):
        self.num_states = num_states
        self.edges = edges
        self.initial_state = initial_state
        self.terminal_states = set(terminal_states)

        # 成本 & 奖励参数
        self.attack_cost_base = attack_cost          # 一次真实攻击成本
        self.defense_cost_base = defense_cost        # 一次实时防御成本
        self.reward_success = reward_success         # 攻击成功奖励/损失
        self.predeploy_discount = predeploy_discount # 预部署时的成本折扣系数 η

        # 每个状态有哪些出边 (索引用 attack_actions 的 index)
        self.state_out_edges: Dict[int, List[int]] = {s: [] for s in range(num_states)}
        for idx, e in enumerate(edges):
            self.state_out_edges[e.src].append(idx)

        # 攻击动作空间: 每条边一个动作 + 1 个 no-op
        self.attack_edges: List[Edge] = edges[:]     # index -> Edge
        self.num_attack_actions = len(self.attack_edges) + 1
        self.attack_noop = self.num_attack_actions - 1

        # 防御动作空间: 阻断目标状态 + 1 个 no-op
        self.num_defense_actions = num_states + 1
        self.defense_noop = self.num_defense_actions - 1

        # ===== INSSP 基础转移矩阵 P0(s'|s) =====
        # 这里只做 demo: 对出边目标状态均匀分布; 没有出边则自环
        self.P0 = np.zeros((num_states, num_states))
        for s in range(num_states):
            outs = self.state_out_edges[s]
            if outs:
                prob = 1.0 / len(outs)
                for idx in outs:
                    e = self.attack_edges[idx]
                    self.P0[s, e.dst] = prob
            else:
                self.P0[s, s] = 1.0

    # ---------- 基本接口 ----------

    def reset(self) -> int:
        """复位环境到初始情境状态 s0"""
        return self.initial_state

    def predict_next_state(self, s: int) -> int:
        """
        INSSP 预测:
        ŝ = argmax_s' P0(s'|s)
        实际应用中可用训练好的 MC/INSSP 模型替换这里的 P0。
        """
        probs = self.P0[s]
        return int(np.argmax(probs))

    # ---------- 双阶段博弈的一步演化 ----------

    def step(
        self,
        s: int,
        a_real: int,
        d_real: int,
        s_pred: int,
        d_pre: int,
    ) -> Tuple[int, float, float, bool]:
        """
        执行一次“双阶段博弈”:

        输入:
          - s: 当前真实状态
          - a_real: 真实态势下攻击者选择的攻击动作
          - d_real: 真实态势下防御者选择的实时防御动作
          - s_pred: INSSP 预测的下一状态 ŝ
          - d_pre: 预测态势下、防御者作为 Leader 选择的预部署防御动作

        输出:
          - s_next: 真实下一状态
          - R_A: 攻击者即时奖励
          - R_D: 防御者即时奖励
          - done: 是否到达终端
        """

        # ===== 阶段 1: 真实攻击推进(不考虑预部署) =====
        if a_real == self.attack_noop:
            chosen_edge = None
        else:
            chosen_edge = self.attack_edges[a_real]

        s_raw = s
        attack_used = False

        # 只有当选中的边确实从当前状态出发, 才算一次有效攻击推动
        if chosen_edge is not None and chosen_edge.src == s:
            s_raw = chosen_edge.dst
            attack_used = True

        # ===== 预部署成本 =====
        # 预部署总是要付出一笔成本, 无论命中与否
        C_pre = 0.0
        if d_pre != self.defense_noop:
            # 预部署的基准成本按 η 折扣 (可以理解为“更廉价的提前加固”)
            C_pre = self.predeploy_discount * self.defense_cost_base

        # ===== 判断预部署是否命中 =====
        predeploy_hit = (
            (s_raw == s_pred)                # 预测状态与真实候选后继一致
            and (d_pre != self.defense_noop) # 确实部署了防御
            and (d_pre == s_raw)             # 防御动作针对的就是该状态
        )

        # ===== 实时防御 =====
        C_real = 0.0
        defense_hit_real = False

        if predeploy_hit:
            # 预测命中 + 预部署拦截成功:
            # 直接阻断, 攻击者不前移
            s_next = s
            attack_success = False
        else:
            # 预测失败或预部署无效: 才需要实时防御 d_real
            if d_real != self.defense_noop:
                C_real = self.defense_cost_base
                defense_hit_real = (s_raw == d_real)

            if defense_hit_real:
                # 实时防御拦截成功
                s_next = s
                attack_success = False
            else:
                # 没挡住, 攻击按 s_raw 前进
                s_next = s_raw
                attack_success = attack_used

        # ===== 成本与奖励 =====
        # 攻击成本: 只要发起了有效攻击就记一次
        C_A = self.attack_cost_base if attack_used else 0.0
        # 防御成本: 预部署成本 + 实时防御成本
        C_D = C_pre + C_real

        R_A = -C_A
        R_D = -C_D

        done = s_next in self.terminal_states
        if done and attack_success:
            # 攻击成功到达终端: 攻击者+R, 防御者-R
            R_A += self.reward_success
            R_D -= self.reward_success

        return s_next, R_A, R_D, done


# ================================
# 2. 角色切换随机–Stackelberg MARL
# ================================

class RSStackelbergMARL:
    """
    角色切换随机–Stackelberg Q-learning (RS-SQL):

    - 使用联合 Q_A(s,a,d), Q_D(s,a,d) 近似角色切换随机–Stackelberg 均衡
    - 阶段 1: 真实态势 s 上, 攻击者为 Leader
        * 防御者给出 best-response d_BR(s,a)
        * 攻击者选择使 Q_A 最大的 a_R^*(s)
    - 阶段 2: 预测态势 ŝ 上, 防御者为 Leader
        * 攻击者给出 best-response a_BR(ŝ,d)
        * 防御者选择使 Q_D 最小的 d_P^*(ŝ)
    - 预测命中 + 预部署有效: 攻击阻断, s_{t+1}=s_t
    """

    def __init__(
        self,
        env: ICSGameEnv,
        gamma: float = 0.9,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        episodes: int = 1000,
    ):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.episodes = episodes

        S = env.num_states
        A = env.num_attack_actions
        D = env.num_defense_actions

        # 联合 Q 表: Q_A, Q_D ∈ R^{S × A × D}
        self.QA = np.random.randn(S, A, D) * 1e-3
        self.QD = np.random.randn(S, A, D) * 1e-3

        # 用于画初始状态的策略演变 (可选)
        self.policy_hist_A: List[np.ndarray] = []
        self.policy_hist_D: List[np.ndarray] = []

    # ---------- 阶段 1: 真实态势下攻击者为 Leader ----------

    def local_attack_leader(self, s: int):
        """
        在状态 s 上求局部 Stackelberg 解 (攻击者为 Leader):

        1) 防御者对每个 a 的最佳响应:
           d_BR(a) = argmin_d Q_D(s,a,d)
        2) 攻击者在此基础上选:
           a* = argmax_a Q_A(s,a,d_BR(a))
        """
        QA_s = self.QA[s]  # [A,D]
        QD_s = self.QD[s]  # [A,D]
        A, D = QA_s.shape

        # 防御者对每个 a 的 best response
        d_br = np.argmin(QD_s, axis=1)          # shape [A]
        idx = (np.arange(A), d_br)
        QA_lead = QA_s[idx]                     # shape [A], 对应 Q_A(s,a,d_BR(a))

        a_star = int(np.argmax(QA_lead))
        d_star = int(d_br[a_star])

        W_A = QA_s[a_star, d_star]
        W_D = QD_s[a_star, d_star]
        return a_star, d_star, W_A, W_D

    def select_real_actions(self, s: int) -> Tuple[int, int]:
        """
        在真实态势 s 上选择 (a_real, d_real):

        先由 local_attack_leader 得到 Stackelberg 对 (a*,d*),
        然后在其基础上做 ε-greedy 探索。
        """
        A = self.env.num_attack_actions
        D = self.env.num_defense_actions

        a_star, d_star, _, _ = self.local_attack_leader(s)

        # 攻击动作
        if random.random() < self.epsilon:
            a = random.randrange(A)
        else:
            a = a_star

        # 防御动作
        if random.random() < self.epsilon:
            d = random.randrange(D)
        else:
            d = d_star

        return a, d

    # ---------- 阶段 2: 预测态势下防御者为 Leader ----------

    def local_defense_leader(self, s_pred: int) -> Tuple[int, int]:
        """
        在预测态势 ŝ 上求局部 Stackelberg 解 (防御者为 Leader):

        1) 攻击者对每个 d 的最佳响应:
           a_BR(d) = argmax_a Q_A(ŝ,a,d)
        2) 防御者在此基础上选:
           d_pre = argmin_d Q_D(ŝ,a_BR(d),d)
        """
        QA_s = self.QA[s_pred]
        QD_s = self.QD[s_pred]
        A, D = QA_s.shape

        # 攻击者对每个 d 的 best response
        a_br = np.argmax(QA_s, axis=0)          # shape [D]
        idx = (a_br, np.arange(D))
        QD_lead = QD_s[idx]                     # shape [D]

        d_pre = int(np.argmin(QD_lead))
        a_pre = int(a_br[d_pre])
        return a_pre, d_pre

    # ---------- 训练主循环 ----------

    def train(self):
        for ep in range(self.episodes):
            s = self.env.reset()
            steps = 0

            while True:
                # 1) 阶段 1: 真实态势下攻击者为 Leader
                a_real, d_real = self.select_real_actions(s)

                # 2) INSSP 预测 + 阶段 2: 预测态势下防御者为 Leader
                s_pred = self.env.predict_next_state(s)
                _, d_pre = self.local_defense_leader(s_pred)

                # 3) 环境执行双阶段博弈, 得到 s_next, R_A, R_D
                s_next, R_A, R_D, done = self.env.step(
                    s, a_real, d_real, s_pred, d_pre
                )

                # 4) 下一真实态势上, 仍采用“攻击者为 Leader”的价值备份
                _, _, W_A_next, W_D_next = self.local_attack_leader(s_next)

                # 5) 时序差分更新 (真实态势下的真实动作对)
                self.QA[s, a_real, d_real] = \
                    (1 - self.alpha) * self.QA[s, a_real, d_real] + \
                    self.alpha * (R_A + self.gamma * W_A_next)

                self.QD[s, a_real, d_real] = \
                    (1 - self.alpha) * self.QD[s, a_real, d_real] + \
                    self.alpha * (R_D + self.gamma * W_D_next)

                s = s_next
                steps += 1
                if done or steps > 100:
                    break

            # ------- 记录初始状态的局部 Stackelberg 行为(可选, 用于画策略演变) -------
            s0 = self.env.initial_state
            a_star, d_star, _, _ = self.local_attack_leader(s0)
            piA = np.zeros(self.env.num_attack_actions)
            piD = np.zeros(self.env.num_defense_actions)
            piA[a_star] = 1.0
            piD[d_star] = 1.0
            self.policy_hist_A.append(piA)
            self.policy_hist_D.append(piD)

    # ---------- 提取最终策略 ----------

    def extract_policies(self) -> Dict[int, Tuple[int, int]]:
        """
        返回每个状态 s 的局部 Stackelberg 策略 (a_R^*(s), d_R^*(s))，
        对应真实态势下的攻防对抗行为。
        """
        S = self.env.num_states
        policies: Dict[int, Tuple[int, int]] = {}
        for s in range(S):
            a_star, d_star, _, _ = self.local_attack_leader(s)
            policies[s] = (a_star, d_star)
        return policies


# =========================
# 3. 一个简单的示例攻击图
# =========================

if __name__ == "__main__":
    # 例子: 线性攻击链 0 -> 1 -> 2 -> ... -> 10, 10 为终端
    edges = [
        Edge(0, 1, 0),
        Edge(1, 2, 1),
        Edge(2, 3, 2),
        Edge(3, 4, 3),
        Edge(4, 5, 4),
        Edge(5, 6, 5),
        Edge(6, 7, 6),
        Edge(7, 8, 7),
        Edge(8, 9, 8),
        Edge(9, 10, 9),
    ]

    env = ICSGameEnv(
        num_states=11,
        edges=edges,
        initial_state=0,
        terminal_states=[10],
        attack_cost=10.0,
        defense_cost=10.0,
        reward_success=100.0,
        predeploy_discount=0.5,  # 预部署成本折扣 η
    )

    agent = RSStackelbergMARL(
        env,
        gamma=0.9,
        alpha=0.1,
        epsilon=0.1,
        episodes=500,
    )

    agent.train()
    policies = agent.extract_policies()

    print("学习到的局部 Stackelberg 策略 (state -> (a_R^*(s), d_R^*(s))):")
    for s, (a_star, d_star) in policies.items():
        print(f"  state {s}: a*={a_star}, d*={d_star}")

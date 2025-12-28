#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack-Graph-Constrained Markov Chain with Vulnerability Weights (AGC-MC-V)
适用于：工业网络安全情境预测（考虑攻击图约束与CVE/漏洞暴露度权重）
"""

from typing import List, Dict, Tuple, Optional


class AttackGraphConstrainedMarkov:
    def __init__(self,
                 states: List[str],
                 edges: List[Tuple[str, str]],
                 epsilon: float = 1e-3,
                 state_weights: Optional[Dict[str, float]] = None):
        """
        :param states: 态势状态集合，如 ["S0", "S1", "S2", "S3", "END"]
                       每个状态可对应一个节点或一类聚合情境。
        :param edges: 攻击图允许的有向边列表 [(from_state, to_state), ...]
                      仅这些转移被视为物理/逻辑可达。
        :param epsilon: 拉普拉斯平滑系数，避免合法边长期为零概率。
        :param state_weights: 态势状态权重 W_j，用于表示目标状态的漏洞暴露度等。
                              例如 {"S1": 3.0, "S2": 5.0, "S3": 9.5, ...}。
                              未提供的状态默认权重为 1.0。
        """
        self.states = states
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.idx_to_state = {i: s for i, s in enumerate(states)}
        self.n = len(states)
        self.epsilon = epsilon

        # 掩码矩阵 M：1 表示 (Si -> Sj) 合法；0 表示不可达（强制 Pij = 0）
        self.M = [[0] * self.n for _ in range(self.n)]
        for u, v in edges:
            if u not in self.state_to_idx or v not in self.state_to_idx:
                raise ValueError(f"Edge ({u}->{v}) 包含未定义状态")
            i = self.state_to_idx[u]
            j = self.state_to_idx[v]
            self.M[i][j] = 1

        # 状态权重 W_j（漏洞暴露度、资产价值等）
        self.W = [1.0] * self.n
        if state_weights is not None:
            for s, w in state_weights.items():
                if s not in self.state_to_idx:
                    raise ValueError(f"state_weights 中包含未定义状态: {s}")
                j = self.state_to_idx[s]
                if w < 0:
                    raise ValueError("state_weights 不应为负数")
                self.W[j] = float(w)

        # 频次矩阵 N 与 转移概率矩阵 P
        self.N = [[0] * self.n for _ in range(self.n)]
        self.P = [[0.0] * self.n for _ in range(self.n)]

        # 历史序列（索引形式），供滑动窗口更新使用
        self.history: List[int] = []

    # -------------------- 工具函数 --------------------

    def _to_index_sequence(self, seq: List[str]) -> List[int]:
        idx_seq = []
        for s in seq:
            if s not in self.state_to_idx:
                raise ValueError(f"未知态势状态: {s}")
            idx_seq.append(self.state_to_idx[s])
        return idx_seq

    def set_state_weights(self, state_weights: Dict[str, float]) -> None:
        """
        运行中更新态势状态权重 W_j（例如根据最新漏洞扫描结果）。
        """
        for s, w in state_weights.items():
            if s not in self.state_to_idx:
                raise ValueError(f"state_weights 中包含未定义状态: {s}")
            j = self.state_to_idx[s]
            if w < 0:
                raise ValueError("state_weights 不应为负数")
            self.W[j] = float(w)
        # 更新权重后，可根据需要重新计算 P（若依赖 W_j）
        self._update_P_from_N()

    # -------------------- 离线训练 / 初始估计 --------------------

    def fit_from_history(self, state_sequence: List[str]) -> None:
        """
        使用完整历史态势序列估计转移矩阵（攻击图约束 + 漏洞权重）。
        :param state_sequence: 如 ["S0","S1","S1","S2","S3","END",...]
        """
        idx_seq = self._to_index_sequence(state_sequence)
        self.history = idx_seq[:]  # 保存一份历史索引

        # 清空频次
        self.N = [[0] * self.n for _ in range(self.n)]

        # 仅累计合法转移
        for t in range(len(idx_seq) - 1):
            i = idx_seq[t]
            j = idx_seq[t + 1]
            if self.M[i][j] == 1:
                self.N[i][j] += 1

        # 基于 N 和 W 更新 P
        self._update_P_from_N()

    # -------------------- 动态滑动窗口更新 --------------------

    def update_with_sliding_window(self,
                                   new_state: str,
                                   window_size: int = 1000) -> None:
        """
        动态更新：采用滑动窗口维护历史与 N，实现时间相关的 P(t)。
        :param new_state: 新观测的聚合态势状态名称，如 "S1"
        :param window_size: 窗口长度上限
        """
        if new_state not in self.state_to_idx:
            raise ValueError(f"未知态势状态: {new_state}")
        j_new = self.state_to_idx[new_state]

        # 如有前一状态，则更新对应合法转移频次
        if self.history:
            i_prev = self.history[-1]
            if self.M[i_prev][j_new] == 1:
                self.N[i_prev][j_new] += 1

            # 若窗口已满，移除最早一次转移的贡献
            if len(self.history) >= window_size:
                i_old = self.history[0]
                j_old = self.history[1]
                if self.M[i_old][j_old] == 1 and self.N[i_old][j_old] > 0:
                    self.N[i_old][j_old] -= 1
                # 弹出最早元素
                self.history.pop(0)

        # 加入新状态
        self.history.append(j_new)

        # 更新转移概率矩阵
        self._update_P_from_N()

    # -------------------- 概率矩阵更新（含漏洞权重） --------------------

    def _update_P_from_N(self) -> None:
        """
        根据当前 N、掩码 M 与状态权重 W_j 计算转移矩阵 P。
        对于合法边 (i->j): P_ij ∝ (N_ij + ε) * W_j
        对于非法边: P_ij = 0
        """
        for i in range(self.n):
            # 计算分母：仅对合法后继 j，累加 (N_ij + ε) * W_j
            denom = 0.0
            for j in range(self.n):
                if self.M[i][j] == 1:
                    denom += (self.N[i][j] + self.epsilon) * self.W[j]

            if denom <= 0.0:
                # 无合法后继或尚无观测，整行设为 0（可由上层逻辑处理自保持等策略）
                for j in range(self.n):
                    self.P[i][j] = 0.0
                continue

            for j in range(self.n):
                if self.M[i][j] == 1:
                    num = (self.N[i][j] + self.epsilon) * self.W[j]
                    self.P[i][j] = num / denom
                else:
                    self.P[i][j] = 0.0

    # -------------------- 预测接口 --------------------

    def predict_next(self,
                     current_state: str,
                     top_k: int = 1) -> List[Tuple[str, float]]:
        """
        基于当前态势状态，执行攻击图约束 + 漏洞加权的马尔可夫预测。
        :param current_state: 当前态势状态名称，如 "S1"
        :param top_k: 返回前 K 个候选状态
        :return: [(state_name, prob), ...]；若无合法后继则返回空列表
        """
        if current_state not in self.state_to_idx:
            raise ValueError(f"未知态势状态: {current_state}")
        i = self.state_to_idx[current_state]

        candidates = []
        for j in range(self.n):
            if self.M[i][j] == 1 and self.P[i][j] > 0.0:
                candidates.append((self.idx_to_state[j], self.P[i][j]))

        if not candidates:
            # 无合法后继或尚未形成分布，由调用方决定是自保持还是进入END等策略
            return []

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    # -------------------- 调试输出 --------------------

    def print_transition_matrix(self):
        """
        打印当前转移矩阵 P（用于验证与实验展示）。
        """
        header = "      " + "  ".join([f"{s:>8}" for s in self.states])
        print(header)
        for i, si in enumerate(self.states):
            row = [f"{si:>6}"]
            for j in range(self.n):
                row.append(f"{self.P[i][j]:8.3f}")
            print("  ".join(row))


# =========================
# 示例：考虑漏洞权重的 INSSP 预测
# =========================

if __name__ == "__main__":
    # 1. 定义态势状态（示例）
    states = ["S0", "S1", "S2", "S3", "END"]
    # S0: 正常
    # S1: 边界/DMZ受攻
    # S2: 过程层异常
    # S3: 控制层受攻（关键PLC/HMI）
    # END: 攻击结束/阻断

    # 2. 定义攻击图允许转移（依据实际拓扑/策略调整）
    edges = [
        ("S0", "S0"),
        ("S0", "S1"),
        ("S1", "S1"),
        ("S1", "S2"),
        ("S1", "S3"),
        ("S1", "END"),
        ("S2", "S2"),
        ("S2", "S3"),
        ("S2", "END"),
        ("S3", "S3"),
        ("S3", "END"),
    ]

    # 3. 定义态势状态漏洞权重 W_j（示例）
    # 可由该态势对应节点上的 CVE 暴露度聚合而来
    state_weights = {
        "S0": 1.0,   # 正常态最低权重
        "S1": 2.0,   # 边界有中危漏洞
        "S2": 4.0,   # 过程层存在可远程利用漏洞
        "S3": 9.0,   # 控制层关键PLC/HMI存在高危CVE
        "END": 0.5   # 终止态视为不吸引攻击
    }

    model = AttackGraphConstrainedMarkov(
        states=states,
        edges=edges,
        epsilon=1e-3,
        state_weights=state_weights
    )

    # 4. 历史情境序列（示例数据）
    history = [
        "S0", "S0", "S1", "S1", "S2", "S3", "END",
        "S0", "S1", "S2", "END",
        "S0", "S1", "S3", "END",
        "S0", "S1", "S1", "S3", "END",
        "S0", "S1", "S2", "S3", "END",
    ]

    # 5. 基于历史数据估计转移矩阵（含漏洞权重）
    model.fit_from_history(history)

    print("=== 转移矩阵 P（攻击图约束 + 漏洞权重） ===")
    model.print_transition_matrix()

    # 6. 示例预测：当前态势为 S1（边界受攻），预测下一情境
    current = "S1"
    top_candidates = model.predict_next(current_state=current, top_k=3)

    print(f"\n当前态势: {current}")
    if not top_candidates:
        print("无合法后继或尚无统计数据")
    else:
        print("预测下一时刻最可能态势（Top-3）:")
        for s, p in top_candidates:
            print(f"  {s}: {p:.3f}")

    # 7. 动态更新示例（新观测序列）
    new_observations = ["S1", "S2", "S3", "END", "S0", "S1", "S3", "END"]
    for obs in new_observations:
        model.update_with_sliding_window(new_state=obs, window_size=20)

    print("\n=== 动态更新后的转移矩阵 P ===")
    model.print_transition_matrix()

    current = "S1"
    top_candidates = model.predict_next(current_state=current, top_k=2)
    print(f"\n动态更新后，当前态势: {current}")
    if not top_candidates:
        print("无合法后继或尚无统计数据")
    else:
        print("预测下一时刻最可能态势（Top-2）:")
        for s, p in top_candidates:
            print(f"  {s}: {p:.3f}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from .constants import *
from .dlutils import *

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from scipy.spatial import KDTree
# import numpy as np
# class SparseLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1, connectivity_ratio=0.5):
#         super(SparseLSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.connectivity_ratio = connectivity_ratio
#
#         # 初始化 LSTM 的权重
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#
#         # 构造稀疏连接
#         self.sparse_weights = self._initialize_sparse_weights()
#
#     def _initialize_sparse_weights(self):
#         # 初始化随机图 g(V, p_ij)
#         n_input = self.input_size
#         n_hidden = self.hidden_size
#         total_neurons = n_input + n_hidden
#
#         # 计算阈值 Q
#         Q = n_input * n_hidden * self.connectivity_ratio
#
#         # 随机生成连接概率
#         p_ij = torch.rand(n_input, n_hidden)
#
#         # 根据阈值 Q 确定连接状态
#         sparse_mask = (p_ij > Q).float()
#
#         return sparse_mask
#
#     def forward(self, x, hidden_state=None):
#         batch_size, seq_len, _ = x.size()
#
#         # 应用稀疏连接
#         if self.sparse_weights is not None:
#             x = torch.matmul(x, self.sparse_weights)
#
#         # LSTM 前向传播
#         lstm_out, hidden_state = self.lstm(x, hidden_state)
#         return lstm_out, hidden_state
#
#
# class FPE_16(nn.Module):
#     def __init__(self):
#         super(FPE_16, self).__init__()
#         self.name = 'FPE_16'
#         self.lr = 0.0001
#         self.n_hosts = 16
#         self.n_feats = 3 * self.n_hosts
#         self.n_window = 5
#         self.n_latent = 30
#         self.n_hidden = 16
#         self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
#
#         # 使用稀疏连接的 LSTM 替换 GRU
#         self.lstm = SparseLSTM(self.n_window, self.n_window, connectivity_ratio=0.5)
#
#         src_ids = torch.tensor(list(range(self.n_feats)))
#         dst_ids = torch.tensor([self.n_feats] * self.n_feats)
#         self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
#         self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1)
#         self.encoder = nn.Sequential(
#             nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent), nn.LeakyReLU(True),
#         )
#         self.anomaly_decoder = nn.Sequential(
#             nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
#         )
#         self.prototype_decoder = nn.Sequential(
#             nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
#         )
#         self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]
#
#     def encode(self, t, s):
#         h = torch.randn(1, self.n_window, dtype=torch.double)
#
#         # 使用稀疏连接的 LSTM
#         lstm_t, _ = self.lstm(torch.t(t).unsqueeze(0), (h.unsqueeze(0), h.unsqueeze(0)))
#         lstm_t = torch.t(lstm_t.squeeze(0))
#
#         graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
#         gat_t = self.gat(torch.t(graph))
#         gat_t = torch.t(gat_t)
#         concat_t = torch.cat((lstm_t, gat_t), dim=1)
#
#         o, _ = self.mha(concat_t, concat_t, concat_t)
#         t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)
#         return t
#
#     def anomaly_decode(self, t):
#         anomaly_scores = []
#         for elem in t:
#             anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))
#         return anomaly_scores
#
#     def prototype_decode(self, t):
#         prototypes = []
#         for elem in t:
#             prototypes.append(self.prototype_decoder(elem))
#         return prototypes
#
#     def forward(self, t, s):
#         t = self.encode(t, s)
#         anomaly_scores = self.anomaly_decode(t)
#         prototypes = self.prototype_decode(t)
#         return anomaly_scores, prototypes

################# 2 一层LSTM
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from scipy.spatial import KDTree
# import numpy as np
# import dgl
# # 定义稀疏连接的 LSTM
# class SparseLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1, connectivity_ratio=0.5):
#         super(SparseLSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.connectivity_ratio = connectivity_ratio
#
#         # 初始化 LSTM 的权重
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).double()
#
#         # 构造稀疏连接
#         self.sparse_weights = self._initialize_sparse_weights()
#
#     def _initialize_sparse_weights(self):
#         # 初始化随机图 g(V, p_ij)
#         n_input = self.input_size
#         n_hidden = self.hidden_size
#
#         # 计算阈值 Q
#         Q = n_input * n_hidden * self.connectivity_ratio
#
#         # 随机生成连接概率
#         p_ij = torch.rand(n_input, n_hidden, dtype=torch.float64)
#
#         # 根据阈值 Q 确定连接状态
#         sparse_mask = (p_ij > Q).double()
#
#         return sparse_mask
#
#     def forward(self, x, hidden_state=None):
#         batch_size, seq_len, _ = x.size()
#
#         # 应用稀疏连接
#         if self.sparse_weights is not None:
#             x = torch.matmul(x.double(), self.sparse_weights)
#
#         # LSTM 前向传播
#         lstm_out, hidden_state = self.lstm(x, hidden_state)
#         return lstm_out, hidden_state
#
#
# # 定义 GAT 模型
# class GAT(nn.Module):
#     def __init__(self, graph, in_feats, out_feats):
#         super(GAT, self).__init__()
#         self.graph = graph
#         self.fc = nn.Linear(in_feats, out_feats).double()
#
#     def forward(self, x):
#         return self.fc(x.double())
#
#
# # 定义 FPE_16 模型
# class FPE_16(nn.Module):
#     def __init__(self):
#         super(FPE_16, self).__init__()
#         self.name = 'FPE_16'
#         self.lr = 0.0001
#         self.n_hosts = 16
#         self.n_feats = 3 * self.n_hosts  # 特征维度
#         self.n_window = 5
#         self.n_latent = 30
#         self.n_hidden = 16
#         self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
#
#         # 使用稀疏连接的 LSTM 替换 GRU
#         self.lstm = SparseLSTM(self.n_window, self.n_window, connectivity_ratio=0.5)
#
#         # 定义 GAT 模型的图结构（关键修复点）
#         # 确保节点 ID 是整数且有效
#         src_ids = torch.arange(self.n_feats, dtype=torch.int32)  # 源节点 ID 为 0 到 n_feats-1
#         dst_ids = torch.full((self.n_feats,), self.n_feats, dtype=torch.int32)  # 目标节点 ID 统一为 n_feats
#         self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
#
#         # 定义多头注意力机制
#         self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1).double()
#
#         # 定义编码器
#         self.encoder = nn.Sequential(
#             nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent).double(),
#             nn.LeakyReLU(True),
#         )
#
#         # 定义异常解码器
#         self.anomaly_decoder = nn.Sequential(
#             nn.Linear(self.n_latent, 2).double(),
#             nn.Softmax(dim=0),
#         )
#
#         # 定义原型解码器
#         self.prototype_decoder = nn.Sequential(
#             nn.Linear(self.n_latent, PROTO_DIM).double(),
#             nn.Sigmoid(),
#         )
#
#         # 初始化原型
#         self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.float64) for _ in range(3)]
#
#     def encode(self, t, s):
#         # 初始化隐藏状态
#         h = torch.randn(1, self.n_window, dtype=torch.float64)
#
#         # 使用稀疏连接的 LSTM
#         lstm_t, _ = self.lstm(torch.t(t).unsqueeze(0), (h.unsqueeze(0), h.unsqueeze(0)))
#         lstm_t = torch.t(lstm_t.squeeze(0))
#
#         # 构建图数据
#         graph = torch.cat((t, torch.zeros(self.n_window, 1, dtype=torch.float64)), dim=1)
#         gat_t = self.gat(torch.t(graph))
#         gat_t = torch.t(gat_t)
#
#         # 拼接 LSTM 和 GAT 的输出
#         concat_t = torch.cat((lstm_t.double(), gat_t.double()), dim=1)
#
#         # 使用多头注意力机制
#         o, _ = self.mha(concat_t, concat_t, concat_t)
#
#         # 编码器
#         t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)
#         return t
#
#     def anomaly_decode(self, t):
#         anomaly_scores = []
#         for elem in t:
#             anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))
#         return anomaly_scores
#
#     def prototype_decode(self, t):
#         prototypes = []
#         for elem in t:
#             prototypes.append(self.prototype_decoder(elem))
#         return prototypes
#
#     def forward(self, t, s):
#         t = self.encode(t.double(), s)  # 确保输入是 double 类型
#         anomaly_scores = self.anomaly_decode(t)
#         prototypes = self.prototype_decode(t)
#         return anomaly_scores, prototypes
# # 定义生成器模型
# class Gen_16(nn.Module):
#     def __init__(self):
#         super(Gen_16, self).__init__()
#         self.name = 'Gen_16'
#         self.lr = 0.00005
#         self.n_hosts = 16
#         self.n_hidden = 64
#         self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
#         self.delta = nn.Sequential(
#             nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
#             nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
#         ).double()
#
#     def forward(self, e, s):
#         del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
#         return s + del_s.reshape(self.n_hosts, self.n_hosts)
#
#
# # 定义判别器模型
# class Disc_16(nn.Module):
#     def __init__(self):
#         super(Disc_16, self).__init__()
#         self.name = 'Disc_16'
#         self.lr = 0.00005
#         self.n_hosts = 16
#         self.n_hidden = 64
#         self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
#         self.probs = nn.Sequential(
#             nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
#             nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
#         ).double()
#
#     def forward(self, o, n):
#         probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
#         return probs


################# 3   2层LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import os

# 定义稀疏连接的 LSTM
class SparseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, connectivity_ratio=0.5, dropout=0.2):
        super(SparseLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.connectivity_ratio = connectivity_ratio

        # 初始化 LSTM 的权重
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout).double()

        # 构造稀疏连接
        self.sparse_weights = self._initialize_sparse_weights()

        # 初始化 LSTM 的权重
        self._initialize_lstm_weights()

    def _initialize_sparse_weights(self):
        # 初始化随机图 g(V, p_ij)
        n_input = self.input_size
        n_hidden = self.hidden_size

        # 计算阈值 Q
        Q = n_input * n_hidden * self.connectivity_ratio

        # 随机生成连接概率
        p_ij = torch.rand(n_input, n_hidden, dtype=torch.float64)

        # 根据阈值 Q 确定连接状态
        sparse_mask = (p_ij > Q).double()

        return sparse_mask

    def _initialize_lstm_weights(self):
        # 使用 Xavier 初始化 LSTM 的权重
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x, hidden_state=None):
        batch_size, seq_len, _ = x.size()

        # 确保输入维度与稀疏权重匹配
        if self.sparse_weights is not None:
            # 这里的 x 是 (batch_size, seq_len, input_size)，要进行矩阵乘法前需要展平
            # 将 x 从 (batch_size, seq_len, input_size) 展平为 (batch_size * seq_len, input_size)
            # print(f"x.size() before reshaping: {x.size()}")  # 打印 x 的形状来调试

            x_reshaped = x.view(-1, self.input_size)

            # print(f"x_reshaped size: {x_reshaped.size()}")  # 打印重塑后的大小来调试

            # 执行稀疏矩阵乘法
            x_reshaped = torch.matmul(x_reshaped, self.sparse_weights)

            # print(f"x_reshaped after matmul: {x_reshaped.size()}")  # 打印矩阵乘法后的大小来调试

            # 恢复到 (batch_size, seq_len, hidden_size)
            x = x_reshaped.view(batch_size, seq_len, self.hidden_size)

            # print(f"x.size() after reshaping: {x.size()}")  # 打印重塑后的 x 形状

        # LSTM 前向传播
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        return lstm_out, hidden_state

# 定义 GAT 模型
class GAT(nn.Module):
    def __init__(self, graph, in_feats, out_feats):
        super(GAT, self).__init__()
        self.graph = graph
        self.fc = nn.Linear(in_feats, out_feats).double()

    def forward(self, x):
        return self.fc(x.double())

# 定义 FEE_16 模型
class FEE_16(nn.Module):
    def __init__(self):
        super(FEE_16, self).__init__()
        self.name = 'FEE_16'
        self.lr = 0.0001
        self.n_hosts = 16
        self.n_feats = 3 * self.n_hosts  # 特征维度
        self.n_window = 5
        self.n_latent = 30
        self.n_hidden = 16
        self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts

        # 使用稀疏连接的 LSTM 替换 GRU
        self.lstm = SparseLSTM(self.n_window, self.n_window, num_layers=2, connectivity_ratio=0.5, dropout=0.2)

        # 定义 GAT 模型的图结构（关键修复点）
        # 确保节点 ID 是整数且有效
        src_ids = torch.arange(self.n_feats, dtype=torch.int32)  # 源节点 ID 为 0 到 n_feats-1
        dst_ids = torch.full((self.n_feats,), self.n_feats, dtype=torch.int32)  # 目标节点 ID 统一为 n_feats
        self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)

        # 定义多头注意力机制
        self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1).double()

        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent).double(),
            nn.LeakyReLU(True),
        )

        # 定义异常解码器
        self.anomaly_decoder = nn.Sequential(
            nn.Linear(self.n_latent, 2).double(),
            nn.Softmax(dim=0),
        )

        # 定义原型解码器
        self.prototype_decoder = nn.Sequential(
            nn.Linear(self.n_latent, PROTO_DIM).double(),
            nn.Sigmoid(),
        )

        # 初始化原型
        self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.float64) for _ in range(3)]

    def encode(self, t, s):
        # 初始化隐藏状态
        h = torch.randn(2, self.n_window, dtype=torch.float64)  # 2 层 LSTM，隐藏状态维度为 (num_layers, batch_size, hidden_size)

        # 使用稀疏连接的 LSTM
        lstm_t, _ = self.lstm(torch.t(t).unsqueeze(0), (h.unsqueeze(1), h.unsqueeze(1)))  # 2 层 LSTM
        lstm_t = torch.t(lstm_t.squeeze(0))

        # 构建图数据
        graph = torch.cat((t, torch.zeros(self.n_window, 1, dtype=torch.float64)), dim=1)
        gat_t = self.gat(torch.t(graph))
        gat_t = torch.t(gat_t)

        # 拼接 LSTM 和 GAT 的输出
        concat_t = torch.cat((lstm_t.double(), gat_t.double()), dim=1)

        # 使用多头注意力机制
        o, _ = self.mha(concat_t, concat_t, concat_t)

        # 编码器
        t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)
        return t

    def anomaly_decode(self, t):
        anomaly_scores = []
        for elem in t:
            anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))
        return anomaly_scores

    def prototype_decode(self, t):
        prototypes = []
        for elem in t:
            prototypes.append(self.prototype_decoder(elem))
        return prototypes

    def forward(self, t, s):
        t = self.encode(t.double(), s)  # 确保输入是 double 类型
        anomaly_scores = self.anomaly_decode(t)
        prototypes = self.prototype_decode(t)
        return anomaly_scores, prototypes

# 定义生成器模型
class Gen_16(nn.Module):
    def __init__(self):
        super(Gen_16, self).__init__()
        self.name = 'Gen_16'
        self.lr = 0.00005
        self.n_hosts = 16
        self.n_hidden = 32#64
        self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
        self.delta = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
        ).double()

    def forward(self, e, s):
        del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
        return s + del_s.reshape(self.n_hosts, self.n_hosts)

# 定义判别器模型
class Disc_16(nn.Module):
    def __init__(self):
        super(Disc_16, self).__init__()
        self.name = 'Disc_16'
        self.lr = 0.00005
        self.n_hosts = 16
        self.n_hidden = 32#64
        self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
        self.probs = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
        ).double()

    def forward(self, o, n):
        probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
        return probs



######### 稀疏LSTM + GCN
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import dgl
# import dgl.function as fn
# import os
#
# # 定义稀疏连接的 LSTM
# class SparseLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=2, connectivity_ratio=0.5, dropout=0.2):
#         super(SparseLSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.connectivity_ratio = connectivity_ratio
#
#         # 初始化 LSTM 的权重
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout).double()
#
#         # 构造稀疏连接
#         self.sparse_weights = self._initialize_sparse_weights()
#
#         # 初始化 LSTM 的权重
#         self._initialize_lstm_weights()
#
#     def _initialize_sparse_weights(self):
#         # 初始化随机图 g(V, p_ij)
#         n_input = self.input_size
#         n_hidden = self.hidden_size
#
#         # 计算阈值 Q
#         Q = n_input * n_hidden * self.connectivity_ratio
#
#         # 随机生成连接概率
#         p_ij = torch.rand(n_input, n_hidden, dtype=torch.float64)
#
#         # 根据阈值 Q 确定连接状态
#         sparse_mask = (p_ij > Q).double()
#
#         return sparse_mask
#
#     def _initialize_lstm_weights(self):
#         # 使用 Xavier 初始化 LSTM 的权重
#         for name, param in self.lstm.named_parameters():
#             if 'weight' in name:
#                 nn.init.xavier_uniform_(param)
#
#     def forward(self, x, hidden_state=None):
#         batch_size, seq_len, _ = x.size()
#
#         # 确保输入维度与稀疏权重匹配
#         if self.sparse_weights is not None:
#             # 这里的 x 是 (batch_size, seq_len, input_size)，要进行矩阵乘法前需要展平
#             # 将 x 从 (batch_size, seq_len, input_size) 展平为 (batch_size * seq_len, input_size)
#             x_reshaped = x.view(-1, self.input_size)
#
#             # 执行稀疏矩阵乘法
#             x_reshaped = torch.matmul(x_reshaped, self.sparse_weights)
#
#             # 恢复到 (batch_size, seq_len, hidden_size)
#             x = x_reshaped.view(batch_size, seq_len, self.hidden_size)
#
#         # LSTM 前向传播
#         lstm_out, hidden_state = self.lstm(x, hidden_state)
#         return lstm_out, hidden_state
#
# # 定义 GCN 模型
# class GCN(nn.Module):
#     def __init__(self, graph, in_feats, out_feats):
#         super(GCN, self).__init__()
#         self.graph = graph
#         self.fc = nn.Linear(in_feats, out_feats).double()
#
#     def forward(self, x):
#         # 使用 DGL 的 GCN 消息传递
#         self.graph.ndata['h'] = x
#         self.graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
#         h = self.graph.ndata['h']
#         return self.fc(h)
#
# # 定义 FPE_16 模型
# class FPE_16(nn.Module):
#     def __init__(self):
#         super(FPE_16, self).__init__()
#         self.name = 'FPE_16'
#         self.lr = 0.0001
#         self.n_hosts = 16
#         self.n_feats = 3 * self.n_hosts  # 特征维度
#         self.n_window = 5
#         self.n_latent = 30
#         self.n_hidden = 16
#         self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
#
#         # 使用稀疏连接的 LSTM 替换 GRU
#         self.lstm = SparseLSTM(self.n_window, self.n_window, num_layers=2, connectivity_ratio=0.5, dropout=0.2)
#
#         # 定义 GCN 模型的图结构（关键修复点）
#         # 确保节点 ID 是整数且有效
#         src_ids = torch.arange(self.n_feats, dtype=torch.int32)  # 源节点 ID 为 0 到 n_feats-1
#         dst_ids = torch.full((self.n_feats,), self.n_feats, dtype=torch.int32)  # 目标节点 ID 统一为 n_feats
#         self.gcn = GCN(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
#
#         # 定义多头注意力机制
#         self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1).double()
#
#         # 定义编码器
#         self.encoder = nn.Sequential(
#             nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent).double(),
#             nn.LeakyReLU(True),
#         )
#
#         # 定义异常解码器
#         self.anomaly_decoder = nn.Sequential(
#             nn.Linear(self.n_latent, 2).double(),
#             nn.Softmax(dim=0),
#         )
#
#         # 定义原型解码器
#         self.prototype_decoder = nn.Sequential(
#             nn.Linear(self.n_latent, PROTO_DIM).double(),
#             nn.Sigmoid(),
#         )
#
#         # 初始化原型
#         self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.float64) for _ in range(3)]
#
#     def encode(self, t, s):
#         # 初始化隐藏状态
#         h = torch.randn(2, self.n_window, dtype=torch.float64)  # 2 层 LSTM，隐藏状态维度为 (num_layers, batch_size, hidden_size)
#
#         # 使用稀疏连接的 LSTM
#         lstm_t, _ = self.lstm(torch.t(t).unsqueeze(0), (h.unsqueeze(1), h.unsqueeze(1)))  # 2 层 LSTM
#         lstm_t = torch.t(lstm_t.squeeze(0))
#
#         # 构建图数据
#         graph = torch.cat((t, torch.zeros(self.n_window, 1, dtype=torch.float64)), dim=1)
#         gcn_t = self.gcn(torch.t(graph))
#         gcn_t = torch.t(gcn_t)
#
#         # 拼接 LSTM 和 GCN 的输出
#         concat_t = torch.cat((lstm_t.double(), gcn_t.double()), dim=1)
#
#         # 使用多头注意力机制
#         o, _ = self.mha(concat_t, concat_t, concat_t)
#
#         # 编码器
#         t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)
#         return t
#
#     def anomaly_decode(self, t):
#         anomaly_scores = []
#         for elem in t:
#             anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))
#         return anomaly_scores
#
#     def prototype_decode(self, t):
#         prototypes = []
#         for elem in t:
#             prototypes.append(self.prototype_decoder(elem))
#         return prototypes
#
#     def forward(self, t, s):
#         t = self.encode(t.double(), s)  # 确保输入是 double 类型
#         anomaly_scores = self.anomaly_decode(t)
#         prototypes = self.prototype_decode(t)
#         return anomaly_scores, prototypes
#
# # 定义生成器模型
# class Gen_16(nn.Module):
#     def __init__(self):
#         super(Gen_16, self).__init__()
#         self.name = 'Gen_16'
#         self.lr = 0.00005
#         self.n_hosts = 16
#         self.n_hidden = 32#64
#         self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
#         self.delta = nn.Sequential(
#             nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
#             nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
#         ).double()
#
#     def forward(self, e, s):
#         del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
#         return s + del_s.reshape(self.n_hosts, self.n_hosts)
#
# # 定义判别器模型
# class Disc_16(nn.Module):
#     def __init__(self):
#         super(Disc_16, self).__init__()
#         self.name = 'Disc_16'
#         self.lr = 0.00005
#         self.n_hosts = 16
#         self.n_hidden = 32#64
#         self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
#         self.probs = nn.Sequential(
#             nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
#             nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
#         ).double()
#
#     def forward(self, o, n):
#         probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
#         return probs




######### 稀疏LSTM + GIN
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import dgl
# import os
#
# # 定义稀疏连接的 LSTM
# class SparseLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=2, connectivity_ratio=0.5, dropout=0.2):
#         super(SparseLSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.connectivity_ratio = connectivity_ratio
#
#         # 初始化 LSTM 的权重
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout).double()
#
#         # 构造稀疏连接
#         self.sparse_weights = self._initialize_sparse_weights()
#
#         # 初始化 LSTM 的权重
#         self._initialize_lstm_weights()
#
#     def _initialize_sparse_weights(self):
#         # 初始化随机图 g(V, p_ij)
#         n_input = self.input_size
#         n_hidden = self.hidden_size
#
#         # 计算阈值 Q
#         Q = n_input * n_hidden * self.connectivity_ratio
#
#         # 随机生成连接概率
#         p_ij = torch.rand(n_input, n_hidden, dtype=torch.float64)
#
#         # 根据阈值 Q 确定连接状态
#         sparse_mask = (p_ij > Q).double()
#
#         return sparse_mask
#
#     def _initialize_lstm_weights(self):
#         # 使用 Xavier 初始化 LSTM 的权重
#         for name, param in self.lstm.named_parameters():
#             if 'weight' in name:
#                 nn.init.xavier_uniform_(param)
#
#     def forward(self, x, hidden_state=None):
#         batch_size, seq_len, _ = x.size()
#
#         # 确保输入维度与稀疏权重匹配
#         if self.sparse_weights is not None:
#             # 将 x 从 (batch_size, seq_len, input_size) 展平为 (batch_size * seq_len, input_size)
#             x_reshaped = x.view(-1, self.input_size)
#
#             # 执行稀疏矩阵乘法
#             x_reshaped = torch.matmul(x_reshaped, self.sparse_weights)
#
#             # 恢复到 (batch_size, seq_len, hidden_size)
#             x = x_reshaped.view(batch_size, seq_len, self.hidden_size)
#
#         # LSTM 前向传播
#         lstm_out, hidden_state = self.lstm(x, hidden_state)
#         return lstm_out, hidden_state
#
# # 定义 GIN 模型
# class GIN(nn.Module):
#     def __init__(self, graph, in_feats, out_feats):
#         super(GIN, self).__init__()
#         self.graph = graph
#         # 定义 MLP 用于 GIN 的聚合函数
#         self.mlp = nn.Sequential(
#             nn.Linear(in_feats, out_feats).double(),
#             nn.ReLU(),
#             nn.Linear(out_feats, out_feats).double()
#         )
#         # 使用 DGL 的 GINConv 层
#         self.conv = dgl.nn.pytorch.GINConv(self.mlp, 'sum').double()
#
#     def forward(self, x):
#         # 将节点特征输入 GIN 层
#         return self.conv(self.graph, x)
#
# # 定义 FPE_16 模型
# class FPE_16(nn.Module):
#     def __init__(self):
#         super(FPE_16, self).__init__()
#         self.name = 'FPE_16'
#         self.lr = 0.0001
#         self.n_hosts = 16
#         self.n_feats = 3 * self.n_hosts  # 特征维度
#         self.n_window = 5
#         self.n_latent = 30
#         self.n_hidden = 16
#         self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
#
#         # 使用稀疏连接的 LSTM 替换 GRU
#         self.lstm = SparseLSTM(self.n_window, self.n_window, num_layers=2, connectivity_ratio=0.5, dropout=0.2)
#
#         # 定义 GIN 模型的图结构
#         src_ids = torch.arange(self.n_feats, dtype=torch.int32)  # 源节点 ID 为 0 到 n_feats-1
#         dst_ids = torch.full((self.n_feats,), self.n_feats, dtype=torch.int32)  # 目标节点 ID 统一为 n_feats
#         self.gin = GIN(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
#
#         # 定义多头注意力机制
#         self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1).double()
#
#         # 定义编码器
#         self.encoder = nn.Sequential(
#             nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent).double(),
#             nn.LeakyReLU(True),
#         )
#
#         # 定义异常解码器
#         self.anomaly_decoder = nn.Sequential(
#             nn.Linear(self.n_latent, 2).double(),
#             nn.Softmax(dim=0),
#         )
#
#         # 定义原型解码器
#         self.prototype_decoder = nn.Sequential(
#             nn.Linear(self.n_latent, PROTO_DIM).double(),
#             nn.Sigmoid(),
#         )
#
#         # 初始化原型
#         self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.float64) for _ in range(3)]
#
#     def encode(self, t, s):
#         # 初始化隐藏状态
#         h = torch.randn(2, self.n_window, dtype=torch.float64)  # 2 层 LSTM，隐藏状态维度为 (num_layers, batch_size, hidden_size)
#
#         # 使用稀疏连接的 LSTM
#         lstm_t, _ = self.lstm(torch.t(t).unsqueeze(0), (h.unsqueeze(1), h.unsqueeze(1)))  # 2 层 LSTM
#         lstm_t = torch.t(lstm_t.squeeze(0))
#
#         # 构建图数据
#         graph = torch.cat((t, torch.zeros(self.n_window, 1, dtype=torch.float64)), dim=1)
#         gin_t = self.gin(torch.t(graph))
#         gin_t = torch.t(gin_t)
#
#         # 拼接 LSTM 和 GIN 的输出
#         concat_t = torch.cat((lstm_t.double(), gin_t.double()), dim=1)
#
#         # 使用多头注意力机制
#         o, _ = self.mha(concat_t, concat_t, concat_t)
#
#         # 编码器
#         t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)
#         return t
#
#     def anomaly_decode(self, t):
#         anomaly_scores = []
#         for elem in t:
#             anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))
#         return anomaly_scores
#
#     def prototype_decode(self, t):
#         prototypes = []
#         for elem in t:
#             prototypes.append(self.prototype_decoder(elem))
#         return prototypes
#
#     def forward(self, t, s):
#         t = self.encode(t.double(), s)  # 确保输入是 double 类型
#         anomaly_scores = self.anomaly_decode(t)
#         prototypes = self.prototype_decode(t)
#         return anomaly_scores, prototypes
#
# # 定义生成器模型
# class Gen_16(nn.Module):
#     def __init__(self):
#         super(Gen_16, self).__init__()
#         self.name = 'Gen_16'
#         self.lr = 0.00005
#         self.n_hosts = 16
#         self.n_hidden = 32  # 64
#         self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
#         self.delta = nn.Sequential(
#             nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
#             nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
#         ).double()
#
#     def forward(self, e, s):
#         del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
#         return s + del_s.reshape(self.n_hosts, self.n_hosts)
#
# # 定义判别器模型
# class Disc_16(nn.Module):
#     def __init__(self):
#         super(Disc_16, self).__init__()
#         self.name = 'Disc_16'
#         self.lr = 0.00005
#         self.n_hosts = 16
#         self.n_hidden = 32  # 64
#         self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
#         self.probs = nn.Sequential(
#             nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
#             nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
#         ).double()
#
#     def forward(self, o, n):
#         probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
#         return probs




######################################  k-d
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.neighbors import NearestNeighbors
#
# class SparseLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=2, connectivity_ratio=0.5, dropout=0.2, k_neighbors=5):
#         super(SparseLSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.connectivity_ratio = connectivity_ratio
#         self.k_neighbors = k_neighbors  # 选择最相关的k个记忆
#
#         # 初始化 LSTM 的权重
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout).double()
#
#         # 构造稀疏连接
#         self.sparse_weights = self._initialize_sparse_weights()
#
#         # 初始化 LSTM 的权重
#         self._initialize_lstm_weights()
#
#         # 记忆池，用于存储历史隐藏状态
#         self.memory_pool = []  # 存储历史隐藏状态
#         self.memory_capacity = 1000  # 记忆池容量
#
#     def _initialize_sparse_weights(self):
#         # 初始化随机图 g(V, p_ij)
#         n_input = self.input_size
#         n_hidden = self.hidden_size
#
#         # 计算阈值 Q
#         Q = n_input * n_hidden * self.connectivity_ratio
#
#         # 随机生成连接概率
#         p_ij = torch.rand(n_input, n_hidden, dtype=torch.float64)
#
#         # 根据阈值 Q 确定连接状态
#         sparse_mask = (p_ij > Q).double()
#
#         return sparse_mask
#
#     def _initialize_lstm_weights(self):
#         # 使用 Xavier 初始化 LSTM 的权重
#         for name, param in self.lstm.named_parameters():
#             if 'weight' in name:
#                 nn.init.xavier_uniform_(param)
#
#     def _update_memory_pool(self, hidden_state):
#         """
#         更新记忆池，存储历史隐藏状态
#         """
#         if len(self.memory_pool) >= self.memory_capacity:
#             self.memory_pool.pop(0)  # 如果记忆池已满，移除最旧的记忆
#         self.memory_pool.append(hidden_state.detach().cpu().numpy())
#
#     def _select_relevant_memory(self, x, batch_size, seq_len):
#         """
#         使用搜索树选择与当前输入最相关的k个记忆
#         """
#         if len(self.memory_pool) == 0:
#             return None  # 如果记忆池为空，返回None
#
#         # 将记忆池中的隐藏状态转换为特征向量
#         memory_features = torch.tensor(self.memory_pool, dtype=torch.float64).view(len(self.memory_pool), -1)
#
#         # 确保 x 的特征维度与记忆池中的特征维度一致
#         x_flattened = x.view(batch_size * seq_len, -1)  # 将 x 展平为 (batch_size * seq_len, hidden_size)
#         x_flattened = x_flattened.mean(dim=0).unsqueeze(0)  # 取均值并增加 batch 维度
#
#         # 使用K近邻搜索找到与当前输入最相关的k个记忆
#         knn = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(self.memory_pool)))
#         knn.fit(memory_features)
#         distances, indices = knn.kneighbors(x_flattened.cpu().numpy())
#
#         # 返回最相关的k个记忆
#         relevant_memory = torch.tensor([self.memory_pool[i] for i in indices[0]], dtype=torch.float64)
#         return relevant_memory
#
#     def forward(self, x, hidden_state=None):
#         batch_size, seq_len, _ = x.size()
#
#         # 确保输入维度与稀疏权重匹配
#         if self.sparse_weights is not None:
#             # 将 x 从 (batch_size, seq_len, input_size) 展平为 (batch_size * seq_len, input_size)
#             x_reshaped = x.view(-1, self.input_size)
#
#             # 执行稀疏矩阵乘法
#             x_reshaped = torch.matmul(x_reshaped, self.sparse_weights)
#
#             # 恢复到 (batch_size, seq_len, hidden_size)
#             x = x_reshaped.view(batch_size, seq_len, self.hidden_size)
#
#         # LSTM 前向传播
#         lstm_out, hidden_state = self.lstm(x, hidden_state)
#
#         # 更新记忆池
#         self._update_memory_pool(hidden_state[0])  # 存储隐藏状态
#
#         # 选择与当前输入最相关的k个记忆
#         relevant_memory = self._select_relevant_memory(x, batch_size, seq_len)
#         if relevant_memory is not None:
#             # 将相关记忆与当前隐藏状态融合
#             relevant_memory = relevant_memory.to(x.device)
#             hidden_state = (hidden_state[0] + relevant_memory.mean(dim=0), hidden_state[1])
#
#         return lstm_out, hidden_state
#
#
# import torch
# import torch.nn as nn
# import dgl
# import dgl.nn.pytorch as dglnn
#
# class GAT(nn.Module):
#     def __init__(self, graph, in_feats, out_feats, num_heads=2, dropout=0.2):
#         super(GAT, self).__init__()
#         self.graph = dgl.add_self_loop(graph)
#         self.num_heads = num_heads
#
#         # 使用 dgl 的 GATConv 层
#         self.gat_conv = dglnn.GATConv(
#             in_feats,  # 输入特征维度
#             out_feats,  # 输出特征维度
#             num_heads=num_heads,  # 注意力头的数量
#             feat_drop=dropout,  # 特征 dropout
#             attn_drop=dropout,  # 注意力 dropout
#             activation=nn.LeakyReLU(0.2),  # 激活函数
#         ).double()
#
#     def forward(self, x):
#         # 在图上执行图注意力卷积
#         x = self.gat_conv(self.graph, x)
#
#         # 将多头注意力的输出取平均
#         x = x.mean(dim=1)
#         return x
#
# class FPE_16(nn.Module):
#     def __init__(self):
#         super(FPE_16, self).__init__()
#         self.name = 'FPE_16'
#         self.lr = 0.0001
#         self.n_hosts = 16
#         self.n_feats = 3 * self.n_hosts  # 特征维度
#         self.n_window = 5
#         self.n_latent = 30
#         self.n_hidden = 16
#         self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
#
#         # 使用稀疏连接的 LSTM 替换 GRU
#         self.lstm = SparseLSTM(self.n_window, self.n_window, num_layers=2, connectivity_ratio=0.5, dropout=0.2)
#
#         # 定义 GAT 模型的图结构
#         src_ids = torch.arange(self.n_feats, dtype=torch.int32)  # 源节点 ID 为 0 到 n_feats-1
#         dst_ids = torch.full((self.n_feats,), self.n_feats, dtype=torch.int32)  # 目标节点 ID 统一为 n_feats
#         graph = dgl.graph((src_ids, dst_ids))
#
#         # 为边添加 'weight' 属性
#         graph.edata['weight'] = torch.ones(graph.number_of_edges(), dtype=torch.float64)
#
#         # 初始化 GAT
#         self.gat = GAT(graph, self.n_window, self.n_window, num_heads=2, dropout=0.2)
#
#         # 定义多头注意力机制
#         self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1).double()
#
#         # 定义编码器
#         self.encoder = nn.Sequential(
#             nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent).double(),
#             nn.LeakyReLU(True),
#         )
#
#         # 定义异常解码器
#         self.anomaly_decoder = nn.Sequential(
#             nn.Linear(self.n_latent, 2).double(),
#             nn.Softmax(dim=0),
#         )
#
#         # 定义原型解码器
#         self.prototype_decoder = nn.Sequential(
#             nn.Linear(self.n_latent, PROTO_DIM).double(),
#             nn.Sigmoid(),
#         )
#
#         # 初始化原型
#         self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.float64) for _ in range(3)]
#
#     def encode(self, t, s):
#         # 初始化隐藏状态
#         h = torch.randn(2, self.n_window, dtype=torch.float64)  # 2 层 LSTM，隐藏状态维度为 (num_layers, batch_size, hidden_size)
#
#         # 使用稀疏连接的 LSTM
#         lstm_t, _ = self.lstm(torch.t(t).unsqueeze(0), (h.unsqueeze(1), h.unsqueeze(1)))  # 2 层 LSTM
#         lstm_t = torch.t(lstm_t.squeeze(0))
#
#         # 构建图数据
#         graph = torch.cat((t, torch.zeros(self.n_window, 1, dtype=torch.float64)), dim=1)
#         gat_t = self.gat(torch.t(graph))
#         gat_t = torch.t(gat_t)
#
#         # 拼接 LSTM 和 GAT 的输出
#         concat_t = torch.cat((lstm_t.double(), gat_t.double()), dim=1)
#
#         # 使用多头注意力机制
#         o, _ = self.mha(concat_t, concat_t, concat_t)
#
#         # 编码器
#         t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)
#         return t
#
#     def anomaly_decode(self, t):
#         anomaly_scores = []
#         for elem in t:
#             anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))
#         return anomaly_scores
#
#     def prototype_decode(self, t):
#         prototypes = []
#         for elem in t:
#             prototypes.append(self.prototype_decoder(elem))
#         return prototypes
#
#     def forward(self, t, s):
#         t = self.encode(t.double(), s)  # 确保输入是 double 类型
#         anomaly_scores = self.anomaly_decode(t)
#         prototypes = self.prototype_decode(t)
#         return anomaly_scores, prototypes


################# 4   3层LSTM
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import dgl
# import os
#
# # 定义稀疏连接的 LSTM
# class SparseLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=3, connectivity_ratio=0.5):
#         super(SparseLSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.connectivity_ratio = connectivity_ratio
#
#         # 初始化 LSTM 的权重
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).double()
#
#         # 构造稀疏连接
#         self.sparse_weights = self._initialize_sparse_weights()
#
#     def _initialize_sparse_weights(self):
#         # 初始化随机图 g(V, p_ij)
#         n_input = self.input_size
#         n_hidden = self.hidden_size
#
#         # 计算阈值 Q
#         Q = n_input * n_hidden * self.connectivity_ratio
#
#         # 随机生成连接概率
#         p_ij = torch.rand(n_input, n_hidden, dtype=torch.float64)
#
#         # 根据阈值 Q 确定连接状态
#         sparse_mask = (p_ij > Q).double()
#
#         return sparse_mask
#
#     def forward(self, x, hidden_state=None):
#         batch_size, seq_len, _ = x.size()
#
#         # 应用稀疏连接
#         if self.sparse_weights is not None:
#             x = torch.matmul(x.double(), self.sparse_weights)
#
#         # 调整输出维度以匹配 LSTM 的 input_size
#         x = x.view(batch_size, seq_len, self.input_size)
#
#         # LSTM 前向传播
#         lstm_out, hidden_state = self.lstm(x, hidden_state)
#         return lstm_out, hidden_state
#
#
# # 定义 GAT 模型
# class GAT(nn.Module):
#     def __init__(self, graph, in_feats, out_feats):
#         super(GAT, self).__init__()
#         self.graph = graph
#         self.fc = nn.Linear(in_feats, out_feats).double()
#
#     def forward(self, x):
#         return self.fc(x.double())
#
#
# # 定义 FPE_16 模型
# class FPE_16(nn.Module):
#     def __init__(self):
#         super(FPE_16, self).__init__()
#         self.name = 'FPE_16'
#         self.lr = 0.0001
#         self.n_hosts = 16
#         self.n_feats = 3 * self.n_hosts  # 特征维度
#         self.n_window = 5
#         self.n_latent = 30
#         self.n_hidden = 16
#         self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
#
#         # 使用稀疏连接的 LSTM 替换 GRU
#         self.lstm = SparseLSTM(self.n_window, self.n_window, num_layers=3, connectivity_ratio=0.5)
#
#         # 定义 GAT 模型的图结构（关键修复点）
#         # 确保节点 ID 是整数且有效
#         src_ids = torch.arange(self.n_feats, dtype=torch.int32)  # 源节点 ID 为 0 到 n_feats-1
#         dst_ids = torch.full((self.n_feats,), self.n_feats, dtype=torch.int32)  # 目标节点 ID 统一为 n_feats
#         self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
#
#         # 定义多头注意力机制
#         self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1).double()
#
#         # 定义编码器
#         self.encoder = nn.Sequential(
#             nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent).double(),
#             nn.LeakyReLU(True),
#         )
#
#         # 定义异常解码器
#         self.anomaly_decoder = nn.Sequential(
#             nn.Linear(self.n_latent, 2).double(),
#             nn.Softmax(dim=0),
#         )
#
#         # 定义原型解码器
#         self.prototype_decoder = nn.Sequential(
#             nn.Linear(self.n_latent, PROTO_DIM).double(),
#             nn.Sigmoid(),
#         )
#
#         # 初始化原型
#         self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.float64) for _ in range(3)]
#
#     def encode(self, t, s):
#         # 初始化隐藏状态
#         h = torch.randn(3, self.n_window, dtype=torch.float64)  # 3 层 LSTM，隐藏状态维度为 (num_layers, batch_size, hidden_size)
#
#         # 使用稀疏连接的 LSTM
#         lstm_t, _ = self.lstm(torch.t(t).unsqueeze(0), (h.unsqueeze(1), h.unsqueeze(1)))  # 3 层 LSTM
#         lstm_t = torch.t(lstm_t.squeeze(0))
#
#         # 构建图数据
#         graph = torch.cat((t, torch.zeros(self.n_window, 1, dtype=torch.float64)), dim=1)
#         gat_t = self.gat(torch.t(graph))
#         gat_t = torch.t(gat_t)
#
#         # 拼接 LSTM 和 GAT 的输出
#         concat_t = torch.cat((lstm_t.double(), gat_t.double()), dim=1)
#
#         # 使用多头注意力机制
#         o, _ = self.mha(concat_t, concat_t, concat_t)
#
#         # 编码器
#         t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)
#         return t
#
#     def anomaly_decode(self, t):
#         anomaly_scores = []
#         for elem in t:
#             anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))
#         return anomaly_scores
#
#     def prototype_decode(self, t):
#         prototypes = []
#         for elem in t:
#             prototypes.append(self.prototype_decoder(elem))
#         return prototypes
#
#     def forward(self, t, s):
#         t = self.encode(t.double(), s)  # 确保输入是 double 类型
#         anomaly_scores = self.anomaly_decode(t)
#         prototypes = self.prototype_decode(t)
#         return anomaly_scores, prototypes
#
#
# # 定义生成器模型
# class Gen_16(nn.Module):
#     def __init__(self):
#         super(Gen_16, self).__init__()
#         self.name = 'Gen_16'
#         self.lr = 0.00005
#         self.n_hosts = 16
#         self.n_hidden = 64
#         self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
#         self.delta = nn.Sequential(
#             nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
#             nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
#         ).double()
#
#     def forward(self, e, s):
#         del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
#         return s + del_s.reshape(self.n_hosts, self.n_hosts)
#
#
# # 定义判别器模型
# class Disc_16(nn.Module):
#     def __init__(self):
#         super(Disc_16, self).__init__()
#         self.name = 'Disc_16'
#         self.lr = 0.00005
#         self.n_hosts = 16
#         self.n_hidden = 64
#         self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
#         self.probs = nn.Sequential(
#             nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
#             nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
#         ).double()
#
#     def forward(self, o, n):
#         probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
#         return probs


############### 普通LSTM
# import torch
# import torch.nn as nn
# import dgl
#
# # 定义 GAT 模型
# class GAT(nn.Module):
#     def __init__(self, graph, in_feats, out_feats):
#         super(GAT, self).__init__()
#         self.graph = graph
#         self.fc = nn.Linear(in_feats, out_feats)
#
#     def forward(self, x):
#         return self.fc(x)
#
#
# # 定义 FPE_16 模型
# PROTO_DIM = 10  # 假设 PROTO_DIM 的值
#
#
# class FPE_16(nn.Module):
#     def __init__(self):
#         super(FPE_16, self).__init__()
#         self.name = 'FPE_16'
#         self.lr = 0.0001
#         self.n_hosts = 16  # n_hosts 表示主机的数量，即模型中的设备数量。这个数量在 IIoT 中可以代表边缘设备的数量。
#         self.n_feats = 3 * self.n_hosts  # 每个设备有三个（特征）
#         self.n_window = 5  # w_size = 3
#         self.n_latent = 30  # 潜在空间的维度（n_latent），表示在潜在空间中用于表示数据特征的维度数目。  yuan 10  30
#         self.n_hidden = 16  # yuan 16
#         self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts  # 计算输入数据总维数
#
#         # 使用 LSTM 替换 GRU
#         self.lstm = nn.LSTM(self.n_window, self.n_window, 1)  # 3维的输入输出，使用1个LSTM层
#
#         src_ids = torch.tensor(list(range(self.n_feats)))
#         dst_ids = torch.tensor([self.n_feats] * self.n_feats)
#         self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window,
#                        self.n_window)  # 使用 dgl.graph 创建了一个图，其中节点之间的连接通过 src_ids 和 dst_ids 定义。
#         self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1)  # 从不同子空间提取信息，增强模型对关键特征的关注
#         self.encoder = nn.Sequential(
#             nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent), nn.LeakyReLU(True),
#         )
#         self.anomaly_decoder = nn.Sequential(
#             # 故障检测的解码器。它将 n_latent 的潜在特征映射到 2 维输出（可能表示“正常”与“故障”类），并通过 Softmax 激活函数得到概率分布。
#             nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
#         )
#         self.prototype_decoder = nn.Sequential(
#             nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),  # 使用 Sigmoid 激活函数输出一个值域在 [0, 1] 之间的预测，可能代表故障的严重程度等。
#         )
#         self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]
#
#     def encode(self, t, s):
#         # 初始化 LSTM 的隐藏状态和细胞状态
#         h = torch.randn(1, self.n_window, dtype=torch.double)  # 隐藏状态
#         c = torch.randn(1, self.n_window, dtype=torch.double)  # 细胞状态
#
#         # LSTM 前向传播
#         lstm_t, (h, c) = self.lstm(torch.t(t), (h, c))
#         lstm_t = torch.t(lstm_t)
#
#         # 构建图数据
#         graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
#
#         # GAT 处理
#         gat_t = self.gat(torch.t(graph))
#         gat_t = torch.t(gat_t)
#
#         # 拼接 LSTM 和 GAT 的输出
#         concat_t = torch.cat((lstm_t, gat_t), dim=1)
#
#         # 多头注意力机制
#         o, _ = self.mha(concat_t, concat_t, concat_t)
#
#         # 编码器
#         t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)
#         return t
#
#     def anomaly_decode(self, t):
#         anomaly_scores = []
#         for elem in t:
#             anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))
#         return anomaly_scores
#
#     def prototype_decode(self, t):
#         prototypes = []
#         for elem in t:
#             prototypes.append(self.prototype_decoder(elem))
#         return prototypes
#
#     def forward(self, t, s):
#         t = self.encode(t, s)
#         anomaly_scores = self.anomaly_decode(t)
#         prototypes = self.prototype_decode(t)
#         return anomaly_scores, prototypes
#
#
# # Generator Network : Input = Schedule, Embedding; Output = New Schedule
# class Gen_16(nn.Module):
#     def __init__(self):
#         super(Gen_16, self).__init__()
#         self.name = 'Gen_16'
#         self.lr = 0.00005
#         self.n_hosts = 16
#         self.n_hidden = 64
#         self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
#         self.delta = nn.Sequential(
#             nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
#             nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
#         )
#
#     def forward(self, e, s):
#         del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
#         return s + del_s.reshape(self.n_hosts, self.n_hosts)
#
#
# # Discriminator Network : Input = Schedule, New Schedule; Output = Likelihood scores
# class Disc_16(nn.Module):
#     def __init__(self):
#         super(Disc_16, self).__init__()
#         self.name = 'Disc_16'
#         self.lr = 0.00005
#         self.n_hosts = 16
#         self.n_hidden = 64
#         self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
#         self.probs = nn.Sequential(
#             nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
#             nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
#         )
#
#     def forward(self, o, n):
#         probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
#         return probs




# FPE
# class FPE_16(nn.Module):
#     def __init__(self):
#         super(FPE_16, self).__init__()
#         self.name = 'FPE_16'
#         self.lr = 0.0001
#         self.n_hosts = 16  # n_hosts 表示主机的数量，即模型中的设备数量。这个数量在 IIoT 中可以代表边缘设备的数量。
#         self.n_feats = 3 * self.n_hosts  # 每个设备有三个（特征）
#         self.n_window = 5  # w_size = 3
#         self.n_latent = 30  # 潜在空间的维度（n_latent），表示在潜在空间中用于表示数据特征的维度数目。  yuan 10  30
#         self.n_hidden = 16  # yuan 16
#         self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts  # 计算输入数据总维数
#         self.gru = nn.GRU(self.n_window, self.n_window, 1)  # 3维的输入输出，使用1个GRU层
#         # self.gru = nn.GRU(self.n_window, self.n_hidden, 2, bidirectional=True, dropout=0.2)  # 增加层数和 Dropout
#
#         # self.gru = nn.GRU(self.n_window, self.n_window, num_layers=2)    ###### 2层2D
#         # self.gru = nn.GRU(48, self.n_window, num_layers=2)    ###### 2层3D
#
#         src_ids = torch.tensor(list(range(self.n_feats)));
#         dst_ids = torch.tensor([self.n_feats] * self.n_feats)
#         self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window,
#                        self.n_window)  # 使用 dgl.graph 创建了一个图，其中节点之间的连接通过 src_ids 和 dst_ids 定义。
#         self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1)  # 从不同子空间提取信息，增强模型对关键特征的关注
#         self.encoder = nn.Sequential(
#             nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent), nn.LeakyReLU(True),
#         )
#         self.anomaly_decoder = nn.Sequential(
#             # 故障检测的解码器。它将 n_latent 的潜在特征映射到 2 维输出（可能表示“正常”与“故障”类），并通过 Softmax 激活函数得到概率分布。
#             nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
#         )
#         self.prototype_decoder = nn.Sequential(
#             nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),  # 使用 Sigmoid 激活函数输出一个值域在 [0, 1] 之间的预测，可能代表故障的严重程度等。
#         )
#         self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]
#
#     # 初始化了三个故障原型的张量，每个张量的大小为 PROTO_DIM，表示故障原型的特征。这些原型将用于故障检测和分类，可能与生成模型中使用的“原型”概念类似。
#     def encode(self, t, s):
#         h = torch.randn(1, self.n_window, dtype=torch.double)
#
#         # h = torch.randn(2, 3, dtype=torch.double)      ######## 2层2D
#         # h = torch.randn(2, 1, self.n_window, dtype=torch.double)        ######## 2层3D
#         # t = torch.randn(self.n_window, 1, self.n_feats)         ########  2层3D
#         # t = t.double()  # 将输入数据转换为 torch.float64
#         # h = h.double()  # 将隐藏状态转换为 torch.float64
#         # gru_t, _ = self.gru(t, h)           ########  2层3D
#
#         gru_t, _ = self.gru(torch.t(t), h)
#         gru_t = torch.t(gru_t)
#         graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
#
#         # graph = torch.cat((t, torch.zeros(self.n_window, 1, 1)), dim=2)      ######### 2层3D
#
#         gat_t = self.gat(torch.t(graph))
#         gat_t = torch.t(gat_t)
#         concat_t = torch.cat((gru_t, gat_t), dim=1)
#
#         # concat_t = torch.cat((gru_t, gat_t), dim=2)        ######### 2层3D
#
#         o, _ = self.mha(concat_t, concat_t, concat_t)
#         t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)
#         return t
#
#     def anomaly_decode(self, t):
#         anomaly_scores = []
#         for elem in t:
#             anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))
#         return anomaly_scores
#
#     def prototype_decode(self, t):
#         prototypes = []
#         for elem in t:
#             prototypes.append(self.prototype_decoder(elem))
#         return prototypes
#
#     def forward(self, t, s):
#         t = self.encode(t, s)
#         anomaly_scores = self.anomaly_decode(t)
#         prototypes = self.prototype_decode(t)
#         return anomaly_scores, prototypes
#
#
# # Generator Network : Input = Schedule, Embedding; Output = New Schedule
# class Gen_16(nn.Module):
#     def __init__(self):
#         super(Gen_16, self).__init__()
#         self.name = 'Gen_16'
#         self.lr = 0.00005
#         self.n_hosts = 16
#         self.n_hidden = 64
#         self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
#         self.delta = nn.Sequential(
#             nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
#             nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
#         )
#
#     def forward(self, e, s):
#         del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
#         return s + del_s.reshape(self.n_hosts, self.n_hosts)
#
#
# # Discriminator Network : Input = Schedule, New Schedule; Output = Likelihood scores
# class Disc_16(nn.Module):
#     def __init__(self):
#         super(Disc_16, self).__init__()
#         self.name = 'Disc_16'
#         self.lr = 0.00005
#         self.n_hosts = 16
#         self.n_hidden = 64
#         self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
#         self.probs = nn.Sequential(
#             nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
#             nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
#         )
#
#     def forward(self, o, n):
#         probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
#         return probs


## FPE
class FEE_50(nn.Module):
    def __init__(self):
        super(FEE_50, self).__init__()
        self.name = 'FEE_50'
        self.lr = 0.0001
        self.n_hosts = 50
        self.n_feats = 3 * self.n_hosts
        self.n_window = 3  # w_size = 5
        self.n_latent = 10
        self.n_hidden = 50
        self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
        self.gru = nn.GRU(self.n_window, self.n_window, 1)
        src_ids = torch.tensor(list(range(self.n_feats)));
        dst_ids = torch.tensor([self.n_feats] * self.n_feats)
        self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
        self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent), nn.LeakyReLU(True),
        )
        self.anomaly_decoder = nn.Sequential(
            nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
        )
        self.prototype_decoder = nn.Sequential(
            nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
        )
        self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

    def encode(self, t, s):
        h = torch.randn(1, self.n_window, dtype=torch.double)
        gru_t, _ = self.gru(torch.t(t), h)
        gru_t = torch.t(gru_t)
        graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
        gat_t = self.gat(torch.t(graph))
        gat_t = torch.t(gat_t)
        concat_t = torch.cat((gru_t, gat_t), dim=1)
        o, _ = self.mha(concat_t, concat_t, concat_t)
        t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)
        return t

    def anomaly_decode(self, t):
        anomaly_scores = []
        for elem in t:
            anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))
        return anomaly_scores

    def prototype_decode(self, t):
        prototypes = []
        for elem in t:
            prototypes.append(self.prototype_decoder(elem))
        return prototypes

    def forward(self, t, s):
        t = self.encode(t, s)
        anomaly_scores = self.anomaly_decode(t)
        prototypes = self.prototype_decode(t)
        return anomaly_scores, prototypes


## Simple Multi-Head Self-Attention Model
class Attention_50(nn.Module):
    def __init__(self):
        super(Attention_50, self).__init__()
        self.name = 'Attention_50'
        self.lr = 0.0008
        self.n_hosts = 50
        self.n_feats = 3 * self.n_hosts
        self.n_window = 3  # w_size = 5
        self.n_latent = 10
        self.n_hidden = 16
        self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
        # self.atts = [ nn.Sequential( nn.Linear(self.n, self.n_feats * self.n_feats),
        # 		nn.Sigmoid())	for i in range(1)]
        # self.encoder_atts = nn.ModuleList(self.atts)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_window * self.n_feats, self.n_hosts * self.n_latent), nn.LeakyReLU(True),
        )
        self.anomaly_decoder = nn.Sequential(
            nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
        )
        self.prototype_decoder = nn.Sequential(
            nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
        )
        self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

    def encode(self, t, s):
        # for at in self.encoder_atts:
        # 	inp = torch.cat((t.view(-1), s.view(-1)))
        # 	ats = at(inp).reshape(self.n_feats, self.n_feats)
        # 	t = torch.matmul(t, ats)
        t = self.encoder(t.view(-1)).view(self.n_hosts, self.n_latent)
        return t

    def anomaly_decode(self, t):
        anomaly_scores = []
        for elem in t:
            anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))
        return anomaly_scores

    def prototype_decode(self, t):
        prototypes = []
        for elem in t:
            prototypes.append(self.prototype_decoder(elem))
        return prototypes

    def forward(self, t, s):
        t = self.encode(t, s)
        anomaly_scores = self.anomaly_decode(t)
        prototypes = self.prototype_decode(t)
        return anomaly_scores, prototypes


# Generator Network : Input = Schedule, Embedding; Output = New Schedule
class Gen_50(nn.Module):
    def __init__(self):
        super(Gen_50, self).__init__()
        self.name = 'Gen_50'
        self.lr = 0.00003
        self.n_hosts = 50
        self.n_hidden = 64
        self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
        self.delta = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
        )

    def forward(self, e, s):
        del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
        return s + del_s.reshape(self.n_hosts, self.n_hosts)


# Discriminator Network : Input = Schedule, New Schedule; Output = Likelihood scores
class Disc_50(nn.Module):
    def __init__(self):
        super(Disc_50, self).__init__()
        self.name = 'Disc_50'
        self.lr = 0.00003
        self.n_hosts = 50
        self.n_hidden = 64
        self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
        self.probs = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
        )

    def forward(self, o, n):
        probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
        return probs

############## PreGANPlus Models ##############

# Transformer Model
# class Transformer_16(nn.Module):
# 	def __init__(self):
# 		super(Transformer_16, self).__init__()
# 		self.name = 'Transformer_16'
# 		self.lr = 0.0001
# 		self.n_hosts = 16
# 		feats = 3 * self.n_hosts
# 		self.n_feats = 3 * self.n_hosts
# 		self.n_window = 3 # w_size = 5
# 		self.n_latent = 10
# 		self.n_hidden = 16
# 		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
# 		src_ids = torch.tensor(list(range(self.n_feats))); dst_ids = torch.tensor([self.n_feats] * self.n_feats)
# 		self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
# 		self.time_encoder = nn.Sequential(
# 			nn.Linear(feats, feats * 2 + 1),
# 		)
# 		self.pos_encoder = PositionalEncoding(feats * 2 + 1, 0.1, self.n_window)
# 		encoder_layers = TransformerEncoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
# 		self.encoder = TransformerEncoder(encoder_layers, 1)
# 		a_decoder_layers = TransformerDecoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
# 		self.anomaly_decoder = TransformerDecoder(a_decoder_layers, 1)
# 		self.anomaly_decoder2 = nn.Sequential(
# 			nn.Linear((feats * 2 + 1) * self.n_window * self.n_window, 2 * self.n_hosts),
# 		)
# 		self.softm = nn.Softmax(dim=1)
# 		p_decoder_layers = TransformerDecoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
# 		self.prototype_decoder = TransformerDecoder(p_decoder_layers, 1)
# 		self.prototype_decoder2 = nn.Sequential(
# 			nn.Linear((feats * 2 + 1) * self.n_window * self.n_window, PROTO_DIM * self.n_hosts),
# 		)
# 		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]
#
# 	def encode(self, t, s):
# 		t = torch.squeeze(t, 1)
# 		graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
# 		gat_t = self.gat(torch.t(graph))
# 		gat_t = torch.t(gat_t)
# 		o = torch.cat((t, gat_t), dim=1)
# 		t = o * math.sqrt(self.n_feats)
# 		t = self.pos_encoder(t) # window size, batch size (1), feats (3 metrics * 16 hosts)
# 		memory = self.encoder(t)
# 		return memory
#
# 	def anomaly_decode(self, t, memory):
# 		anomaly_scores = self.anomaly_decoder(t, memory)
# 		anomaly_scores = self.anomaly_decoder2(anomaly_scores.view(-1)).view(-1, 1, 2)
# 		return anomaly_scores
#
# 	def prototype_decode(self, t, memory):
# 		prototypes = self.prototype_decoder(t, memory)
# 		prototypes = self.prototype_decoder2(prototypes.view(-1)).view(-1, PROTO_DIM)
# 		return prototypes
#
# 	def forward(self, t, s):
# 		encoded_t = self.time_encoder(t).unsqueeze(dim=1).expand(-1, self.n_window, -1)
# 		t = t.unsqueeze(dim=1)
# 		memory = self.encode(t, s)
# 		anomaly_scores = self.anomaly_decode(encoded_t, memory)
# 		prototypes = self.prototype_decode(encoded_t, memory)
# 		return anomaly_scores, prototypes

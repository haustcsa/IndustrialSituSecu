# Directory paths
model_folder = 'recovery/SGATGANSrc/checkpoints/'
model_plus_folder = 'recovery/SGATGANSrc/checkpointsplus/'
data_folder = 'recovery/SGATGANSrc/data/'
plot_folder = 'recovery/SGATGANSrc/plots'
data_filename = 'time_series.npy'  # 可能是用于训练的时间特征
schedule_filename = 'schedule_series.npy'  # 可能是用于建模的调度序列

# Hyperparameters
num_epochs = 5  # 训练模型的总轮数
PERCENTILES = 20  # 用于计算百分位数的超参数，可能在某些统计或模型正则化过程中使用。yuan 98
PROTO_DIM = 2  # Prototype（原型）的维度，可能与模型的输出层或解码器有关
PROTO_UPDATE_FACTOR = 0.4  # 原型更新的初始因子，用于控制模型中原型参数更新的速度。yuan 0.2  0.4zuigao
PROTO_UPDATE_MIN = 0.02   # 原型更新因子的最小值，防止更新因子过小，影响模型性能。yuan 0.02
PROTO_FACTOR_DECAY = 0.05   # 原型更新因子的衰减率，随着训练进行，更新速度逐渐减慢。 0.995  0.05最高
LATEST_WINDOW_SIZE = 20   # 窗口大小超参数，可能用于滑动窗口操作（如时间序列处理或实时评估） yuan 10

percentile_multiplier = 0.99999 #########################

# GAN parameters  这些系数表明在当前配置下，模型更关注能量效率（权重为 0.8），而对延迟的优化相对次要（权重为 0.2）
Coeff_Energy = 0.95   # 能量（资源消耗）相关的系数，用于权衡生成对抗网络（GAN）中能量与其他目标之间的平衡  0.8
Coeff_Latency = 0.2   # 延迟相关的系数，用于衡量 GAN 模型中延迟的重要性   0.2


# Hyperparameters
# num_epochs = 30  # 训练模型的总轮数
# PERCENTILES = 98  # 用于计算百分位数的超参数，可能在某些统计或模型正则化过程中使用。
# PROTO_DIM = 2  # Prototype（原型）的维度，可能与模型的输出层或解码器有关
# PROTO_UPDATE_FACTOR = 0.2  # 原型更新的初始因子，用于控制模型中原型参数更新的速度。
# PROTO_UPDATE_MIN = 0.02   # 原型更新因子的最小值，防止更新因子过小，影响模型性能。
# PROTO_FACTOR_DECAY = 0.05   # 原型更新因子的衰减率，随着训练进行，更新速度逐渐减慢。
# LATEST_WINDOW_SIZE = 5  # 窗口大小超参数，可能用于滑动窗口操作（如时间序列处理或实时评估）
#
# # GAN parameters  这些系数表明在当前配置下，模型更关注能量效率（权重为 0.8），而对延迟的优化相对次要（权重为 0.2）
# Coeff_Energy = 0.8   # 能量（资源消耗）相关的系数，用于权衡生成对抗网络（GAN）中能量与其他目标之间的平衡
# Coeff_Latency = 0.2   # 延迟相关的系数，用于衡量 GAN 模型中延迟的重要性
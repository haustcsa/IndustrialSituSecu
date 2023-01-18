# 核密度估计图：
# 核密度估计是概率论中用来估计未知的密度函数，属于非参数校验方法之一。
# 通过核密度估计图可以比较直观的看出数据样本本身的分布特征。

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# fig, axes = plt.subplots(2, 3)
# kd_train = pd.read_csv(r'Pearson-KDD/kdd_train_header.csv')
# kd_test = pd.read_csv(r'Pearson-KDD/kdd_test_header.csv')
kd_train = pd.read_csv(r'KDDTrain+ calss-5string.csv')
kd_test = pd.read_csv(r'KDDTest+ class-5string.csv')
print(kd_train.columns)

sns.set()
x1 = kd_train["class"]
x2 = kd_test["class"]
# ax = sns.distplot(x)
fig = plt.figure()
ax1 = fig.add_subplot(121)
x_ticks = np.linspace(0, 4, 5)
plt.xticks(x_ticks)
ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)

ax = sns.distplot(x1, bins=np.arange(-0.5, 5.5), hist_kws={'label': 'hist'},
                  kde_kws={'color': 'blue', 'label': 'kde'},
                  axlabel='attack type of training set',  kde=True, ax=ax1)
ax = sns.distplot(x2, bins=np.arange(-0.5, 5.5), hist_kws={'label': 'hist'},
                  kde_kws={'color': 'blue', 'label': 'kde'},
                  axlabel='attack type of testing set',  kde=True, ax=ax2)
# fig.legend(labels=["kde"])
plt.legend()
plt.show()

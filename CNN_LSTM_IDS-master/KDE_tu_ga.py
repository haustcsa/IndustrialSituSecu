import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# fig, axes = plt.subplots(2, 3)
ga_train = pd.read_csv(r'gas\15-8-gas_tr.csv')
ga_test = pd.read_csv(r'gas\15-8-gas_te.csv')
# kd_train = pd.read_csv(r'G:\SJH---\git\sklearn-NaiveBayes-NB15-master\kdd-train-CN.csv')
# kd_test = pd.read_csv(r'G:\SJH---\git\sklearn-NaiveBayes-NB15-master\kdd-test-CN.csv')
print(ga_train.columns)
# x = kd_train["classN"]
# sns.displot(data=kd_train, x=kd_train['classN'], kde=True, rug=True,
# label='train')
sns.set()
x1 = ga_train["categorized result"]
x2 = ga_test["categorized result"]
# ax = sns.distplot(x)
'''
x_ticks = np.linspace(0, 5, 6)
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
plt.xticks(x_ticks)
ax = sns.distplot(x1, bins=np.arange(0.5, 4.5), hist_kws={"rwidth":1,}, axlabel='attack type of training set', kde=True, ax=ax1)
ax = sns.distplot(x2, bins=np.arange(0.5, 4.5), hist_kws={"rwidth":1,}, axlabel='attack type of testing set', kde=True, ax=ax2)
'''
fig = plt.figure()
ax1 = fig.add_subplot(121)
x_ticks = np.linspace(0, 7, 8)
plt.xticks(x_ticks)
ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)

ax = sns.distplot(x1, bins=np.arange(-0.5, 8.5), hist_kws={'label': 'hist'},
                  kde_kws={'color': 'blue', 'label': 'kde'},
                  axlabel='attack type of training set',  kde=True, ax=ax1)
ax = sns.distplot(x2, bins=np.arange(-0.5, 8.5), hist_kws={'label': 'hist'},
                  kde_kws={'color': 'blue', 'label': 'kde'},
                  axlabel='attack type of testing set',  kde=True, ax=ax2)
# fig.legend(labels=["kde"])
plt.legend()
plt.show()
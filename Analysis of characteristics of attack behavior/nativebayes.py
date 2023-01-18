import numpy as np
import pandas as pd
dataset= pd.read_csv('UNSW_NB15_type_1.csv')
#randon forest
X = dataset.loc[ : , ['A','B','D','F','G','H','I','J','K','L','M','P','Q','X','Y','Z','AA','AB','AE','AF','AI','AJ','AK','AP']].values
#XG-Boost
#X = dataset.loc[ : , ['B','C','G','H','J','X','Y','AD','AF','AI','AJ']].values
#langerange
#X = dataset.loc[ : , ['C','D','J','K','T']].values

y = dataset.loc[ : , ['AQ']].values

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

model = GaussianNB()
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3195, random_state=None)
sss.get_n_splits(X, y)

cm_sum = np.zeros((2,2))

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_sum = cm_sum + cm
print('\nNaive Bayes Gaussian Algorithms')
print('\nConfusion Matrix')
print('_'*20)
print('     Predicted')
print('     pos neg')
print('pos: %i %i' % (cm_sum[1,1], cm_sum[0,1]))
print('neg: %i %i' % (cm_sum[1,1], cm_sum[0,1]))

from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
X = dataset.drop(['Buy_Sell'], axis=1).values
Y = dataset['Buy_Sell'].values
# 创建高斯朴素贝叶斯实例
clf = GaussianNB()
# 使用sigmoid校准创建校准交叉验证
clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
# 校准的概率
clf_sigmoid.fit(X, Y)
"""
CalibratedClassifierCV(base_estimator=GaussianNB(priors=None, var_smoothing=1e-09),
            cv=2, method='sigmoid')
"""

# 创建新观察数据
#new_observation = [[.4, .4, .4, .4, .4, .4, .4, .4, .4]]
#clf_sigmoid.predict_proba(new_observation)
#array([[0.50353248, 0.49646752]])
#clf_sigmoid.score(X,Y)

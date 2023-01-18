import pandas as pd


#dataset['Buy_Sell'] = dataset['Buy_Sell'].astype('int')
#X = np.asarray(dataset[['open', 'high', 'low', 'close', 'volume']])
#X = np.asarray(dataset[['open', 'high', 'low', 'close', 'volume']])
#y = np.asarray(dataset['Buy_Sell'])

dataset= pd.read_csv('UNSW_NB15_type_1.csv')
#randon forest
#X = dataset.loc[ : , ['A','B','D','F','G','H','I','J','K','L','M','P','Q','X','Y','Z','AA','AB','AE','AF','AI','AJ','AK','AP']].values
#XG-Boost
#X = dataset.loc[ : , ['B','C','G','H','J','X','Y','AD','AF','AI','AJ']].values
#langerange
#X = dataset.loc[ : , ['C','D','J','K','T']].values

#y = dataset.loc[ : , ['AQ']].values

dataset= pd.read_csv('gas.csv')
#X = dataset.loc[ : , ['A','G','H','M','N','Q']].values
#X = dataset.loc[ : , ['A','B','G','H','J','N','Q']].values
X = dataset.loc[ : , ['A','H','M','Q']].values
y = dataset.loc[ : , ['label']].values

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3195, random_state = 0)



# 模型训练
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
yhat = LR.predict(X_test)

# predict_proba是所有类的估计值的返回，按类的标签排序。
# 第1列是第1类P(Y=1|X)的概率，第二列是第0类P(Y=0|X)的概率

yhat_prob = LR.predict_proba(X_test)

from sklearn.metrics import classification_report, confusion_matrix 
print (classification_report(y_test, yhat))

y_pred_proba = LR.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(12,6))
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

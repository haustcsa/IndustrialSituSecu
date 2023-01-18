import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


#dataset = pd.read_csv('UNSW_NB15_training-set.csv')

#X = dataset.iloc[:,0:42]
#y = dataset.iloc[:,42]

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=126)
#model = XGBClassifier()
#model.fit(X_train,y_train)
#print(model.feature_importances_)
#data = pd.DataFrame(model.feature_importances_)
#data.columns = ['featureimportances']

#pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
#pyplot.ylabel('Importance')
#pyplot.title('Feature Importances')
#pyplot.show()

train_data= pd.read_csv('gas_train.csv')
test_data= pd.read_csv('gas_test.csv')



#train_data= pd.read_csv('UNSW_NB15_training-set.csv')
#test_data= pd.read_csv('UNSW_NB15_testing-set.csv')

train_Y=np.array(train_data['label'])
train_X=train_data.drop(['label'],axis=1)
train_X_column_name=list(train_X.columns)
train_X=np.array(train_X)

test_Y=np.array(test_data['label'])
test_X=test_data.drop(['label'],axis=1)
test_X=np.array(test_X)

xg_model = XGBClassifier()
xg_model.fit(train_X,train_Y)
print(xg_model.feature_importances_)
data = pd.DataFrame(xg_model.feature_importances_)
data.columns = ['featureimportances']


xg_importance=list(xg_model.feature_importances_)
xg_feature_importance=[(feature,round(importance,6)) 
                                  for feature, importance in zip(train_X_column_name,xg_importance)]
xg_feature_importance=sorted(xg_feature_importance,key=lambda x:x[1],reverse=True)
plt.figure(1)
plt.clf()
importance_plot_x_values=list(range(len(xg_importance)))
plt.bar(importance_plot_x_values,xg_importance,orientation='vertical')
for a, b in zip(importance_plot_x_values,xg_importance):
    plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=8)
plt.xticks(importance_plot_x_values,train_X_column_name,rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()

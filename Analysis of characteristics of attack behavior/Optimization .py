import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

train_data= pd.read_csv('UNSW_NB15_training-set.csv')
test_data= pd.read_csv('UNSW_NB15_testing-set.csv')

#train_data= pd.read_csv('gas_train.csv')
#test_data= pd.read_csv('gas_test.csv')

train_Y=np.array(train_data['label'])
train_X=train_data.drop(['label'],axis=1)
train_X_column_name=list(train_X.columns)
train_X=np.array(train_X)

test_Y=np.array(test_data['label'])
test_X=test_data.drop(['label'],axis=1)
test_X=np.array(test_X)


# 特征提取
model = ExtraTreesClassifier()

# X_embedded = SelectFromModel(model,threshold=0.005).fit_transform(X,Y)
model.fit(train_X,train_Y)
print(model.feature_importances_)

et_importance=list(model.feature_importances_)
et_feature_importance=[(feature,round(importance,6))
                                  for feature, importance in zip(train_X_column_name,et_importance)]
et_feature_importance=sorted(et_feature_importance,key=lambda x:x[1],reverse=True)
plt.figure(1)
plt.clf()
importance_plot_x_values=list(range(len(et_importance)))
plt.bar(importance_plot_x_values,et_importance,orientation='vertical')
for a, b in zip(importance_plot_x_values,et_importance):
    plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=8)
plt.xticks(importance_plot_x_values,train_X_column_name,rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()
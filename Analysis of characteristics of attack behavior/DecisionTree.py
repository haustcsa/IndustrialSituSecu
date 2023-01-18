import pandas as pd

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3195)

from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix  
#print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 


#import numpy as np
import pandas as pd
#import os
#import sys
#import time
#import math
#import matplotlib.pyplot as plt
#from pyecharts.charts import Line
#from keras.models import Sequential  # before using keras, install Tensorflow first
#from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.recurrent import LSTM
#from keras.layers.convolutional import Conv1D
#from keras.layers.pooling import MaxPooling1D
#from sklearn.metrics import accuracy_score
#import sklearn.preprocessing as prep

#UNSW_NB15_Type_two
#dataset= pd.read_csv('UNSW_NB15_type_1.csv')
#randon forest
#X = dataset.loc[ : , ['A','B','D','F','G','H','I','J','K','L','M','P','Q','X','Y','Z','AA','AB','AE','AF','AI','AJ','AK','AP']].values
#XG-Boost
#X = dataset.loc[ : , ['B','C','G','H','J','X','Y','AD','AF','AI','AJ']].values
#langerange
#X = dataset.loc[ : , ['C','D','J','K','T']].values

#y = dataset.loc[ : , ['AQ']].values
#time_start = time.time()

#gas
dataset= pd.read_csv('gas.csv')
#X = dataset.loc[ : , ['A','G','H','M','N','Q']].values
#X = dataset.loc[ : , ['A','B','G','H','J','N','Q']].values
X = dataset.loc[ : , ['A','H','M','Q']].values
y = dataset.loc[ : , ['label']].values



#print(X.shape)
#print(y.shape)

#print(X.dtype)
#y = y.astype(int)
#print(y.dtype)

#print(np.isnan(dataset).any())
#dataset.dropna(inplace=True)
#X = np.nan_to_num（X.astype(np.float64))


from sklearn.model_selection import train_test_split  
#X_train, X_test, y_train, y_test = cross_validation. train_test_split(X, y, test_size=0.3195, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3195)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)



from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier  
knn = KNeighborsClassifier(n_neighbors=5)  
knn.fit(X_train, y_train) 
y_pred = knn.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix  
#print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#time_end = time.time()
#time_c = time_end - time_start
#print('time cost = ', time_c, 's')

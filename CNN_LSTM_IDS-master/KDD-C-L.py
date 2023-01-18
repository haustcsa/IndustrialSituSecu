#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from keras import regularizers
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# For the original '99 KDD dataset: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

from sklearn.utils import shuffle

with open('kddcup2.names', 'r') as infile:
    kdd_names = infile.readlines()
kdd_cols = [x.split(':')[0] for x in kdd_names[1:]]  # 读取18个列名

kdd_cols += ['class']

print('多分类')
kdd = pd.read_csv('Pearson-KDD/kdd_train.csv', names=kdd_cols)
kdd_t = pd.read_csv('Pearson-KDD/kdd_test.csv', names=kdd_cols)
# # # print('二分类')
# # # kdd = pd.read_csv('Pearson-KDD/kdd_train2.csv', names=kdd_cols)
# # # kdd_t = pd.read_csv('Pearson-KDD/kdd_test2.csv', names=kdd_cols)

kdd = shuffle(kdd, random_state=42)

# kdd = shuffle(kdd)
# kdd_t = shuffle(kdd_t, random_state=42)

kdd_cols = [kdd.columns[0]] + sorted(list(set(kdd.protocol_type.values))) + sorted(
    list(set(kdd.service.values))) + sorted(list(set(kdd.flag.values))) + kdd.columns[4:].tolist()


def cat_encode(df, col):
    return pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col].values)], axis=1)


def log_trns(df, col):
    return df[col].apply(np.log1p)


cat_lst = ['protocol_type', 'service', 'flag']
for col in cat_lst:
    kdd = cat_encode(kdd, col)
    kdd_t = cat_encode(kdd_t, col)

log_lst = ['duration']
for col in log_lst:
    kdd[col] = log_trns(kdd, col)
    kdd_t[col] = log_trns(kdd_t, col)

kdd = kdd[kdd_cols]
for col in kdd_cols:
    if col not in kdd_t.columns:
        kdd_t[col] = 0

kdd_t = kdd_t[kdd_cols]

kdd.head()

target = kdd.pop('class')
y_test = kdd_t.pop('class')

target = pd.get_dummies(target)  # train的5类one-hot带序号
y_test = pd.get_dummies(y_test)  # test的5类one-hot带序号

target = target.values  # train的5类one-hot
train = kdd.values  # drop diff和class后的其余122列train data
test = kdd_t.values  # drop diff和class后的其余122列test data
y_test = y_test.values  # test的5类one-hot

# We rescale features to [0, 1]


min_max_scaler = MinMaxScaler()
train = min_max_scaler.fit_transform(train)
test = min_max_scaler.transform(test)

for idx, col in enumerate(list(kdd.columns)):  # 遍历122列索引及特征名
    print(idx, col)

from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional, LSTM, \
    RepeatVector


def build_network():
    model = Sequential()
    model.add(Conv1D(32, kernel_size=5, strides=1,
                     activation='leaky_relu', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=(97, 1)))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(64, 5, activation='leaky_relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.3))
    # model.add(Dense(16, activation='leaky_relu'))
    model.add(RepeatVector(3))
    model.add(LSTM(60, input_shape=(1, 42), return_sequences=True, activation='leaky_relu'))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    # model.add(Bidirectional(LSTM(2, input_shape=(1, 64), return_sequences=True, activation='sigmoid')))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# We use early stopping on a holdout validation set

NN = build_network()  # ←CNN

print(NN.summary())

best_weights_filepath = 'kdd_models/best_weights.hdf5'
checkpoint = ModelCheckpoint(best_weights_filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=10)


history = History()

callbacks_list = [checkpoint, history, early_stopping_monitor]

history = NN.fit(x=np.expand_dims(train, axis=2), y=target, epochs=200, shuffle=True,
                 validation_split=0.3,
                 # validation_data=(np.expand_dims(test, axis=2), y_test),
                 batch_size=350,
                 callbacks=callbacks_list)


print(y_test)

from sklearn.metrics import confusion_matrix

# preds = NN.predict(test)
preds = NN.predict(np.expand_dims(test, axis=2))
pred_lbls = np.argmax(preds, axis=1)
true_lbls = np.argmax(y_test, axis=1)

print(test.shape)
print(y_test.shape)

NN.evaluate(np.expand_dims(test, axis=2), y_test)
# NN.evaluate(test, y_test)

confusion_matrix(true_lbls, pred_lbls)

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

a = accuracy_score(true_lbls, pred_lbls)
# average 为'macro' 为宏平均，每个类别单独计算F1取算术平均为全局指标，这种方式平等对待每个类别，其值主要受稀有类别影响，更能体现模型在稀有类别的表现
# average 为'micro' 为微平均，指先累加各个类别的tp、fp、tn、fn值，再由这些值来计算F1值，这种方式平等地对待每个样本，其值主要受常见类别的影响
f = f1_score(true_lbls, pred_lbls, average='macro')
p = precision_score(true_lbls, pred_lbls, average='macro')
r = recall_score(true_lbls, pred_lbls, average='macro')
print(a, f, p, r)


from sklearn.metrics import multilabel_confusion_matrix

conf = multilabel_confusion_matrix(true_lbls, pred_lbls)

for i in range(conf.shape[0]):
    tn = conf[i][0][0]
    fp = conf[i][0][1]
    fn = conf[i][1][0]
    tp = conf[i][1][1]
    acc = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    precision = tp / (tp + fp)
    F1_score = 2 * tp / (2 * tp + fp + fn)
    print("TPR(recall)", float("{0:5f}".format(tpr)) * 100)
    print("FPR", float("{0:5f}".format(fpr)) * 100)
    print("Precision", float("{0:5f}".format(precision)) * 100)
    print("F1-Score", float("{0:5f}".format(F1_score)) * 100)
    print("Accuracy", float("{0:5f}".format(acc)) * 100)
    print("-------------")
    print('多分类')

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], linestyle='--')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], linestyle='--')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

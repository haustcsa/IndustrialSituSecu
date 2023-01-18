
import time

import pandas as pd
import numpy as np
from keras import regularizers
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

# # # ga = pd.read_csv(r'gas\10_gas_final.arff.csv')
ga = pd.read_csv(r'gas\15-gas_tr.csv')
ga_t = pd.read_csv(r'gas\15-gas_te.csv')

# ga = shuffle(ga, random_state=4)
ga = shuffle(ga, random_state=42)

ga.head()

target = ga.pop('categorized result')
dp1 = ga.pop('pressure measurement')  # 该列相关度极小，且值间差距极大
dp2 = ga_t.pop('pressure measurement')
dp3 = ga.pop('binary result')
dp4 = ga_t.pop('binary result')
y_test = ga_t.pop('categorized result')

##y_test.head(200)

target = pd.get_dummies(target)  # train的8类one-hot带序号
y_test = pd.get_dummies(y_test)  # test的8类one-hot带序号

target = target.values  # train的8类one-hot
train = ga.values  # drop measurement result后的其余17列train data
test = ga_t.values  # drop measurement result后的其余17列test data
y_test = y_test.values  # test的8类one-hot

min_max_scaler = MinMaxScaler()
train = min_max_scaler.fit_transform(train)
test = min_max_scaler.transform(test)

for idx, col in enumerate(list(ga.columns)):  # 遍历17列索引及特征名
    print(idx, col)

from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional, LSTM, \
    RepeatVector


def build_network():
    model = Sequential()
    model.add(Conv1D(16, kernel_size=5, strides=1,
                     activation='leaky_relu',
                     input_shape=(17, 1)))
    model.add(MaxPooling1D(pool_size=2, strides=1))
    model.add(Conv1D(32, 5, activation='leaky_relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # model.add(Dropout(0.3))
    model.add(Dense(16, activation='leaky_relu'))
    model.add(RepeatVector(3))

    model.add(LSTM(60, input_shape=(1, 16), return_sequences=True, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    # model.add(Bidirectional(LSTM(2, input_shape=(1, 64), return_sequences=True, activation='sigmoid')))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


NN = build_network()

print(NN.summary())


filename = 'gas_models/best_weights'
best_weights_filepath = filename + time.strftime('%Y-%m-%d %H-%M-%S') + '.hdf5'
checkpoint = ModelCheckpoint(best_weights_filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=20)

history = History()

callbacks_list = [checkpoint, history, early_stopping_monitor]

history = NN.fit(x=np.expand_dims(train, axis=2), y=target, epochs=300, shuffle=True,
                 validation_data=(np.expand_dims(test, axis=2), y_test),
                 batch_size=32,
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
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


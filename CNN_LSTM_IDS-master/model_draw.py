from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional, LSTM, \
    RepeatVector
from keras import regularizers
import os
os.environ["PATH"] += os.pathsep + 'D:/graphviz/bin/'

from keras.utils.vis_utils import plot_model



def build_cnn_network():
    model = Sequential()
    model.add(Conv1D(32, kernel_size=5, strides=1,
                     activation='leaky_relu', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=(97, 1)))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(64, 5, activation='leaky_relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='leaky_relu'))
    model.add(RepeatVector(3))
    # urn_sequences=True))r
    model.add(LSTM(60, input_shape=(1, 42), return_sequences=True, activation='leaky_relu'))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    # model.add(Bidirectional(LSTM(2, input_shape=(1, 64), return_sequences=True, activation='sigmoid')))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    model = build_cnn_network()
    model.summary()
    # keras自带函数模型可视化以图片展示
    plot_model(model, show_shapes=True, to_file='Pearson-KDD/model.png')

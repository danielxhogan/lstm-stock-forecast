%load_ext tensorboard

import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

!ls "/content/drive/My Drive/Colab Notebooks/LSTM-stock-forecast/AMD"

PATH = '/content/drive/My Drive/Colab Notebooks/LSTM-stock-forecast/AMD'
EPOCHS = 10
BATCH_SIZE = 64
NAME = 'AMD-5-DAYS-BACK-1-DAY-FORWARD'

NODES = 128
DROPOUT = 0.02

train_X = np.load(f'{PATH}/train_X.npy', allow_pickle=True)
train_y = np.load(f'{PATH}/train_y.npy', allow_pickle=True)
test_X = np.load(f'{PATH}/test_X.npy', allow_pickle=True)
test_y = np.load(f'{PATH}/test_y.npy', allow_pickle=True)

model = Sequential()

model.add(LSTM(NODES, input_shape=(train_X.shape[1:]), return_sequences=True))
model.add(Dropout(DROPOUT))
model.add(BatchNormalization())

model.add(LSTM(NODES, input_shape=(train_X.shape[1:]), return_sequences=True))
model.add(Dropout(DROPOUT))
model.add(BatchNormalization())

model.add(LSTM(NODES, input_shape=(train_X.shape[1:])))
model.add(Dropout(DROPOUT))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(DROPOUT))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

model.fit(train_X,
          train_y,
          epochs=EPOCHS,
          validation_data=(test_X, test_y),
          callbacks=[tensorboard])

%tensorboard --logdir logs
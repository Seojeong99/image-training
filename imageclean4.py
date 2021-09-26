#import keras.backend.tensorflow_backend as K
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 정규화(dataset 전처리)
X_train = X_train.astype(float) / 255.0
X_test = X_test.astype(float) / 255.0

## 원-핫 인코딩
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print("X_train data\n ", X_train)
print("y_train data\n ", y_train)
model = keras.Sequential([
    #모델을 구성할 때 계층 구조를 작성한 순서 그대로
    # 순차적으로 쌓아 생성할 수 있게 해주는 함수
    Conv2D(32, kernel_size=(3, 3), padding='same',
           input_shape=X_train.shape[1:], activation=tf.nn.relu),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),#노드 부각시키기
    #conv2d
    Conv2D(64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),

    Flatten(),
    Dense(64, activation=tf.nn.relu),
    Dropout(0.25),
    Dense(num_classes, activation=tf.nn.softmax)

])


model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
# 컴파일 실행
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# 과대적합 막기위해 실행
# 구성 해논 모델의 계층 구조를 간략적으로 보여주는 함수
model.summary()

history = model.fit(X_train, y_train, batch_size=64, epochs=1, validation_data=(X_test, y_test),
                        callbacks=[early_stopping], shuffle=True)

print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))

pred = model.predict(X_test)
x = np.argmax(pred[0])
print(x)
print("완료!")

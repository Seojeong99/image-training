import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import datasets
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

cifar_mnist = datasets.cifar10

(X_train, y_train), (X_test, y_test) = cifar_mnist.load_data()

# 정규화(dataset 전처리)
X_train = X_train.astype(float) / 255.0
X_test = X_test.astype(float) / 255.0

## 원-핫 인코딩
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = keras.Sequential([
    Conv2D(32, kernel_size=(3, 3), padding='same',
           input_shape=X_train.shape[1:], activation=tf.nn.relu),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),

    Conv2D(64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),

    Flatten(),
    Dense(64, activation=tf.nn.relu),
    Dropout(0.25),
    Dense(num_classes, activation=tf.nn.softmax)

])
model.summary()

#predict는 input data가 numpy array나 list of array 형태만 가능하다.
prediction = model.predict(X_test)

np.set_printoptions(formatter={'float': lambda x: "{0:2.1f}".format(x)})
cnt = 0

#파일들이 있으면 해당 파일과 비교.
for i in prediction:
    pre_ans = i.argmax()  # 예측 레이블
    print("i는")
    print(i)
    print(pre_ans)
    pre_ans_str = ''
    if pre_ans == 0: pre_ans_str = "airplane"
    elif pre_ans == 1: pre_ans_str = "automobile"
    elif pre_ans == 2: pre_ans_str = "bird"
    elif pre_ans == 3: pre_ans_str = "cat"
    elif pre_ans == 4: pre_ans_str = "deer"
    elif pre_ans == 5: pre_ans_str = "dog"
    elif pre_ans == 6: pre_ans_str = "frog"
    elif pre_ans == 7: pre_ans_str = "horse"
    elif pre_ans == 8: pre_ans_str = "ship"
    else: pre_ans_str = "truck"
    if i[0] >= 0.1:
        print("해당 이미지는 "+pre_ans_str+"으로 추정됩니다.")
    if i[1] >= 0.1:
        print("해당 이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[2] >= 0.1:
        print("해당 이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[3] >= 0.1:
        print("해당 이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[4] >= 0.1:
        print("해당 이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[5] >= 0.1:
        print("해당 이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[6] >= 0.1:
        print("해당 이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[7] >= 0.1:
        print("해당 이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[8] >= 0.1:
        print("해당 이미지는 "+pre_ans_str+"으로 추정됩니다.")
    if i[9] >= 0.1:
        print("해당 이미지는 "+pre_ans_str+"으로 추정됩니다.")
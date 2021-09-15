import keras.backend.tensorflow_backend as K

from keras.datasets import cifar10
from keras.utils import np_utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
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

with K.tf_ops.device('/device:CPU:0'):
    # Sequential은 모델의 계층을 순차적으로 쌓아 생성하는 방법을 말한다.
    # Conv2D(컨볼루션 필터의 수, 컨볼루션 커널(행,열) 사이즈, padding(valid(input image > output image), same(입력 = 출력),
    #        샘플 수를 제외한 입력 형태(행, 열 채널 수)), 입력 이미지 사이즈, 활성화 함수)
    # MaxPooling은 풀링 사이즈에 맞춰 가장 큰 값을 추출함 (2,2)일 경우 입력 영상 크기에서 반으로 줄어듬.
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 전결합층(Fully-Conneected layer)에 전달하기 위해서 1차원 자료로 바꾸어 주는 함수
    # Dense(출력 뉴런 수, 입력 뉴런 수, 활성화 함수(linear, relu, sigmoid, softmax)) 로 구성된다.
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    # model.compile(loss=카테고리가 3개 이상('categorical_crossentropy'), adam : 경사 하강법, accuracy : 평가 척도)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_dir = './model'

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = model_dir + '/cifar10_classification_test.h5'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=8)

    # 구성 해논 모델의 계층 구조를 간략적으로 보여주는 함수
    model.summary()

    history = model.fit(X_train, y_train, batch_size=64, epochs=80, validation_data=(X_test, y_test),
                        callbacks=[checkpoint, early_stopping], shuffle=True)

print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))
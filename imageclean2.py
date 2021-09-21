import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

cifar_mnist = datasets.cifar10
#데이터 가져오기

(train_images, train_labels),(test_images, test_labels) = cifar_mnist.load_data()
#데이터 로드하기

#class_names = [
#    'Airplane',
#    'Car',
#    'Birds',
#    'Cat',
#    'Deer',
#    'Dog',
#    'Frog',
#    'Horse',
#    'Ship',
#    'Truck'
#]
#클래스 이름 지정

#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)#데이터 시각화
#    plt.xticks([])
#    plt.yticks([])
#   plt.grid(False)
 #   plt.imshow(train_images[i], cmap=plt.cm.binary)
  #  plt.xlabel(train_labels[i])
#plt.show()
#보여주기 위한 친구들
batch_size = 64
num_classes = 10
epochs = 2

train_images = train_images.astype('float32')#데이터 타입 바꿔서 넣기
train_images = train_images / 255

test_images = test_images.astype('float32')
test_images = test_images / 255

train_labels = utils.to_categorical(train_labels, num_classes)
#train label번째에 데이터 넣기. 데이터의 개수는 num_classes
#배열의 크기는 num_classes

test_labels = utils.to_categorical(test_labels, num_classes)
#데이터 선처리

model = keras.Sequential([
    #모델을 구성할 때 계층 구조를 작성한 순서 그대로
    # 순차적으로 쌓아 생성할 수 있게 해주는 함수
    Conv2D(32, kernel_size=(3, 3), padding='same',
           input_shape=train_images.shape[1:], activation=tf.nn.relu),
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

model.summary()
#모델 간략적으로 보여주기
#CNN모델 구성하기

model.compile(
    loss = 'categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
#컴파일 실행
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
#과대적합 막기위해 실행
history = model.fit(
    train_images, train_labels,
    epochs=epochs,
    validation_data=(test_images, test_labels),
    shuffle=True,
    callbacks=[early_stopping]
)
#모델 훈련시키기
loss, acc = model.evaluate(test_images, test_labels)
print("\nLoss: {}, Acc : {}".format(loss,acc))
#훈련 잘 됐는지 테스트하기

#-------------------------------------------------------
predictions = model.predict(test_images)
#훈련된 애 가지고 예측하기

np.set_printoptions(formatter={'float': lambda x: "{0:2.1f}".format(x)})
cnt = 0

#파일들이 있으면 해당 파일과 비교.
for i in predictions:
    pre_ans = i.argmax()  # 예측 레이블
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
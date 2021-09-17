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

class_names = [
    'Airplane',
    'Car',
    'Birds',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]
#클래스 이름 지정

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)#데이터 시각화
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()
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

def plot_image(i, predictions_array, true_label, img):
    predictions_array = predictions_array[i],
    true_label = true_label[i]
    img = img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == np.argmax(true_label):
        color = 'blue'
    else:
        color = 'red'

    #plt.xlabel(100*np.max(predictions_array))

    #plt.xlabel(class_names[predicted_label])
    #plt.xlabel(100*np.max(predictions_array),class_names[np.argmax(true_label)])
    #plt.xlabel(class_names[np.argmax(true_label)])

def plot_value_array(i, predictions_array, true_label):
        predictions_array = predictions_array[i]
        true_label = true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0,1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[np.argmax(true_label)].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()
#import keras.backend.tensorflow_backend as K
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os, re, glob
import cv2

# 경로는 자신이 테스트해볼 파일의 경로로 바꿔주시면 됩니다!
from tensorflow.lite.python.schema_py_generated import np

groups_folder_path = r'C:/imagedata/sample/'
categories = ["h", "j", "k"]
num_classes = len(categories)

image_w = 28
image_h = 28

X = []
Y = []

for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_folder_path + categorie + '/'

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            print(image_dir + filename)
            img = cv2.imread(image_dir + filename)
            img = cv2.resize(img, None, fx=image_w / img.shape[1], fy=image_h / img.shape[0])
            X.append(img / 256)
            Y.append(label)

X = np.array(X)
Y = np.array(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
xy = (X_train, X_test, Y_train, Y_test)

np.save("./img_data.npy", xy)

X_train, X_test, Y_train, Y_test = np.load('./img_data.npy', allow_pickle=True)

#num_classes = y_test.shape[1]

print("X_train data\n ", X_train)
print("y_train data\n ", Y_train)

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


history = model.fit(
    X_train, Y_train,
    batch_size=15, epochs=300,
    validation_data=(X_test, Y_test),
    #callbacks=[early_stopping],
    shuffle=True)

model.save('Gersang.h5')

categories = ["h", "j", "k"]

def Dataization(img_path):
    image_w = 28
    image_h = 28
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=image_w / img.shape[1], fy=image_h / img.shape[0])
    return (img / 256)


src = []
name = []
test = []
image_dir = "C:\\imagedata\\imagetest\\"
for file in os.listdir(image_dir):
    if file.find('.jpg') != -1:
        src.append(image_dir + file)
        name.append(file)
        test.append(Dataization(image_dir + file))

test = np.array(test)
model = load_model('Gersang.h5')
#time.sleep(10)
loss, acc = model.evaluate(X_test, Y_test)
print("\nLoss: {}, Acc : {}".format(loss,acc))
predict = model.predict(test)

for i in range(len(test)):
    print(name[i])
    x = np.argmax(predict[i])
    if(x==0):
        print("한소희입니다!")
    elif(x==1):
        print("전지현입니다!")
    elif (x == 2):
        print("김태희입니다!")
    print(predict[i])

#-----------------표로 나타내기-----------------------
'''
def plot_image(i, predict_array, true_label, img):
    predict_array = predict_array[i],
    true_label = true_label[i]
    img = img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predict_array)
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
    plot_image(i, predict, X_test, Y_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predict, X_test)
plt.show()
'''
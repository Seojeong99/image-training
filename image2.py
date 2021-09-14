from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import tensorflow as tf # 1.15.0 버전 사용
import os
import pandas as pd
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img

path = r"C:\imagedata\\"
trainFileList = os.listdir(path+"train")
testFileList = os.listdir(path+"test")

trainFileList = [x for x in trainFileList if 'jpg' in x]
testFileList = [x for x in testFileList if 'jpg' in x]

word = {'kim': 0, 'han': 1, 'jon': 2}

# np.array에 각 데이터를 넣어주는 작업
for i, file in enumerate(trainFileList):
    label, nm, tr, _ = file.split('_') # 예시 aa_bc_01_02.jpg
    code, ext = _.split('.') # 예시 02.jpg

    # 이미지 불러오기
    new_img = load_img(path+'train\\'+file)
    # 이미지를 array 형식으로 변형
    arr_img = img_to_array(new_img)
    # (1, 28, 28, 3) 검정색 글씨뿐이지만 나름 RGB...zz
    img = arr_img.reshape((1,)+arr_img.shape)

    # 첫 번째 데이터의 경우 container를 생성
    if i == 0:
        print("1")
        container = img
        labels = word[label]
    # 첫 번쨰가 아닌경우 array에 계속 쌓아나감
    else:
        print("2")
        container = np.vstack((container,img))
        labels = np.vstack((labels, word[label]))

xTrain = container
# 카테고리 데이터 one-hot encoding
yTrain = np_utils.to_categorical(labels,5)

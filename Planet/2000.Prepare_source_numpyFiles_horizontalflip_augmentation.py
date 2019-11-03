# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:02:03 2017

@author: SriPrav
"""
import numpy as np
np.random.seed(2017)

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras.layers.normalization import BatchNormalization
from keras import __version__ as keras_version
import matplotlib.pyplot as plt
import seaborn as sns
import random

inDir = 'C:/Users/SriPrav/Documents/R/27Planet'

from tqdm import tqdm

x_train = []
y_train = []
x_test = []

train_file = inDir + "/input/train_images.csv"
test_file = inDir + "/input/test_images.csv"
test_additional_file = inDir + "/input/test_additional_images.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
test_additional_df = pd.read_csv(test_additional_file)
print(train_df.shape) # (40479, 4)
print(test_df.shape)  # (40669, 2)
print(test_additional_df.shape)  # (20522, 2)

test_all = pd.concat([test_df,test_additional_df])
print(test_all.shape)  # (20522, 2)


ROWS     = 224
COLUMNS  = 224
CHANNELS = 3
VERBOSEFLAG = 1



y_train = []

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in train_df['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for tag in train_df.tags.values:
    targets = np.zeros(17)
    for t in tag.split(' '):
        targets[label_map[t]] = 1
    y_train.append(targets)
    
y_train = np.array(y_train, np.uint8)
                 
def centering_image(img):
    size = [256,256]
    
    img_size = img.shape[:2]
    
    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    return resized
    
#def random_rotate(image):
#    cols = image.shape[1]
#    rows = image.shape[0]
#    mean_color = np.mean(image, axis=(0, 1))
#
#    angle = random.uniform(0, 90)
#    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
#    if random.randint(0, 1) == 1:
#        dst = cv2.warpAffine(image, M, (cols, rows), borderValue=mean_color, borderMode=cv2.BORDER_REFLECT)
#    else:
#        dst = cv2.warpAffine(image, M, (cols, rows), borderValue=mean_color)
#    return dst

  
    
def get_im_cv2(path):
    img = cv2.imread(path)#path = "C:/Users/SriPrav/Documents/R/27Planet/input/train-jpg/train_24477.jpg"
#    plt.imshow(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #filp 
    img =cv2.flip(img,0)
    
#    plt.imshow(img)
    #resize
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))
    
    #centering
    img = centering_image(cv2.resize(img, dsize=tile_size))
    
    #out put 224*224px     
    img = cv2.resize(img, (ROWS, COLUMNS))#, cv2.INTER_LINER
    
    return img
          
def load_train_fromfile():
    X_train = []
    X_train_id = []

    start_time = time.time()

    for fl in train_df.image_path.values:
        print(fl)
        flbase = os.path.basename(fl) # "C:\Users\SriPrav\Documents\R\\22Intel\\input\\train\\Type_1\\a_Type_1\\10.jpg"
        img = get_im_cv2(fl)
        X_train.append(img)
        X_train_id.append(flbase)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, X_train_id
    
def load_test_fromfile():
    X_test = []
    X_test_id = []

    start_time = time.time()

    for fl in test_all.image_path.values:
        print(fl)
        flbase = os.path.basename(fl) # "C:\Users\SriPrav\Documents\R\\22Intel\\input\\train\\Type_1\\a_Type_1\\10.jpg"
        img = get_im_cv2(fl)
        
        X_test.append(img)
        X_test_id.append(flbase)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, X_test_id
    

def read_and_normalize_train_data():
    train_data,  train_id = load_train_fromfile()
    train_target = y_train
    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
#    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
#    train_data = train_data.astype('float32')
#    train_data = train_data / 255
#    train_target = np_utils.to_categorical(train_target, 3)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test_fromfile()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

#    test_data = test_data.astype('float32')
#    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id

#train_data, train_target, train_id = read_and_normalize_train_data()
test_data, test_id = read_and_normalize_test_data()

######################################################################################################################
#np.save(inDir +"/input/train_data_hflip_aug_224_3.npy",train_data)
#np.save(inDir +"/input/train_target_hflip_aug_224_3.npy",train_target)
#np.save(inDir +"/input/train_id_hflip_aug_224_3.npy",train_id)

np.save(inDir +"/input/test_data_hflip_aug_224_3.npy",test_data)
np.save(inDir +"/input/test_id_hflip_aug_224_3.npy",test_id)
######################################################################################################################

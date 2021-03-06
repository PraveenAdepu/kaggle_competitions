# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:02:03 2017

@author: SriPrav
"""
import numpy as np
np.random.seed(2016)

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

inDir = 'C:/Users/SriPrav/Documents/R/22Intel'

train_file = inDir + "/input/rectangles_train.csv"
test_file = inDir + "/input/rectangles_test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train_df.shape) # (8211, 11)
print(test_df.shape)  # (512, 12)

train_df.head(2)

ROWS     = 299
COLUMNS  = 299
CHANNELS = 3
VERBOSEFLAG = 1

def get_im_cv2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    resized = cv2.resize(img, (ROWS, COLUMNS), cv2.INTER_LINEAR)
    if (img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1] * 256 / img.shape[0]), 256)
    else:
        tile_size = (256, int(img.shape[0] * 256 / img.shape[1]))

    img = cv2.resize(img, dsize=tile_size)
    return img
    
y_train = []
y_train = train_df.clss.values
       
def load_train_fromfile():
    X_train = []
    X_train_id = []

    start_time = time.time()

    for fl in train_df.image_path.values:
        print(fl)
        flbase = os.path.basename(fl) # "C:\Users\SriPrav\Documents\R\\22Intel\\input\\train\\Type_1\\a_Type_1\\10.jpg"
        img = get_im_cv2(fl)
        train_row   = train_df[train_df['image_path'] == fl]
        img_shape0 = train_row['img_shp_0'].astype(int)
        img_shape1 = train_row['img_shp_1'].astype(int)
        y = train_row['sh0_start'].astype(int)
        x = train_row['sh1_start'].astype(int)
        yh = train_row['sh0_end'].astype(int)
        xw = train_row['sh1_end'].astype(int)
        img = cv2.resize(img, (int(img_shape0), int(img_shape1)), cv2.INTER_LINEAR)
        img = img[int(y): int(yh) , int(x): int(xw)]
        img = cv2.resize(img, (ROWS, COLUMNS), cv2.INTER_LINEAR)
        X_train.append(img)
        X_train_id.append(flbase)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, X_train_id
    



def load_test_fromfile():
    X_test = []
    X_test_id = []

    start_time = time.time()

    for fl in test_df.image_path.values:
        print(fl)
        flbase = os.path.basename(fl) # "C:\Users\SriPrav\Documents\R\\22Intel\\input\\train\\Type_1\\a_Type_1\\10.jpg"
        img = get_im_cv2(fl)
        test_row   = test_df[test_df['image_path'] == fl]
        img_shape0 = test_row['img_shp_0'].astype(int)
        img_shape1 = test_row['img_shp_1'].astype(int)
        y = test_row['sh0_start'].astype(int)
        x = test_row['sh1_start'].astype(int)
        yh = test_row['sh0_end'].astype(int)
        xw = test_row['sh1_end'].astype(int)
        img = cv2.resize(img, (int(img_shape0), int(img_shape1)), cv2.INTER_LINEAR)
        img = img[int(y): int(yh) , int(x): int(xw)]  
        img = cv2.resize(img, (ROWS, COLUMNS), cv2.INTER_LINEAR)
        X_test.append(img)
        X_test_id.append(flbase)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, X_test_id
    

def read_and_normalize_train_data():
    train_data,  train_id = load_train_fromfile()
    train_target = y_train
    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
#    train_data = train_data.astype('float32')
#    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 3)

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

train_data, train_target, train_id = read_and_normalize_train_data()
test_data, test_id = read_and_normalize_test_data()

######################################################################################################################
np.save(inDir +"/input/train_data_299_3.npy",train_data)
np.save(inDir +"/input/train_target_299_3.npy",train_target)
np.save(inDir +"/input/train_id_299_3.npy",train_id)

np.save(inDir +"/input/test_data_299_3.npy",test_data)
np.save(inDir +"/input/test_id_299_3.npy",test_id)
######################################################################################################################

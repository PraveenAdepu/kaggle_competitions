# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:02:03 2017

@author: SriPrav
"""
import numpy as np
np.random.seed(201803)

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

inDir = 'C:/Users/SriPrav/Documents/R/51TGS'

from tqdm import tqdm
import PIL.Image
from skimage.io import imsave, imread
from skimage.transform import resize
import random

x_train = []
y_train = []
x_test = []

train_file = inDir + "/input/Prav_10folds_CVindices.csv"
test_file = inDir + "/input/test_images.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print(train_df.shape) # (5088, 4)
print(test_df.shape)  # (100064, 1)

####################################################################################################################
ROWS        = 128
COLUMNS     = 128
CHANNELS    = 1
VERBOSEFLAG = 1
                

train_data = np.zeros((len(train_df), ROWS, COLUMNS, CHANNELS), dtype=np.uint8)
train_target = np.zeros((len(train_df), ROWS, COLUMNS, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')

for n, id_ in tqdm(enumerate(train_df.img.values), total=len(train_df)):
    print(id_)
    mask_id_ = id_.replace('images','masks')
    print(mask_id_)
    
    img = imread(id_)[:,:,:CHANNELS]
    img = resize(img, (ROWS, COLUMNS), mode='constant', preserve_range=True)
    train_data[n] = img
    
    mask_ = imread(mask_id_)
    mask_ = np.expand_dims(resize(mask_, (ROWS, COLUMNS), mode='constant', 
                                      preserve_range=True), axis=-1)
#    mask = np.maximum(img, mask_)
    train_target[n] = mask_

# Check if training data looks all right
ix = random.randint(0, len(train_df))
plt.imshow(np.dstack((train_data[ix],train_data[ix],train_data[ix])))
plt.show()
tmp = np.squeeze(train_target[ix]).astype(np.float32)
plt.imshow(np.dstack((tmp,tmp,tmp)))
plt.show()

test_data = np.zeros((len(test_df), ROWS, COLUMNS, CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')

for n, id_ in tqdm(enumerate(test_df.img.values), total=len(test_df)):
    img = imread(id_)[:,:,:CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (ROWS, COLUMNS), mode='constant', preserve_range=True)
    test_data[n] = img

print('Done!')

def load_train_fromfile():
   
    X_train_id = []   
    start_time = time.time()
    for fl in train_df.img.values:
        print(fl)
        flbase = os.path.basename(fl) # fl = "C:/Users/SriPrav/Documents/R/29Carvana/input/train_masks/0cdf5b5d0ce1_01_mask.gif"

        X_train_id.append(flbase)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))  
    return X_train_id

def load_test_fromfile():
   
    X_test_id = []

    start_time = time.time()

    for fl in test_df.img.values:
        print(fl)
        flbase = os.path.basename(fl)

        X_test_id.append(flbase)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test_id
train_id = load_train_fromfile()
test_id = load_test_fromfile()



######################################################################################################################
np.save(inDir +"/input/train_data_128_1.npy",train_data)
np.save(inDir +"/input/train_target_128_1.npy",train_target)
np.save(inDir +"/input/train_id_128_1.npy",train_id)

np.save(inDir +"/input/test_data_128_1.npy",test_data)
np.save(inDir +"/input/test_id_128_1.npy",test_id)
np.save(inDir +"/input/test_sizes_128_1.npy",sizes_test)
######################################################################################################################



ROWS        = 101
COLUMNS     = 101
CHANNELS    = 1
VERBOSEFLAG = 1
    
   
def get_im_cv2(path):
    img = cv2.imread(path)#path = "C:/Users/SriPrav/Documents/R/51TGS/input/train/images/000e218f21.png"
#    plt.imshow(img)
#    img = imread(path)[:,:,:CHANNELS]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#    plt.imshow(img)
    #resize
#    if(img.shape[0] > img.shape[1]):
#        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
#    else:
#        tile_size = (256, int(img.shape[0]*256/img.shape[1]))
#    #centering
#    img = centering_image(cv2.resize(img, dsize=tile_size))
    
    #out put 224*224px     
#    img = resize(img, (ROWS, COLUMNS), mode='constant', preserve_range=True)
 
    img = img[...,np.newaxis]
    return img

#def get_im_cv2_gif(path):
#    img = PIL.Image.open(path)#path = "C:/Users/SriPrav/Documents/R/29Carvana/input/train_masks/0cdf5b5d0ce1_01_mask.gif"
#    img = img.convert('L')
#    img = np.array(img)
#       
##    plt.imshow(img)
##    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
##    plt.imshow(img)
#    #resize
##    if(img.shape[0] > img.shape[1]):
##        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
##    else:
##        tile_size = (256, int(img.shape[0]*256/img.shape[1]))
#    #centering
#    
##    img = centering_image(cv2.resize(img, dsize=tile_size))
#    
#    #out put 224*224px     
#    img = resize(img, (ROWS, COLUMNS), preserve_range=True)
#    img = img[...,np.newaxis]
#    return img
          
def load_train_fromfile():
    X_train    = []
    X_train_id = []
    Y_train    = []
    start_time = time.time()

    for fl in train_df.img.values:
        print(fl)
        flbase = os.path.basename(fl) # fl = "C:/Users/SriPrav/Documents/R/29Carvana/input/train_masks/0cdf5b5d0ce1_01_mask.gif"
        img = get_im_cv2(fl)
        X_train.append(img)
        X_train_id.append(flbase)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    
    for fl in train_df['mask'].values:
        print(fl)
         # fl = "C:/Users/SriPrav/Documents/R/29Carvana/input/train_masks/0cdf5b5d0ce1_01_mask.gif"
        img = get_im_cv2(fl)
        Y_train.append(img)       
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    
    return X_train, X_train_id, Y_train

    
def load_test_fromfile():
    X_test = []
    X_test_id = []

    start_time = time.time()

    for fl in test_df.img.values:
        print(fl)
        flbase = os.path.basename(fl) # "C:\Users\SriPrav\Documents\R\\22Intel\\input\\train\\Type_1\\a_Type_1\\10.jpg"
        img = get_im_cv2(fl)
        
        X_test.append(img)
        X_test_id.append(flbase)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, X_test_id
    

def read_and_normalize_train_data():
    train_data,  train_id, y_train = load_train_fromfile()
    train_target = y_train
    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

#    print('Reshape...')
#    train_data   = train_data.transpose((0, 1, 2, 3))
#    train_target = train_target.transpose((0, 1, 2, 3))

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
#    test_data = test_data.transpose((0, 1, 2,3))

#    test_data = test_data.astype('float32')
#    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id

train_data, train_target, train_id = read_and_normalize_train_data()
test_data, test_id = read_and_normalize_test_data()

######################################################################################################################
np.save(inDir +"/input/train_data_101_1.npy",train_data)
np.save(inDir +"/input/train_target_101_1.npy",train_target)
np.save(inDir +"/input/train_id_101_1.npy",train_id)

np.save(inDir +"/input/test_data_101_1.npy",test_data)
np.save(inDir +"/input/test_id_101_1.npy",test_id)
######################################################################################################################

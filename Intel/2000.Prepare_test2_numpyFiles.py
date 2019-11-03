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


test_file = inDir + "/input/rectangles_test2.csv"

test_df = pd.read_csv(test_file)

print(test_df.shape)  # (3506, 12)

test_df.head(2)

ROWS     = 224
COLUMNS  = 224
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
    
# C:\Users\SriPrav\Documents\R\22Intel\input\stage2\test\11393.jpg
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


test_data, test_id = read_and_normalize_test_data()

######################################################################################################################


np.save(inDir +"/input/test2_data_224_3.npy",test_data)
np.save(inDir +"/input/test2_id_224_3.npy",test_id)
######################################################################################################################


#Read train data time: 799.66 seconds
#('Test shape:', (3506L, 3L, 224L, 224L))
#(3506L, 'test samples')
#Read and process test data time: 800.07 seconds

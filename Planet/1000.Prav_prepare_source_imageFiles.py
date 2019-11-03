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

inDir = 'C:/Users/SriPrav/Documents/R/27Planet'

train_file = inDir + "/input/train_v2.csv"
#test_file = inDir + "/input/rectangles_test.csv"
train_df = pd.read_csv(train_file)
#test_df = pd.read_csv(test_file)
print(train_df.shape) # (8211, 11)
#print(test_df.shape)  # (512, 12)

train_df['image_path'] = train_df['image_name'].map(lambda x: inDir + '/input/train-jpg/' + x + '.jpg')

cv_file = inDir + "/CVSchema/Prav_CVindices_10folds.csv"
CV_Schema = pd.read_csv(cv_file)

train_df = pd.merge(train_df, CV_Schema, how = 'left', on = ['image_name'])

train_df.head(2)

sub_file = inDir +'/input/train_images.csv'
train_df.to_csv(sub_file, index=False)

def write_test_csv():

    out = open('test_images.csv', "w")
    out.write("image_name,image_path\n")
    # for f in sorted(train_files + test_files + additional_files):  
        
    path = os.path.join('C:\Users\SriPrav\Documents\R\\27Planet', 'input','test', '*.jpg')
    files = glob.glob(path)#[:10]
    for f in files:            
        image_name = os.path.basename(f)       

        out.write(image_name)
        out.write(',' + f)        
        out.write('\n')
    
    out.close()

def write_test_additional_csv():

    out = open('test_additional_images.csv', "w")
    out.write("image_name,image_path\n")
    # for f in sorted(train_files + test_files + additional_files):  
        
    path = os.path.join('C:\Users\SriPrav\Documents\R\\27Planet', 'input','test-additional', '*.jpg')
    files = glob.glob(path)#[:10]
    for f in files:            
        image_name = os.path.basename(f)       

        out.write(image_name)
        out.write(',' + f)        
        out.write('\n')
    
    out.close()
    
if __name__ == '__main__':
    write_test_csv()
    write_test_additional_csv()



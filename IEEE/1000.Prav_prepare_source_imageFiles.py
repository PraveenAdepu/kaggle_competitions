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

inDir = 'C:/Users/SriPrav/Documents/R/43IEEE'


inDir = 'C:\\Users\\SriPrav\\Documents\\R\\43IEEE\\input\\train'


def write_train_csv():
    out = open('images_train_v5.csv', "w")
    out.write("image_path,image_name,image_category\n")
    for path, subdirs, files in os.walk(inDir):
        for name in files:
            image_path =  os.path.join(path, name)           
            image_name = name
            image_category = path.replace(inDir,'').replace('\\','')
            
            out.write(str(image_path))
            out.write(',' + str(image_name))
            out.write(',' + str(image_category))            
            out.write('\n')
    
    out.close()

if __name__ == '__main__':
    write_train_csv()


#3.095,080
inDir = 'C:\\Users\\SriPrav\\Documents\\R\\43IEEE\\input'

test_images = 'C:\\Users\\SriPrav\\Documents\\R\\43IEEE\\input\\test'

def write_test_csv():
    out = open('images_test.csv', "w")
    out.write("image_path,_id,image_id\n")
    for files in os.listdir(test_images):          
        image_path =  os.path.join(test_images,files) 
        image_name = files

        out.write(str(image_path))
        out.write(',' + str(image_name))
                    
        out.write('\n')
    
    out.close()

if __name__ == '__main__':
    write_test_csv()


inDir = 'C:\\Users\\SriPrav\\Documents\\R\\43IEEE\\input\\train_5'


def write_train_csv():
    out = open('images_train_v5.csv', "w")
    out.write("image_path,image_name,image_category\n")
    for path, subdirs, files in os.walk(inDir):
        for name in files:
            image_path =  os.path.join(path, name)           
            image_name = name
            image_category = path.replace(inDir,'').replace('\\','')
            
            out.write(str(image_path))
            out.write(',' + str(image_name))
            out.write(',' + str(image_category))            
            out.write('\n')
    
    out.close()

if __name__ == '__main__':
    write_train_csv()
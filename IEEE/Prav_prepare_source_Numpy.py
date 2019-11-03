# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:49:48 2018

@author: SriPrav
"""



import os

import numpy as np
random_state = 2017
np.random.seed(random_state)

import pandas as pd
import io
import bson
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import concurrent.futures


import time
import glob
import math

from tqdm import tqdm
from PIL import Image
from io import BytesIO
import jpeg4py as jpeg
import random

import warnings
warnings.filterwarnings("ignore")


from joblib import Parallel, delayed
from save_image_numpy import save_image_numpy_array
import time
from keras.utils import np_utils

inDir = 'C:/Users/SriPrav/Documents/R/43IEEE'

#train_data_224_3   = np.load(inDir +"/input/y_build_224_31.npy")
#def normalize_image(x):
#    x = np.array(x, dtype=np.uint8)
#    #x=x.transpose((0,1,2,3))
#    x= x.astype('float32')
#    # Subtract ImageNet mean pixel 
##    x[:, :, :, 0] -= 103.939
##    x[:, :, :, 1] -= 116.779
##    x[:, :, :, 2] -= 123.68
##    x = x / 255
##    x -= 0.5
##    x *= 2.
#    return x
#train_data_224_3 = normalize_image(train_data_224_3)
#import matplotlib.pyplot as plt
#plt.imshow(train_data_224_3[22])


images_val = pd.read_csv(inDir + '/input/images_val_balanced.csv')
images_val.groupby(['CVindices','image_category'])[['image_category']].size()
images_val.head()

images_val['image_category_lower'] = images_val['image_category'].str.lower()
images_val = images_val.sort_values('image_category_lower')

del images_val['image_category_lower']
y, rev_labels = pd.factorize(images_val['image_category'])
images_val['y'] = y

images_train = pd.read_csv(inDir + '/input/images_train_balanced.csv')
images_train.groupby(['CVindices','image_category'])[['image_category']].size()
images_train.head()

images_train['image_category_lower'] = images_train['image_category'].str.lower()
images_train = images_train.sort_values('image_category_lower')

del images_train['image_category_lower']
y, rev_labels = pd.factorize(images_train['image_category'])
images_train['y'] = y


num_classes = 10  # This will reduce the max accuracy to about 0.75
CHANNELS = 3
ROWS = 224
COLUMNS = 224
nb_epoch = 2
VERBOSEFLAG = 1
batch_size  = 16
patience = 10
optim_type = 'Adam'
learning_rate = 1e-3
crop_size = 224

def convert_to_numpy(x):
    x = np.array(x, dtype=np.uint8)
    return x

if __name__=='__main__':
    for ep in range(1,nb_epoch+1):
        print("EPOCH processing {}".format(ep))
        
        if ep !=1:
            print("build and valid deleted")
            del X_build, y_build, X_valid, y_valid
        
        t0 = time.time()
        print("X_build Fetching Data ")
        X_build = Parallel(n_jobs=10)(delayed(save_image_numpy_array)(file) for file in  images_train.image_path.values)
        print("X_valid Fetching Data ")
        X_valid = Parallel(n_jobs=10)(delayed(save_image_numpy_array)(file) for file in  images_val.image_path.values)
        
        print("X_build and X_valid convert to Numpy ")
        X_build = convert_to_numpy(X_build)
        X_valid = convert_to_numpy(X_valid)
        
        print("X_build and X_valid Reshaping ")
        X_build = X_build.reshape(len(X_build),224,224,3)
        X_valid = X_valid.reshape(len(X_valid),224,224,3)
        
       
        print("y_build and y_valid preparing ")
        y_build = np_utils.to_categorical(images_train['y'],10)
        y_valid = np_utils.to_categorical(images_val['y'],10)
        
        np.save(inDir +"/input/X_build_224_3"+str(ep)+".npy",X_build)
        np.save(inDir +"/input/X_valid_224_3"+str(ep)+".npy",X_valid)
        
        np.save(inDir +"/input/y_build_224_3"+str(ep)+".npy",y_build)
        np.save(inDir +"/input/y_valid_224_3"+str(ep)+".npy",y_valid)
        
        t1 = time.time()
        times = t1 - t0
        print("times {}".format(times))
    


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
from multiprocessing import cpu_count
from sklearn.cross_validation import train_test_split
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.models import Model
from ipywidgets import IntProgress

from keras.utils import np_utils

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
from sklearn.utils import shuffle

from joblib import Parallel, delayed
from save_image_numpy import save_image_numpy_array
import time


from keras.applications.resnet50 import ResNet50


inDir = 'C:/Users/SriPrav/Documents/R/43IEEE'

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

test = pd.read_csv(inDir+'/input/images_test_v2.csv')
test.head()

def get_im_cv2_test(path):
    img = cv2.imread(path)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def normalize_image(x):
    x = np.array(x, dtype=np.uint8)
    #x=x.transpose((0,1,2,3))
    x= x.astype('float32')
    # Subtract ImageNet mean pixel 
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68
#    x = x / 255
#    x -= 0.5
#    x *= 2.
    return x

def load_test_frombatch(images_batch):
    X_test = []
    for fl in images_batch.image_path.values:        
        img = get_im_cv2_test(fl)        
        X_test.append(img)
    return X_test

def batch_generator_X_build(images_build,X_build, y_build ,batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    
    number_of_batches = np.ceil(len(images_build)/batch_size)
    counter = 0
    sample_index = images_build.index.get_values()
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_build[list(batch_index)]
        y_batch = y_build[list(batch_index)]       
        
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0 
            
def batch_generator_X_valid(images_valid,X_valid, y_valid ,batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(len(images_valid)/batch_size)
    counter = 0
    sample_index = images_valid.index.get_values()
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_valid[list(batch_index)]
        y_batch = y_valid[list(batch_index)]       
        
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

            

def model_ResNet50(num_classes):
    base_model = ResNet50(weights='imagenet')
    x = base_model.output
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=x)    
    return model

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
 
ModelName= 'Resnet_101'
i=1
MODEL_WEIGHTS_FILE = inDir + '/Prav_01_Resnet101_'+str(i)+'.h5'
print('Fold ', i , ' Processing')

model = model_ResNet50(num_classes=num_classes)

save_checkpoint = ModelCheckpoint(
            MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True, verbose=VERBOSEFLAG, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.25, patience=10, min_lr=1e-9, epsilon = 0.00001, verbose=1, mode='max')
learning_rate = 1e-3
if optim_type == 'SGD':
    optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
else:
    optim = Adam(lr=learning_rate)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'])



if __name__=='__main__':
    for ep in range(1,nb_epoch+1):
        print("EPOCH processing {}".format(ep))
        
        if ep !=1:
            print("build and valid deleted")
            del X_build, y_build, X_valid, y_valid
        
        print("train and val Shuffeling")
        images_train = shuffle(images_train)
        images_val = shuffle(images_val)

        t0 = time.time()
        print("X_build Fetching Data ")
        X_build = Parallel(n_jobs=10)(delayed(save_image_numpy_array)(file) for file in  images_train.image_path.values)
        print("X_valid Fetching Data ")
        X_valid = Parallel(n_jobs=10)(delayed(save_image_numpy_array)(file) for file in  images_val.image_path.values)
        
        print("X_build and X_valid Reshaping ")
        X_build = X_build.reshape(len(X_build),224,224,3)
        X_valid = X_valid.reshape(len(X_valid),224,224,3)
        
        print("X_build and X_valid Normalising ")
        X_build = normalize_image(X_build)
        X_valid = normalize_image(X_valid)
        
        print("y_build and y_valid preparing ")
        y_build = np_utils.to_categorical(images_train['y'],10)
        y_valid = np_utils.to_categorical(images_val['y'],10)
        
        print("Model training")
        model.fit_generator( generator=batch_generator_X_build(images_train,X_build, y_build, batch_size, shuffle=False),
                             #samples_per_epoch = len(build_index), 
                             steps_per_epoch = math.ceil(len(images_train) / batch_size), #int(len(build_index)/float(batch_size)),
#                             initial_epoch = initial_epoch,
                             nb_epoch = nb_epoch, 
                             callbacks = [save_checkpoint, reduce_lr],
                             validation_data=batch_generator_X_valid(images_val,X_valid,y_valid, batch_size, shuffle=False), 
                             #nb_val_samples=len(valid_index), 
                             validation_steps = math.ceil(len(images_val) / batch_size), #int(len(valid_index)/float(batch_size)),
                             max_q_size=10,
                             verbose = VERBOSEFLAG 
                  )
        
        t1 = time.time()
        times = t1 - t0
        print("times {}".format(times))
    
model.load_weights(MODEL_WEIGHTS_FILE)

del X_build, y_build, X_valid, y_valid

X_valid_data = load_test_frombatch(images_val)
X_valid_data = normalize_image(X_valid_data)

X_test = load_test_frombatch(test)
X_test = normalize_image(X_test)

pred_cv = np.zeros([images_val.shape[0],10])    
pred_test = np.zeros([X_test.shape[0],10])
       
pred_cv += model.predict(X_valid_data, batch_size=batch_size, verbose=VERBOSEFLAG)        
pred_test += model.predict(X_test, batch_size=batch_size, verbose=VERBOSEFLAG)

pred_cv = pd.DataFrame(pred_cv)
pred_cv.columns = rev_labels
pred_cv["fname"] = images_val.image_name.values
pred_cv["fname_patch"] = images_val.image_name_patch.values
pred_cv = pred_cv[["fname","fname_patch",'HTC-1-M7', 'iPhone-4s', 'iPhone-6', 'LG-Nexus-5x',
   'Motorola-Droid-Maxx', 'Motorola-Nexus-6', 'Motorola-X',
   'Samsung-Galaxy-Note3', 'Samsung-Galaxy-S4', 'Sony-NEX-7']]
sub_valfile = inDir+'/submissions/Prav.'+ModelName+'.fold' + str(i) + '.csv'    
pred_cv.to_csv(sub_valfile, index=False)

pred_test = pd.DataFrame(pred_test)
pred_test.columns = rev_labels
pred_test["fname"] = test._id.values
pred_test["fname_patch"] = test._id1.values
pred_test = pred_test[["fname","fname_patch",'HTC-1-M7', 'iPhone-4s', 'iPhone-6', 'LG-Nexus-5x',
   'Motorola-Droid-Maxx', 'Motorola-Nexus-6', 'Motorola-X',
   'Samsung-Galaxy-Note3', 'Samsung-Galaxy-S4', 'Sony-NEX-7']]
sub_file = inDir+'/submissions/Prav.'+ModelName+'.fold' + str(i) + '-test' + '.csv'
pred_test.to_csv(sub_file, index=False)

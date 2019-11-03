# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:34:36 2017

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
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.models import Model
from ipywidgets import IntProgress

from keras.utils import np_utils

import time
import glob
import math

import colour_demosaicing

import colour
from colour_demosaicing import (
    
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)

OETF = colour.RGB_COLOURSPACES['sRGB'].encoding_cctf


from keras.applications.resnet50 import ResNet50


from PIL import Image
from skimage.transform import resize
from random import shuffle
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

inDir = 'C:/Users/SriPrav/Documents/R/43IEEE'

MODEL_WEIGHTS_FILE = inDir + '/Prav_01_CustomeNet.h5'


images_train = pd.read_csv(inDir + '/input/Prav_10folds_CVindices.csv')

images_train.head()

def RowWiseOperation(x):
    x1 = x.replace('\\','\\\\')
    return x1

images_train['image_path'] = images_train['image_path'].apply(RowWiseOperation)

y, rev_labels = pd.factorize(images_train['image_category'])

images_train['y'] = y
test = pd.read_csv(inDir+'/input/images_test.csv')
del test['image_id']

test.head()
test['image_path'] = test['image_path'].apply(RowWiseOperation)

def read_and_resize(filepath):
    im_array = np.array(Image.open((filepath)), dtype="uint8")
    pil_im = Image.fromarray(im_array)
    new_array = np.array(pil_im.resize((256, 256)))
    return new_array/255

X_train = np.array([read_and_resize(filepath) for filepath in images_train['image_path']])

y= np_utils.to_categorical(images_train['y'],10)

X_test = np.array([read_and_resize(filepath) for filepath in test['image_path']])

#######################################################################################################################
#np.save(inDir +"/input/train_data_224_3.npy",train_data)
#np.save(inDir +"/input/train_target_224_3.npy",train_target)
#np.save(inDir +"/input/train_id_224_3.npy",train_id)
#
#np.save(inDir +"/input/test_data_224_3.npy",test_data)
#np.save(inDir +"/input/test_id_224_3.npy",test_id)
#######################################################################################################################

 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalMaxPool2D, Concatenate
input_shape = (256, 256, 3)
nclass = 10

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(        
                               
                                  )

valid_datagen = ImageDataGenerator(
                                   
                                   )

test_datagen = ImageDataGenerator(        
                                 
                                 )

i = 1



trainindex = images_train[images_train['CVindices'] != i].index.tolist()
valindex   = images_train[images_train['CVindices'] == i].index.tolist()

shuffle(trainindex)
shuffle(valindex)

X_build , X_valid = X_train[trainindex,:], X_train[valindex,:]
y_build , y_valid = y[trainindex,:], y[valindex,:]  

def get_model():

    nclass = 10
    inp = Input(shape=input_shape)
    norm_inp = BatchNormalization()(inp)
    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu, padding="same")(norm_inp)
    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(64, kernel_size=2, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution2D(20, kernel_size=2, activation=activations.relu, padding="same")(img_1)
    img_1 = GlobalMaxPool2D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    dense_1 = Dense(20, activation=activations.relu)(img_1)
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(lr=1e-4)
    optim = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model
	
model = get_model()
file_path="weights.best.hdf5"

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor="val_acc", mode="max", patience=1)

callbacks_list = [checkpoint, early] #early
batch_size = 32
nb_epoch = 10
VERBOSEFLAG = 1
patience = 3

callbacks = [
                EarlyStopping(monitor='val_acc', patience=patience, verbose=VERBOSEFLAG, mode='max'),
                ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True, verbose=VERBOSEFLAG, mode='max'),
                        ]

model.fit_generator( train_datagen.flow( X_build, y_build, batch_size = batch_size,shuffle=True),
                         samples_per_epoch = math.ceil(len(X_build) / batch_size), nb_epoch = nb_epoch, callbacks = callbacks,
                         validation_data=valid_datagen.flow(X_valid, y_valid, batch_size=batch_size), 
                         validation_steps = math.ceil(len(X_valid) / batch_size), verbose = VERBOSEFLAG )

#print(history)

model.load_weights(file_path)

predicts = model.predict(X_test)
predicts = np.argmax(predicts, axis=1)
predicts = [label_index[p] for p in predicts]

df = pd.DataFrame(columns=['fname', 'camera'])
df['fname'] = index
df['camera'] = predicts
df.to_csv("sub.csv", index=False)


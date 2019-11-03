# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:02:03 2017

@author: SriPrav
"""
import numpy as np

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
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, UpSampling2D
from keras.layers.pooling import  AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras.layers.normalization import BatchNormalization
from keras import __version__ as keras_version

from keras.layers import Input, merge, Reshape
from keras.models import Model
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator
from Models.scale_layer import Scale

from sklearn.metrics import fbeta_score

import gc

    
np.random.seed(2017)

inDir = 'C:/Users/SriPrav/Documents/R/27Planet'

MODEL_WEIGHTS_FILE = inDir + '/Unet_CCN01_weights.h5'

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
print(test_all.shape)  # (61191, 2)

def double_conv_layer(x, size, dropout, batch_norm):
    from keras.models import Model
    from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.core import Dropout, Activation
    conv = Convolution2D(size, 3, 3, border_mode='same')(x)
    if batch_norm == True:
        conv = BatchNormalization(mode=0, axis=1)(conv)
    conv = Activation('relu')(conv)
    conv = Convolution2D(size, 3, 3, border_mode='same')(conv)
    if batch_norm == True:
        conv = BatchNormalization(mode=0, axis=1)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    return conv


def UNET_224(ROWS, COLUMNS, CHANNELS, num_classes=17,dropout_val=0.05, batch_norm=True):
    inputs = Input((CHANNELS, ROWS, COLUMNS))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(num_classes, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    
    return model
    



ROWS     = 224
COLUMNS  = 224
CHANNELS = 3
VERBOSEFLAG = 2

train_data_224_3   = np.load(inDir +"/input/train_data_224_3.npy")
train_target_224_3 = np.load(inDir +"/input/train_target_224_3.npy")
train_id_224_3     = np.load(inDir +"/input/train_id_224_3.npy")

train_data_rotate_aug_224_3   = np.load(inDir +"/input/train_data_rotate_aug_224_3.npy")
train_target_rotate_aug_224_3 = np.load(inDir +"/input/train_target_rotate_aug_224_3.npy")
train_id_rotate_aug_224_3     = np.load(inDir +"/input/train_id_rotate_aug_224_3.npy")

train_data_hflip_aug_224_3   = np.load(inDir +"/input/train_data_hflip_aug_224_3.npy")
train_target_hflip_aug_224_3 = np.load(inDir +"/input/train_target_hflip_aug_224_3.npy")
train_id_hflip_aug_224_3     = np.load(inDir +"/input/train_id_hflip_aug_224_3.npy")



test_data_224_3    = np.load(inDir +"/input/test_data_224_3.npy")
test_id_224_3      = np.load(inDir +"/input/test_id_224_3.npy")

train_data_224_3 = train_data_224_3.astype('float32')
train_data_rotate_aug_224_3 = train_data_rotate_aug_224_3.astype('float32')
train_data_hflip_aug_224_3 = train_data_hflip_aug_224_3.astype('float32')

gc.collect()
train_data_224_3 = train_data_224_3 / 255.
train_data_rotate_aug_224_3 = train_data_rotate_aug_224_3 / 255.
train_data_hflip_aug_224_3 = train_data_hflip_aug_224_3 / 255.
    
test_data_224_3 = test_data_224_3.astype('float32')
test_data_224_3 = test_data_224_3 / 255.


train_datagen = ImageDataGenerator(        
                                  
                                  )

valid_datagen = ImageDataGenerator(
                                   
                                   )

test_datagen = ImageDataGenerator(        
                                 
                                 )

batch_size = 16
nb_epoch = 100
random_state = 2017
patience = 10
optim_type = 'Adam'
learning_rate = 1e-3

i= 10 
    
def train_nn(i):
    trainindex = train_df[train_df['CVindices'] != i].index.tolist()
    valindex   = train_df[train_df['CVindices'] == i].index.tolist()
    X_val_df = train_df.iloc[valindex,:]
    X_build1 , X_valid = train_data_224_3[trainindex,:], train_data_224_3[valindex,:]
    y_build1 , y_valid = train_target_224_3[trainindex,:], train_target_224_3[valindex,:]  
    
    X_build2  = train_data_rotate_aug_224_3[trainindex,:]
    y_build2  = train_target_rotate_aug_224_3[trainindex,:]
    
    X_build3  = train_data_hflip_aug_224_3[trainindex,:]
    y_build3  = train_target_hflip_aug_224_3[trainindex,:] 
    
    del train_data_rotate_aug_224_3, train_data_hflip_aug_224_3
    
    X_build  = np.vstack([np.vstack([X_build1,X_build2]),X_build3])
    y_build  = np.vstack([np.vstack([y_build1,y_build2]),y_build3])
    
    del X_build1,X_build2,X_build3
    
    print('Split train: ', len(X_build), len(y_build)) #('Split train: ', 109410, 109410)
    print('Split valid: ', len(X_valid), len(y_valid)) #('Split valid: ', 4009, 4009)
    model = UNET_224(ROWS, COLUMNS, CHANNELS, num_classes=17,dropout_val=0.05, batch_norm=True)    
#        callbacks = [
#            EarlyStopping(monitor='val_loss', patience=3, verbose=VERBOSEFLAG),
#        ]
#    callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=1)]
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=0),
                ]
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])#'categorical_crossentropy'
#        model.compile(loss='categorical_crossentropy', optimizer="adadelta", \
#              metrics=["accuracy"])
    model.fit_generator( train_datagen.flow( X_build, y_build, batch_size = batch_size,shuffle=True),
                             samples_per_epoch = len(X_build), nb_epoch = nb_epoch, callbacks = callbacks,
                             validation_data=valid_datagen.flow(X_valid, y_valid, batch_size=batch_size), 
                             nb_val_samples=X_valid.shape[0], verbose = VERBOSEFLAG )
#    min_loss = min(model.model['val_loss'])
#    print('Minimum loss for given fold: ', min_loss)
    model.load_weights(MODEL_WEIGHTS_FILE)
        
    pred_cv = model.predict_generator(valid_datagen.flow(X_valid, batch_size=batch_size,shuffle=False),val_samples=X_valid.shape[0])
    print('F2 Score : ',fbeta_score(y_valid, np.array(pred_cv) > 0.2, beta=2, average='samples'))
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.head()
    pred_cv.columns = ["slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]
    pred_cv["image_name"] = X_val_df.image_name.values
    
    sub_valfile = inDir + '/submissions/Prav.Unet_01.fold' + str(i) + '.csv'
    pred_cv = pred_cv[["image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]]
    pred_cv.to_csv(sub_valfile, index=False)
    pred_test = model.predict_generator(test_datagen.flow(test_data_224_3, batch_size=batch_size,shuffle=False),val_samples=test_data_224_3.shape[0])
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]
    pred_test["image_name"] = test_all.image_name.values
    pred_test = pred_test[["image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]]
    sub_file = inDir + '/submissions/Prav.Unet_01.fold' + str(i) + '-test'+'.csv'
    pred_test.to_csv(sub_file, index=False)   



i = 10
train_nn(i)


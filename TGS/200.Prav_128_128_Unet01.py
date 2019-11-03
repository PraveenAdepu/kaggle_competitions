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

import tensorflow as tf

from sklearn.metrics import fbeta_score

import gc
from skimage.transform import resize
    
np.random.seed(201803)

inDir = 'C:/Users/SriPrav/Documents/R/51TGS'

MODEL_WEIGHTS_FILE = inDir + '/Unet_CCN01_weights.h5'

train_file = inDir + "/input/Prav_10folds_CVindices.csv"
test_file = inDir + "/input/test_images.csv"

train_df = pd.read_csv(train_file)
test_all = pd.read_csv(test_file)

print(train_df.shape) # (40479, 4)
print(test_all.shape)  # (61191, 2)

smooth = 1.
#image_rows = 1280
#image_cols = 1918
#def prep(img):
#    img = img.astype('float32')
#    img = (img > 0.5).astype(np.uint8)  # threshold
##        img = resize(img, (image_cols, image_rows), preserve_range=True)
#    return img
    
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

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

from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate

# Build U-Net model
def unet():
    inputs = Input((128, 128, 1))
    s = Lambda(lambda x: x / 255) (inputs)
    
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)
    
    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)
    
    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)
    
    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)
    
    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
#    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
#    model.summary()
    return model

#def ZF_UNET_128(dropout_val=0.05, batch_norm=True):
#    from keras.models import Model
#    from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
#    from keras.layers.normalization import BatchNormalization
#    from keras.layers.core import Dropout, Activation
#    inputs = Input(( 101, 101,1))
#    conv1 = double_conv_layer(inputs, 32, dropout_val, batch_norm)
##    pool1 = MaxPooling2D(pool_size=(1, 1))(conv1)
#
#    conv2 = double_conv_layer(conv1, 64, dropout_val, batch_norm)
##    pool2 = MaxPooling2D(pool_size=(1, 1))(conv2)
#
#    conv3 = double_conv_layer(conv2, 128, dropout_val, batch_norm)
##    pool3 = MaxPooling2D(pool_size=(1, 1))(conv3)
#
#    conv4 = double_conv_layer(conv3, 256, dropout_val, batch_norm)
##    pool4 = MaxPooling2D(pool_size=(1, 1))(conv4)
#
#    conv5 = double_conv_layer(conv4, 512, dropout_val, batch_norm)
#
#    up5 = merge([UpSampling2D(size=(1, 1))(conv5), conv4], mode='concat', concat_axis=3)
#    conv6 = double_conv_layer(up5, 256, dropout_val, batch_norm)
#
#    up6 = merge([UpSampling2D(size=(1, 1))(conv6), conv3], mode='concat', concat_axis=3)
#    conv7 = double_conv_layer(up6, 128, dropout_val, batch_norm)
#
#    up7 = merge([UpSampling2D(size=(1, 1))(conv7), conv2], mode='concat', concat_axis=3)
#    conv8 = double_conv_layer(up7, 64, dropout_val, batch_norm)
#
#    up8 = merge([UpSampling2D(size=(1, 1))(conv8), conv1], mode='concat', concat_axis=3)
#    conv9 = double_conv_layer(up8, 32, 0, batch_norm)
#
#    conv10 = Convolution2D(1, 1, 1, border_mode='same')(conv9)
##    conv10 = BatchNormalization(mode=0, axis=1)(conv10)
#    conv10 = Activation('sigmoid')(conv10)
#
#    model = Model(input=inputs, output=conv10)
#    return model

#def ZF_UNET_128(dropout_val=0.05, batch_norm=True):
#    from keras.models import Model
#    from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
#    from keras.layers.normalization import BatchNormalization
#    from keras.layers.core import Dropout, Activation
#    inputs = Input(( 101, 101,1))
#    conv1 = double_conv_layer(inputs, 32, dropout_val, batch_norm)
#    pool1 = MaxPooling2D(pool_size=(1, 1))(conv1)
#
#    conv2 = double_conv_layer(pool1, 64, dropout_val, batch_norm)
#    pool2 = MaxPooling2D(pool_size=(1, 1))(conv2)
#
#    conv3 = double_conv_layer(pool2, 128, dropout_val, batch_norm)
#    pool3 = MaxPooling2D(pool_size=(1, 1))(conv3)
#
#    conv4 = double_conv_layer(pool3, 256, dropout_val, batch_norm)
#    pool4 = MaxPooling2D(pool_size=(1, 1))(conv4)
#
#    conv5 = double_conv_layer(pool2, 512, dropout_val, batch_norm)
#
#    up5 = merge([UpSampling2D(size=(1, 1))(conv5), conv4], mode='concat', concat_axis=3)
#    conv6 = double_conv_layer(up5, 256, dropout_val, batch_norm)
#
#    up6 = merge([UpSampling2D(size=(1, 1))(conv6), conv3], mode='concat', concat_axis=3)
#    conv7 = double_conv_layer(up6, 128, dropout_val, batch_norm)
#
#    up7 = merge([UpSampling2D(size=(1, 1))(conv7), conv2], mode='concat', concat_axis=3)
#    conv8 = double_conv_layer(up7, 64, dropout_val, batch_norm)
#
#    up8 = merge([UpSampling2D(size=(1, 1))(conv8), conv1], mode='concat', concat_axis=3)
#    conv9 = double_conv_layer(up8, 32, 0, batch_norm)
#
#    conv10 = Convolution2D(1, 1, 1, border_mode='same')(conv9)
##    conv10 = BatchNormalization(mode=0, axis=1)(conv10)
#    conv10 = Activation('sigmoid')(conv10)
#
#    model = Model(input=inputs, output=conv10)
#    return model

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
    
#def ZF_UNET_128(dropout_val=0.05, batch_norm=True):
#    from keras.models import Model
#    from keras.layers import Conv2D, Input, Concatenate
#    from keras.callbacks import EarlyStopping, ModelCheckpoint
#    def conv_block(num_layers,inp,units,kernel):
#        x = inp
#        for l in range(num_layers):
#            x = Conv2D(units, kernel_size=kernel, padding='SAME',activation='relu')(x)
#        return x
#    
#    
#    inp = Input(shape=(101,101,1))
#    cnn1 = conv_block(2,inp,64,3)
#    cnn2 = conv_block(4,inp,32,3)
#    cnn3 = conv_block(4,inp,24,5)
#    cnn4 = conv_block(4,inp,16,7)
#    concat = Concatenate()([cnn1,cnn2,cnn3,cnn4])
#    d1 = Conv2D(64,1, activation='relu')(concat)
#    d2 = Conv2D(16,1, activation='relu')(d1)
#    out = Conv2D(1,1, activation='sigmoid')(d2)
#    
#    model = Model(inputs = inp, outputs = out)
##    model.summary()
#    return model
#
#from keras.models import *
#from keras.layers import *
#from keras.optimizers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from keras import backend as keras
#
#def unet(input_size = (104,104,1)):
#    inputs = Input(input_size)
#    conv1 = Convolution2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
#    conv1 = Convolution2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
#    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#    conv2 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
#    conv2 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
#    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#    conv3 = Convolution2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
#    conv3 = Convolution2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
#    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#    conv4 = Convolution2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
#    conv4 = Convolution2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
#    drop4 = Dropout(0.5)(conv4)
#    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#
#    conv5 = Convolution2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
#    conv5 = Convolution2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
#    drop5 = Dropout(0.5)(conv5)
#
#    up6 = Convolution2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
#    merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
#    conv6 = Convolution2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
#    conv6 = Convolution2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
#
#    up7 = Convolution2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
#    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
#    conv7 = Convolution2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
#    conv7 = Convolution2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
#
#    up8 = Convolution2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
#    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
#    conv8 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
#    conv8 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
#
#    up9 = Convolution2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
#    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
#    conv9 = Convolution2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
#    conv9 = Convolution2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#    conv9 = Convolution2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#    conv10 = Convolution2D(1, 1, activation = 'sigmoid')(conv9)
#
#    model = Model(input = inputs, output = conv10)
#
##    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
#    
##    model.summary()
#    return model

#    if(pretrained_weights):
#    	model.load_weights(pretrained_weights)
#
#    return model

#def ZF_UNET_128(dropout_val=0.05, batch_norm=True):
#    from keras.models import Model
#    from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
#    from keras.layers.normalization import BatchNormalization
#    from keras.layers.core import Dropout, Activation
#    inputs = Input(( 101, 101,1))
#    conv1 = double_conv_layer(inputs, 32, dropout_val, batch_norm)
#    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#
#    conv2 = double_conv_layer(pool1, 64, dropout_val, batch_norm)
#    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#    conv3 = double_conv_layer(pool2, 128, dropout_val, batch_norm)
#    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
#    conv4 = double_conv_layer(pool3, 256, dropout_val, batch_norm)
#    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#    conv5 = double_conv_layer(pool4, 512, dropout_val, batch_norm)
#
#    up5 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat')
#    conv6 = double_conv_layer(up5, 256, dropout_val, batch_norm)
#
#    up6 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat')
#    conv7 = double_conv_layer(up6, 128, dropout_val, batch_norm)
#
#    up7 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat')
#    conv8 = double_conv_layer(up7, 64, dropout_val, batch_norm)
#
#    up8 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat')
#    conv9 = double_conv_layer(up8, 32, 0, batch_norm)
#
#    conv10 = Convolution2D(1, 1, 1, border_mode='same')(conv9)
##    conv10 = BatchNormalization(mode=0, axis=1)(conv10)
#    conv10 = Activation('sigmoid')(conv10)
#
#    model = Model(input=inputs, output=conv10)
#    return model
    
#def ZF_UNET_128():
#    inputs = Input((3, 128, 128, 1))
#    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
#    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
#    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#
#    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
#    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
#    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
#    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
#    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
#    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
#    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
#    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
#    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
#
#    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
#    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
#    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)
#
#    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
#    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
#    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)
#
#    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
#    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
#    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
#
#    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
#    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
#    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
#
#    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
#
#    model = Model(input=[inputs], output=[conv10])
#
#    #model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
#
#    return model
    
#def ZF_UNET_128(dropout_val=0.05, batch_norm=True):
#    from keras.models import Model
#    from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
#    from keras.layers.normalization import BatchNormalization
#    from keras.layers.core import Dropout, Activation
#    inputs = Input((3, 128, 128))
#    conv1 = double_conv_layer(inputs, 32, dropout_val, batch_norm)
#    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#
#    conv2 = double_conv_layer(pool1, 64, dropout_val, batch_norm)
#    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#    conv3 = double_conv_layer(pool2, 128, dropout_val, batch_norm)
#    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
#    conv4 = double_conv_layer(pool3, 256, dropout_val, batch_norm)
#    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#    conv5 = double_conv_layer(pool4, 512, dropout_val, batch_norm)
#    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
#
#    conv6 = double_conv_layer(pool5, 1024, dropout_val, batch_norm)
#
#    up6 = merge([UpSampling2D(size=(2, 2))(conv6), conv5], mode='concat', concat_axis=1)
#    conv7 = double_conv_layer(up6, 512, dropout_val, batch_norm)
#
#    up7 = merge([UpSampling2D(size=(2, 2))(conv7), conv4], mode='concat', concat_axis=1)
#    conv8 = double_conv_layer(up7, 256, dropout_val, batch_norm)
#
#    up8 = merge([UpSampling2D(size=(2, 2))(conv8), conv3], mode='concat', concat_axis=1)
#    conv9 = double_conv_layer(up8, 128, dropout_val, batch_norm)
#
#    up9 = merge([UpSampling2D(size=(2, 2))(conv9), conv2], mode='concat', concat_axis=1)
#    conv10 = double_conv_layer(up9, 64, dropout_val, batch_norm)
#
#    up10 = merge([UpSampling2D(size=(2, 2))(conv10), conv1], mode='concat', concat_axis=1)
#    conv11 = double_conv_layer(up10, 32, 0, batch_norm)
#
#    conv12 = Convolution2D(1, 1, 1)(conv11)
#    conv12 = BatchNormalization(mode=0, axis=1)(conv12)
#    conv12 = Activation('sigmoid')(conv12)
#
#    model = Model(input=inputs, output=conv12)
#    return model
#def get_unet():
#    inputs = Input((img_rows, img_cols, 1))
#    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(inputs)
#    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv1)
#    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#
#    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool1)
#    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv2)
#    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool2)
#    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv3)
#    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
#    conv4 = Convolution2D(256, (3, 3), activation='relu', padding='same')(pool3)
#    conv4 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv4)
#    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#    conv5 = Convolution2D(512, (3, 3), activation='relu', padding='same')(pool4)
#    conv5 = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv5)
#
#    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
#    conv6 = Convolution2D(256, (3, 3), activation='relu', padding='same')(up6)
#    conv6 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv6)
#
#    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
#    conv7 = Convolution2D(128, (3, 3), activation='relu', padding='same')(up7)
#    conv7 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv7)
#
#    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
#    conv8 = Convolution2D(64, (3, 3), activation='relu', padding='same')(up8)
#    conv8 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv8)
#
#    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
#    conv9 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up9)
#    conv9 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv9)
#
#    conv10 = Convolution2D(1, (1, 1), activation='sigmoid')(conv9)
#
#    model = Model(inputs=[inputs], outputs=[conv10])
#
#    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
#
#    return model
#    



ROWS     = 128
COLUMNS  = 128
CHANNELS = 1
VERBOSEFLAG = 2

train_data_224_3   = np.load(inDir +"/input/train_data_128_1.npy")
train_target_224_3 = np.load(inDir +"/input/train_target_128_1.npy")
train_id_224_3     = np.load(inDir +"/input/train_id_128_1.npy")


test_data_224_3    = np.load(inDir +"/input/test_data_128_1.npy")
test_id_224_3      = np.load(inDir +"/input/test_id_128_1.npy")
test_sizes     = np.load(inDir +"/input/test_sizes_128_1.npy")
#
#   

train_data_224_3 = train_data_224_3.astype('float32')
#mean = np.mean(train_data_224_3)  # mean for data centering
#std = np.std(train_data_224_3)  # std for data normalization

#train_data_224_3 -= mean
#train_data_224_3 /= std

train_data_224_3 /= 255.

train_target_224_3 = train_target_224_3.astype('float32')
train_target_224_3 /= 255.  # scale masks to [0, 1]

test_data_224_3 = test_data_224_3.astype('float32')
#mean = np.mean(test_data_224_3)  # mean for data centering
#std = np.std(test_data_224_3)  # std for data normalization

#test_data_224_3 -= mean
#test_data_224_3 /= std

test_data_224_3 /= 255.

train_datagen = ImageDataGenerator(        
                                  
                                  )

valid_datagen = ImageDataGenerator(
                                   
                                   )

test_datagen = ImageDataGenerator(        
                                 
                                 )

batch_size = 16
nb_epoch = 5
random_state = 2017
patience = 5
optim_type = 'Adam'
learning_rate = 1e-3

i= 10

    
def train_nn(i):
    trainindex = train_df[train_df['CVindices'] != i].index.tolist()
    valindex   = train_df[train_df['CVindices'] == i].index.tolist()
    X_val_df = train_df.iloc[valindex,:]
    X_build , X_valid = train_data_224_3[trainindex,:], train_data_224_3[valindex,:]
    y_build , y_valid = train_target_224_3[trainindex,:], train_target_224_3[valindex,:]  
       
    print('Split train: ', len(X_build), len(y_build)) #('Split train: ', 4528, 4528)
    print('Split valid: ', len(X_valid), len(y_valid)) #('Split valid: ', 560, 560)
    model = unet()    
#    model = unet()  
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=VERBOSEFLAG),
        ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=2),
                ]
    
    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[mean_iou])#'categorical_crossentropy'dice_coef_loss , metrics=[dice_coef]
#    model.summary()
#        model.compile(loss='categorical_crossentropy', optimizer="adadelta", \
#              metrics=["accuracy"])
#    model.fit(train_data_224_3, train_target_224_3, batch_size=32, nb_epoch=20, shuffle=True,
#              validation_split=0.1,
#              callbacks = callbacks, verbose = VERBOSEFLAG)
    model.fit_generator( train_datagen.flow( X_build, y_build, batch_size = batch_size,shuffle=True),
                             samples_per_epoch = len(X_build), nb_epoch = nb_epoch, callbacks = callbacks,
                             validation_data=valid_datagen.flow(X_valid, y_valid, batch_size=batch_size), 
                             nb_val_samples=X_valid.shape[0], verbose = VERBOSEFLAG )
#    min_loss = min(model.model['val_loss'])
#    print('Minimum loss for given fold: ', min_loss)
    model.load_weights(MODEL_WEIGHTS_FILE)
        
 
    imgs_mask_val = model.predict(X_valid, verbose = True)
    np.save('imgs_mask_val_Unet01.npy', imgs_mask_val)
    
    imgs_mask_test = model.predict(test_data_224_3, verbose = True)
    
    np.save('imgs_mask_test_Unet01.npy', imgs_mask_test)
    
    # Create list of upsampled test masks
    preds_test_upsampled = []
    for i in range(len(imgs_mask_test)):
        preds_test_upsampled.append(resize(np.squeeze(imgs_mask_test[i]), 
                                           (101, 101), 
                                           mode='constant', preserve_range=True))

    np.save('imgs_mask_test_Unet01_01.npy', preds_test_upsampled)
i = 10
train_nn(i)


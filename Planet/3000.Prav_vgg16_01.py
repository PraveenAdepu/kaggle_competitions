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
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.pooling import  AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras.layers.normalization import BatchNormalization
from keras import __version__ as keras_version

from keras.layers import Input, merge, Reshape
from keras.models import Model
from keras import backend as K


from Models.scale_layer import Scale

from sklearn.metrics import fbeta_score

inDir = 'C:/Users/SriPrav/Documents/R/27Planet'
vgg16_weights = 'C:/Users/SriPrav/Documents/R' + '/imagenet_models/vgg16_weights.h5'
MODEL_WEIGHTS_FILE = inDir + '/vgg16_CCN01_weights.h5'

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

def vgg16_model(img_rows, img_cols, channel=3, num_classes=None):
    """VGG 16 Model for Keras
    Model Schema is based on 
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    ImageNet Pretrained Weights 
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of categories for our classification task
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channel, img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    
    # Loads ImageNet pre-trained data
    model.load_weights(vgg16_weights)

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    #for layer in model.layers[:10]:
    #    layer.trainable = False

    # Learning rate is changed to 0.001
    
    

    return model


ROWS     = 224
COLUMNS  = 224
CHANNELS = 3
VERBOSEFLAG = 2

train_data_224_3   = np.load(inDir +"/input/train_data_224_3.npy")
train_target_224_3 = np.load(inDir +"/input/train_target_224_3.npy")
train_id_224_3     = np.load(inDir +"/input/train_id_224_3.npy")

test_data_224_3    = np.load(inDir +"/input/test_data_224_3.npy")
test_id_224_3      = np.load(inDir +"/input/test_id_224_3.npy")

train_data_224_3 = train_data_224_3.astype('float32')
#train_data_224_3 = train_data_224_3 / 255
## check mean pixel value
mean_pixel = [103.939, 116.779, 123.68]
for c in range(3):
    train_data_224_3[:, c, :, :] = train_data_224_3[:, c, :, :] - mean_pixel[c]
# train_data /= 255
    
test_data_224_3 = test_data_224_3.astype('float32')
#test_data_224_3 = test_data_224_3 / 255
for c in range(3):
    test_data_224_3[:, c, :, :] = test_data_224_3[:, c, :, :] - mean_pixel[c]

batch_size = 16
nb_epoch = 25
random_state = 2017

    
def train_nn(i):
    trainindex = train_df[train_df['CVindices'] != i].index.tolist()
    valindex   = train_df[train_df['CVindices'] == i].index.tolist()
    X_val_df = train_df.iloc[valindex,:]
    X_build , X_valid = train_data_224_3[trainindex,:], train_data_224_3[valindex,:]
    y_build , y_valid = train_target_224_3[trainindex,:], train_target_224_3[valindex,:]  
    
    print('Split train: ', len(X_build), len(y_build))
    print('Split valid: ', len(X_valid), len(y_valid))
    model = vgg16_model(ROWS, COLUMNS, CHANNELS, num_classes=17)    
#        callbacks = [
#            EarlyStopping(monitor='val_loss', patience=3, verbose=VERBOSEFLAG),
#        ]
    callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=1)] 
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)    
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])#'categorical_crossentropy'
#        model.compile(loss='categorical_crossentropy', optimizer="adadelta", \
#              metrics=["accuracy"])
    model.fit(X_build, y_build, batch_size=batch_size, nb_epoch=nb_epoch,
          shuffle=True, verbose=VERBOSEFLAG, validation_data=(X_valid, y_valid),
          callbacks=callbacks
          )
    model.load_weights(MODEL_WEIGHTS_FILE)
        
    pred_cv = model.predict(X_valid, verbose=1)
    print('F2 Score : ',fbeta_score(y_valid, np.array(pred_cv) > 0.2, beta=2, average='samples'))
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.head()
    pred_cv.columns = ["slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]
    pred_cv["image_name"] = X_val_df.image_name.values
    
    sub_valfile = inDir + '/submissions/Prav.vgg16_01.fold' + str(i) + '.csv'
    pred_cv = pred_cv[["image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]]
    pred_cv.to_csv(sub_valfile, index=False)
    pred_test = model.predict(test_data_224_3,verbose=1)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]
    pred_test["image_name"] = test_all.image_name.values
    pred_test = pred_test[["image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]]
    sub_file = inDir + '/submissions/Prav.vgg16_01.fold' + str(i) + '-test'+'.csv'
    pred_test.to_csv(sub_file, index=False)   



i = 10
train_nn(i)


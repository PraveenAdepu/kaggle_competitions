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
vgg16_weights = inDir + '/preModels/vgg16_weights.h5'
MODEL_WEIGHTS_FILE = inDir + '/VGG_CCN01_weights.h5'
#
#train_file = inDir + "/input/rectangles_train.csv"
#test_file = inDir + "/input/rectangles_test.csv"
#train_df = pd.read_csv(train_file)
#test_df = pd.read_csv(test_file)
#print(train_df.shape) # (8211, 11)
#print(test_df.shape)  # (512, 12)
#
#train_df.head(2)

ROWS     = 224
COLUMNS  = 224
CHANNELS = 3
VERBOSEFLAG = 1

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


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['Type_1', 'Type_2', 'Type_3'])
    result1.loc[:, 'image_name'] = pd.Series(test_id, index=result1.index)
    imgcolumn = result1['image_name']
    result1.drop(labels=['image_name'], axis=1,inplace = True)
    result1.insert(0, 'image_name', imgcolumn)
    now = datetime.datetime.now()
    sub_file = 'submissions\Prav.VGG16_01_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def vgg_std16_model(ROWS, COLUMNS, CHANNELS=3):
    model = Sequential()
    #model.add(Lambda(vgg_preprocess, input_shape=(CHANNELS,ROWS, COLUMNS))
    model.add(ZeroPadding2D((1, 1), input_shape=(CHANNELS,ROWS, COLUMNS)))
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

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights(vgg16_weights)
    return model
    

def create_model():
    model = vgg_std16_model(ROWS, COLUMNS, CHANNELS)
    model.pop(); model.pop(); model.pop(); model.pop(); model.pop();
    
    for layer in model.layers:
        layer.trainable = False
    
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax')) 
    return model
    
  
             
def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_cross_validation_create_models(nfolds=10):
    # input image dimensions
    batch_size = 32
    nb_epoch = 25
    random_state = 2017

    train_data = train_data_224_3
    train_target = train_target_224_3
    train_id = train_id_224_3

    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf:
        model = create_model()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        
#        callbacks = [
#            EarlyStopping(monitor='val_loss', patience=3, verbose=VERBOSEFLAG),
#        ]
        callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=1)] 
        model.compile(loss='categorical_crossentropy', optimizer="adadelta", \
              metrics=["accuracy"])
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=VERBOSEFLAG, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)
        model.load_weights(MODEL_WEIGHTS_FILE)
        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)
        os.remove(MODEL_WEIGHTS_FILE)
    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)
    return info_string, models


def run_cross_validation_process_test(info_string, models):
    batch_size = 16
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)
    i = 1
    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data = test_data_224_3
        test_id = test_id_224_3
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=VERBOSEFLAG)
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                + '_folds_' + str(nfolds)
    create_submission(test_res, test_id, info_string)


if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 5
    info_string, models = run_cross_validation_create_models(num_folds)
    run_cross_validation_process_test(info_string, models)
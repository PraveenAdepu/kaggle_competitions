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

train_file = inDir + "/input/rectangles_train.csv"
test_file = inDir + "/input/rectangles_test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train_df.shape) # (8211, 11)
print(test_df.shape)  # (512, 12)

train_df.head(2)

ROWS     = 224
COLUMNS  = 224
CHANNELS = 3
VERBOSEFLAG = 1

def get_im_cv2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    resized = cv2.resize(img, (ROWS, COLUMNS), cv2.INTER_LINEAR)
    return img
    
y_train = []
y_train = train_df.clss.values

#fl = "C:\Users\SriPrav\Documents\R\\22Intel\\input\\train\\Type_1\\Type_1\\7.jpg"
#img = get_im_cv2(fl)
#if (img.shape[0] > img.shape[1]):
#        tile_size = (int(img.shape[1] * 256 / img.shape[0]), 256)
#    else:
#        tile_size = (256, int(img.shape[0] * 256 / img.shape[1]))
#        
#plt.imshow(img)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#resized = cv2.resize(img, (192, 256), cv2.INTER_LINEAR)
#plt.imshow(resized)
#train_row   = train_df[train_df['image_path'] == fl]
#y = train_row['sh0_start'].astype(int)
#x = train_row['sh1_start'].astype(int)
#yh = train_row['sh0_end'].astype(int)
#xw = train_row['sh1_end'].astype(int)
#img_crop = resized[int(y): int(yh) , int(x): int(xw)]
#plt.imshow(img_crop)
        
def load_train_fromfile():
    X_train = []
    X_train_id = []

    start_time = time.time()

    for fl in train_df.image_path.values:
        print(fl)
        flbase = os.path.basename(fl) # "C:\Users\SriPrav\Documents\R\\22Intel\\input\\train\\Type_1\\a_Type_1\\10.jpg"
        img = get_im_cv2(fl)
        train_row   = train_df[train_df['image_path'] == fl]
        img_shape0 = train_row['img_shp_0'].astype(int)
        img_shape1 = train_row['img_shp_1'].astype(int)
        y = train_row['sh0_start'].astype(int)
        x = train_row['sh1_start'].astype(int)
        yh = train_row['sh0_end'].astype(int)
        xw = train_row['sh1_end'].astype(int)
        img = cv2.resize(img, (int(img_shape0), int(img_shape1)), cv2.INTER_LINEAR)
        img = img[int(y): int(yh) , int(x): int(xw)]
        img = cv2.resize(img, (ROWS, COLUMNS), cv2.INTER_LINEAR)
        X_train.append(img)
        X_train_id.append(flbase)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, X_train_id
    
# C:\Users\SriPrav\Documents\R\22Intel\input\train\Type_1\a_Type_1\2786.jpg
# C:\Users\SriPrav\Documents\R\22Intel\input\train\Type_1\a_Type_1\4944.jpg 
# C:\Users\SriPrav\Documents\R\22Intel\input\train\Type_1\a_Type_1\5770.jpg
# C:\Users\SriPrav\Documents\R\22Intel\input\train\Type_1\a_Type_1\6038.jpg
# C:\Users\SriPrav\Documents\R\22Intel\input\train\Type_1\a_Type_1\6365.jpg
# C:\Users\SriPrav\Documents\R\22Intel\input\train\Type_2\a_Type_2\2462.jpg
# C:\Users\SriPrav\Documents\R\22Intel\input\train\Type_2\a_Type_2\2731.jpg
#C:\Users\SriPrav\Documents\R\22Intel\input\train\Type_2\a_Type_2\3243.jpg
#C:\Users\SriPrav\Documents\R\22Intel\input\train\Type_2\a_Type_2\4953.jpg
#C:\Users\SriPrav\Documents\R\22Intel\input\train\Type_2\Type_2\1325.jpg
#C:\Users\SriPrav\Documents\R\22Intel\input\train\Type_3\a_Type_3\5585.jpg
#C:\Users\SriPrav\Documents\R\22Intel\input\train\Type_3\a_Type_3\6058.jpg
#def load_train():
#    X_train = []
#    X_train_id = []
#    y_train = []
#    start_time = time.time()
#
#    print('Read train images')
#    folders = ['Type_1', 'Type_2', 'Type_3']
#    for fld in folders:
#        index = folders.index(fld)
#        print('Load folder {} (Index: {})'.format(fld, index))
#        path = os.path.join('C:\Users\SriPrav\Documents\R\\22Intel\\input\\train\\', fld, '*\\*.jpg')
#        files = glob.glob(path)
#        for fl in files:
#            flbase = os.path.basename(fl) # "C:\Users\SriPrav\Documents\R\\22Intel\\input\\train\\Type_1\\a_Type_1\\10.jpg"
#            img = get_im_cv2(fl)
#            X_train.append(img)
#            X_train_id.append(flbase)
#            y_train.append(index)
#
#    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
#    return X_train, y_train, X_train_id

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
    
#def load_test():
#    path = os.path.join('C:\Users\SriPrav\Documents\R\\22Intel', 'input', 'stage1','test', '*.jpg')
#    files = sorted(glob.glob(path))
#
#    X_test = []
#    X_test_id = []
#    for fl in files:
#        flbase = os.path.basename(fl)
#        img = get_im_cv2(fl)
#        X_test.append(img)
#        X_test_id.append(flbase)
#
#    return X_test, X_test_id


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['Type_1', 'Type_2', 'Type_3'])
    result1.loc[:, 'image_name'] = pd.Series(test_id, index=result1.index)
    imgcolumn = result1['image_name']
    result1.drop(labels=['image_name'], axis=1,inplace = True)
    result1.insert(0, 'image_name', imgcolumn)
    now = datetime.datetime.now()
    sub_file = 'submissions\prav.CNN07_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def read_and_normalize_train_data():
    train_data,  train_id = load_train_fromfile()
    train_target = y_train
    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 3)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test_fromfile()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id


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


#def create_model():
#    model = Sequential()
#    model.add(ZeroPadding2D((1, 1), input_shape=(CHANNELS,ROWS, COLUMNS), dim_ordering='th'))
#    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
#    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
#    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
#
#    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
#    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
#    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
#    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
#
#    model.add(Flatten())
#    model.add(Dense(32, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(32, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(8, activation='softmax'))
#
#    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(optimizer=sgd, loss='categorical_crossentropy')
#
#    return model

#def create_model():
#    model = Sequential()
#    model.add(ZeroPadding2D((1, 1), input_shape=(CHANNELS,ROWS, COLUMNS), dim_ordering='th'))
#    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
#    model.add(Dropout(0.2))
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
#    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
#    model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
#    model.add(Dropout(0.2))
#    
#    model.add(Flatten())
#    model.add(Dense(96, activation='relu',init='he_uniform'))
#    model.add(Dropout(0.4))
#    model.add(Dense(24, activation='relu',init='he_uniform'))
#    model.add(Dropout(0.2))
#    model.add(Dense(3, activation='softmax'))
#
#    sgd = SGD(lr=1e-2, decay=1e-4, momentum=0.89, nesterov=False)
#    model.compile(optimizer=sgd, loss='categorical_crossentropy')
#
#    return model
    

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
    
callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=1)]   
             
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

    train_data, train_target, train_id = read_and_normalize_train_data()

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
        test_data, test_id = read_and_normalize_test_data()
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
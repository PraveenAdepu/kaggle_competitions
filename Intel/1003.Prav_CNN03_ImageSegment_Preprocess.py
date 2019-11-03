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
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version

ROWS     = 128
COLUMNS  = 128
CHANNELS = 3
VERBOSEFLAG = 1
###########################################################################################################################
#def get_filename(image_id, image_type):
#    """
#    Method to get image file path from its id and type   
#    """
#    if image_type == "Type_1" or \
#        image_type == "Type_2" or \
#        image_type == "Type_3":
#        data_path = os.path.join(TRAIN_DATA, image_type)
#    elif image_type == "Test":
#        data_path = TEST_DATA
#    elif image_type == "AType_1" or \
#          image_type == "AType_2" or \
#          image_type == "AType_3":
#        data_path = os.path.join(ADDITIONAL_DATA, image_type[1:])
#    else:
#        raise Exception("Image type '%s' is not recognized" % image_type)
#
#    ext = 'jpg'
#    return os.path.join(data_path, "{}.{}".format(image_id, ext))
#
#
#def get_image_data(image_id, image_type):
#    """
#    Method to get image data as np.array specifying image id and type
#    """
#    fname = get_filename(image_id, image_type)
#    img = cv2.imread(fname)
#    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    return img

def sieve(image, size):
    """
    Filter removes small objects of 'size' from binary image
    Input image should be a single band image of type np.uint8
    Idea : use Opencv findContours
    """
    sqLimit = size**2
    linLimit = size*4
    outImage = image.copy()
    image, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if ((hierarchy is not None) and (len(hierarchy) > 0)):
        hierarchy = hierarchy[0]
        index = 0
        while index >= 0:
            contour = contours[index]
            p = cv2.arcLength(contour, True)
            s = cv2.contourArea(contour)
            r = cv2.boundingRect(contour)
            if s <= sqLimit and p <= linLimit:
                outImage[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = 0
            index = hierarchy[index][0]
    else:
        pass
        # print("No contours found")
        #outImage = image
    return outImage


# in HSV :
skin_range_1_min = np.array([120, 0, 0], dtype=np.uint8)
skin_range_1_max = np.array([255, 255, 255], dtype=np.uint8)

skin_range_2_min = np.array([0, 0, 0], dtype=np.uint8)
skin_range_2_max = np.array([45, 255, 255], dtype=np.uint8)

skin_kernel_size = 7
skin_sieve_min_size = 5

def detect_skin(image):
    proc = cv2.medianBlur(image, 7)
    ### Detect skin
    image_hsv = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)
    skin_like_mask = cv2.inRange(image_hsv, skin_range_1_min, skin_range_1_max)
    skin_like_mask_2 = cv2.inRange(image_hsv, skin_range_2_min, skin_range_2_max)
    skin_like_mask = cv2.bitwise_or(skin_like_mask, skin_like_mask_2)    
    # Filter the skin mask :
    skin_mask = sieve(skin_like_mask, skin_sieve_min_size)
    kernel = np.ones((skin_kernel_size, skin_kernel_size), dtype=np.int8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)    
    # Apply skin mask
    skin_segm_rgb = cv2.bitwise_and(image, image, mask=skin_mask)
    return skin_segm_rgb
############################################################################################################################




def get_im_cv2(path):
    img = cv2.imread(path)
    img = cv2.resize(img, dsize=(ROWS, COLUMNS))
    img = detect_skin(img)
    resized = cv2.resize(img, (ROWS, COLUMNS), cv2.INTER_LINEAR)    
    return resized


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['Type_1', 'Type_2', 'Type_3']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('C:\Users\SriPrav\Documents\R\\22Intel', 'input', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test():
    path = os.path.join('C:\Users\SriPrav\Documents\R\\22Intel', 'input', 'stage1','test', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['Type_1', 'Type_2', 'Type_3'])
    result1.loc[:, 'image_name'] = pd.Series(test_id, index=result1.index)
    imgcolumn = result1['image_name']
    result1.drop(labels=['image_name'], axis=1,inplace = True)
    result1.insert(0, 'image_name', imgcolumn)
    now = datetime.datetime.now()
    sub_file = 'submissions\submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()

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
    test_data, test_id = load_test()

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

def create_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(CHANNELS,ROWS, COLUMNS), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))
    
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))
    
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))
    
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(128, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))
    
#    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
#    model.add(Convolution2D(256, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
#    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(96, activation='relu',init='he_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(24, activation='relu',init='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    sgd = SGD(lr=1e-2, decay=1e-4, momentum=0.89, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model
    

def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_cross_validation_create_models(nfolds=10):
    # input image dimensions
    batch_size = 4
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

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=VERBOSEFLAG),
        ]
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=VERBOSEFLAG, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)
    return info_string, models


def run_cross_validation_process_test(info_string, models):
    batch_size = 4
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
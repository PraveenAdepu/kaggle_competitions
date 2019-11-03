# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.models import Model
from ipywidgets import IntProgress

from keras.utils import np_utils

import time
import glob
import math


from keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg19 import VGG19
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
               
inDir = 'C:/Users/SriPrav/Documents/R/43IEEE'
MODEL_WEIGHTS_FILE = inDir + '/Prav_01_Resnet.h5'


images_train = pd.read_csv(inDir + '/input/Prav_5folds_CVindices_v2.csv')

images_train.head()

y, rev_labels = pd.factorize(images_train['image_category'])

images_train['y'] = y


test = pd.read_csv(inDir+'/input/images_test_v2.csv')


test.head()


num_classes = 10  # This will reduce the max accuracy to about 0.75
CHANNELS = 3
ROWS = 224
COLUMNS = 224
nb_epoch = 20
VERBOSEFLAG = 1
batch_size  = 16
patience = 40
optim_type = 'Adam'
learning_rate = 1e-3

#path = 'C:\\Users\\SriPrav\\Documents\\R\\43IEEE\\input\\train_2\\HTC-1-M7\\(HTC-1-M7)1_0.jpg'

def get_im_cv2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    (h, w) = img.shape[:2]
#    center = (w / 2, h / 2)    
#    left = (w - 512)/2
#    top = (h + 512)/2
#    right = (w + 512)/2
#    bottom = (h - 512)/2
#    img = img[int(bottom):int(top), int(left):int(right)]
#    #img = centering_image(cv2.resize(img, dsize=(256,256))
##    img = mosaicing_CFA_Bayer(img)
##    img = np.reshape(img, (512,512,1))
#    #img = demosaicing_CFA_Bayer_Malvar2004(img)
##    resized = cv2.resize(img, (ROWS, COLUMNS), cv2.INTER_LINEAR)
##    if (img.shape[0] > img.shape[1]):
##        tile_size = (int(img.shape[1] * 256 / img.shape[0]), 256)
##    else:
##        tile_size = (256, int(img.shape[0] * 256 / img.shape[1]))
##    img = cv2.resize(img, tile_size)
##    img = centering_image(img)
#    img = cv2.resize(img, dsize=(ROWS, COLUMNS),interpolation=cv2.INTER_AREA)
#    img = np.reshape(img, (ROWS,COLUMNS,1))
    return img

#plt.imshow(img)
#img = normalize_image(img)
#img = np.array(img, dtype=np.uint8)
#img=img.transpose((0,1,2,3))
#img= img.astype('float32')

#def normalize_image_resnet50(x):
#    x = np.array(x, dtype=np.uint8)
##    x=x.transpose((0,3,1,2))
#    x= x.astype('float32')
#    x[:, 0, :, :] -= 103.939
#    x[:, 1, :, :] -= 116.779
#    x[:, 2, :, :] -= 123.68
#    return x
#    
#def InceptionV3_preprocess_input(x):
#    x = np.array(x, dtype=np.uint8)
#    x=x.transpose((0,3,1,2))
#    x= x.astype('float32')
#    x /= 255.
#    x -= 0.5
#    x *= 2.
#    return x
    
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
    
def load_train_frombatch(images_batch):
    X_train = []
    y = []
    y= np_utils.to_categorical(images_batch['y'],10)
    
#    start_time = time.time()

    for fl in images_batch.image_path.values:
#        print(fl)        
        img = get_im_cv2(fl)        
        X_train.append(img)
       
#    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y


def load_test_frombatch(images_batch):
    X_test = []
    for fl in images_batch.image_path.values:    
        img = get_im_cv2(fl)        
        X_test.append(img)
    return X_test

X_test = load_test_frombatch(test)
X_test = normalize_image(X_test)


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
    # Freeze layers not in classifier due to loading imagenet weights
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
#    x = GlobalAveragePooling2D()(x)
#    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
#    x = Dropout(0.2)(x)
#    x = BatchNormalization()(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=x)    
    # print(model.summary())
    return model


 
ModelName= 'Resnet_02'

def dlnet(i):

    print('Fold ', i , ' Processing')
    images_build = images_train[images_train['CVindices']!=i]
    images_valid = images_train[images_train['CVindices']==i]
    
    images_build = images_build.reset_index(drop=True)
    images_valid = images_valid.reset_index(drop=True)
    
    X_build = load_test_frombatch(images_build)
    X_build = normalize_image(X_build)
    y_build = np_utils.to_categorical(images_build['y'],10)
    
    X_valid = load_test_frombatch(images_valid)
    X_valid = normalize_image(X_valid)
    y_valid = np_utils.to_categorical(images_valid['y'],10)
    
    X_valid_data = load_test_frombatch(images_valid)
    X_valid_data = normalize_image(X_valid_data)
    
    pred_cv = np.zeros([images_valid.shape[0],10])    
    pred_test = np.zeros([X_test.shape[0],10])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
#        bag_cv = np.zeros([images_valid.shape[0],10])
        
#        model = CustomeNet(num_classes=num_classes)
        model = model_ResNet50(num_classes=num_classes)
        callbacks = [
                EarlyStopping(monitor='val_acc', patience=patience, verbose=VERBOSEFLAG, mode='max'),
                ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True, verbose=VERBOSEFLAG, mode='max'),
                        ]
        nb_epoch = 20
        learning_rate = 2e-4
        #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        if optim_type == 'SGD':
            optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            optim = Adam(lr=learning_rate,decay=0.0005)
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'])
#        model.summary()
           
        
        model.fit_generator( generator=batch_generator_X_build(images_build,X_build, y_build, batch_size, shuffle=True),
                                 #samples_per_epoch = len(build_index), 
                                 steps_per_epoch = math.ceil(len(images_build) / batch_size), #int(len(build_index)/float(batch_size)),
                                 nb_epoch = nb_epoch, 
                                 callbacks = callbacks,
                                 validation_data=batch_generator_X_valid(images_valid,X_valid,y_valid, batch_size, shuffle=True), 
                                 #nb_val_samples=len(valid_index), 
                                 validation_steps = math.ceil(len(images_valid) / batch_size), #int(len(valid_index)/float(batch_size)),
                                 max_q_size=10,
                                 verbose = VERBOSEFLAG 
                      )
        
        model.load_weights(MODEL_WEIGHTS_FILE)
        
#        nb_epoch = 40
#        initial_epoch = 20
#        learning_rate = 1e-4
#        optim = Adam(lr=learning_rate)
#        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
#        
#        model.fit_generator( generator=batch_generator_X_build(images_build,X_build, y_build, batch_size, shuffle=True),
#                                 #samples_per_epoch = len(build_index), 
#                                 steps_per_epoch = math.ceil(len(images_build) / batch_size), #int(len(build_index)/float(batch_size)),
#                                 initial_epoch = initial_epoch,
#                                 nb_epoch = nb_epoch, 
#                                 callbacks = callbacks,
#                                 validation_data=batch_generator_X_valid(images_valid,X_valid,y_valid, batch_size, shuffle=True), 
#                                 #nb_val_samples=len(valid_index), 
#                                 validation_steps = math.ceil(len(images_valid) / batch_size), #int(len(valid_index)/float(batch_size)),
#                                 max_q_size=10,
#                                 verbose = VERBOSEFLAG 
#                      )
#        
#        model.load_weights(MODEL_WEIGHTS_FILE)
        
        for layer in model.layers[-10:]:
            layer.trainable = True
            
        nb_epoch = 40
        initial_epoch = 20
        learning_rate = 1e-4
        optim = Adam(lr=learning_rate,decay=0.0005)
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
        
        model.fit_generator( generator=batch_generator_X_build(images_build,X_build, y_build, batch_size, shuffle=True),
                                 #samples_per_epoch = len(build_index), 
                                 steps_per_epoch = math.ceil(len(images_build) / batch_size), #int(len(build_index)/float(batch_size)),
                                 initial_epoch = initial_epoch,
                                 nb_epoch = nb_epoch, 
                                 callbacks = callbacks,
                                 validation_data=batch_generator_X_valid(images_valid,X_valid,y_valid, batch_size, shuffle=True), 
                                 #nb_val_samples=len(valid_index), 
                                 validation_steps = math.ceil(len(images_valid) / batch_size), #int(len(valid_index)/float(batch_size)),
                                 max_q_size=10,
                                 verbose = VERBOSEFLAG 
                      )
        model.load_weights(MODEL_WEIGHTS_FILE)
        
        for layer in model.layers:
            layer.trainable = True
            
        nb_epoch = 80
        initial_epoch = 40
        learning_rate = 1e-4
        optim = Adam(lr=learning_rate,decay=0.0005)
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
        
        model.fit_generator( generator=batch_generator_X_build(images_build,X_build, y_build, batch_size, shuffle=True),
                                 #samples_per_epoch = len(build_index), 
                                 steps_per_epoch = math.ceil(len(images_build) / batch_size), #int(len(build_index)/float(batch_size)),
                                 initial_epoch = initial_epoch,
                                 nb_epoch = nb_epoch, 
                                 callbacks = callbacks,
                                 validation_data=batch_generator_X_valid(images_valid,X_valid,y_valid, batch_size, shuffle=True), 
                                 #nb_val_samples=len(valid_index), 
                                 validation_steps = math.ceil(len(images_valid) / batch_size), #int(len(valid_index)/float(batch_size)),
                                 max_q_size=10,
                                 verbose = VERBOSEFLAG 
                      )
        
        model.load_weights(MODEL_WEIGHTS_FILE)
        
        nb_epoch = 100
        initial_epoch = 80
        learning_rate = 1e-5
        optim = Adam(lr=learning_rate,decay=0.0005)
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
        
        model.fit_generator( generator=batch_generator_X_build(images_build,X_build, y_build, batch_size, shuffle=True),
                                 #samples_per_epoch = len(build_index), 
                                 steps_per_epoch = math.ceil(len(images_build) / batch_size), #int(len(build_index)/float(batch_size)),
                                 initial_epoch = initial_epoch,
                                 nb_epoch = nb_epoch, 
                                 callbacks = callbacks,
                                 validation_data=batch_generator_X_valid(images_valid,X_valid,y_valid, batch_size, shuffle=True), 
                                 #nb_val_samples=len(valid_index), 
                                 validation_steps = math.ceil(len(images_valid) / batch_size), #int(len(valid_index)/float(batch_size)),
                                 max_q_size=10,
                                 verbose = VERBOSEFLAG 
                      )
        
        
        model.load_weights(MODEL_WEIGHTS_FILE)
        
        #bag_cv  += model.predict(X_valid_data, batch_size=batch_size, verbose=VERBOSEFLAG)        
        pred_cv += model.predict(X_valid_data, batch_size=batch_size, verbose=VERBOSEFLAG)        
        pred_test += model.predict(X_test, batch_size=batch_size, verbose=VERBOSEFLAG)

    pred_cv /= nbags
    

    pred_test/= nbags

    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = rev_labels
    pred_cv["fname"] = images_valid.image_name.values
    pred_cv["fname_patch"] = images_valid.image_name1.values
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
    del pred_cv
    del pred_test
    del model
    os.remove(MODEL_WEIGHTS_FILE)
    
    
nbags = 1
folds = 5
if __name__ == '__main__':
    for i in range(1, folds+1):
        dlnet(i)
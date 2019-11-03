# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
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

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import time

import os
import glob


np.random.seed(2017)
               
inDir = 'C:/Users/SriPrav/Documents/R/32Cdiscount'
MODEL_WEIGHTS_FILE = inDir + '/Prav_CNN05.h5'

images_train = pd.read_csv(inDir + '/images_train.csv')

images_train.head()

y, rev_labels = pd.factorize(images_train['image_category'])

images_train['y'] = y

images_train_index = images_train.index.get_values()
random_state = 2017

build_index, valid_index,y_build, y_valid = train_test_split(images_train_index, y, test_size=0.2, random_state=random_state, stratify=y)

print len(set(y_build)) # 5270
print len(set(y_valid)) # 5270

images_build = images_train.ix[list(build_index)] # 9897034
images_valid = images_train.ix[list(valid_index)] # 2474259

#import gc
#gc.collect()

num_classes = 5270  # This will reduce the max accuracy to about 0.75

CHANNELS = 3
ROWS = 128
COLUMNS = 128

nb_epoch = 3
VERBOSEFLAG = 1
batch_size  = 128
patience = 2

def get_im_cv2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    resized = cv2.resize(img, (ROWS, COLUMNS), cv2.INTER_LINEAR)
#    if (img.shape[0] > img.shape[1]):
#        tile_size = (int(img.shape[1] * 256 / img.shape[0]), 256)
#    else:
#        tile_size = (256, int(img.shape[0] * 256 / img.shape[1]))

    img = cv2.resize(img, dsize=(ROWS, COLUMNS),interpolation=cv2.INTER_AREA)
    return img

def normalize_image(x):
    x = np.array(x, dtype=np.uint8)
    x=x.transpose((0,3,1,2))
    x= x.astype('float32')
    x = x / 255
#    x -= 0.5
#    x *= 2.
    return x
    
def load_train_frombatch(images_batch):
    X_train = []
    y = []
    y= images_batch['y']
#    start_time = time.time()

    for fl in images_batch.image_path.values:
#        print(fl)        
        img = get_im_cv2(fl)        
        X_train.append(img)
       
#    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y

    
def batch_generator_train(images_build ,batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(len(images_build)/batch_size)
    counter = 0
    sample_index = images_build.index.get_values()
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        images_build_batch = images_build.ix[list(batch_index)]
        X_batch, y_batch = load_train_frombatch(images_build_batch)
        X_batch = normalize_image(X_batch)
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0 
            
def batch_generator_valid(images_valid ,batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(len(images_valid)/batch_size)
    counter = 0
    sample_index = images_valid.index.get_values()
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        images_valid_batch = images_valid.ix[list(batch_index)]
        X_batch, y_batch = load_train_frombatch(images_valid_batch)
        X_batch = normalize_image(X_batch)
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0 

 

model = Sequential()
model.add(Conv2D(16, 3,3, activation='relu', input_shape=(CHANNELS,ROWS, COLUMNS)))
model.add(Conv2D(16, 3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
model.add(Conv2D(32, 3,3, activation='relu'))
model.add(Conv2D(32, 3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
model.add(Conv2D(64, 3,3, activation='relu'))
model.add(Conv2D(64, 3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
model.add(Conv2D(128, 3,3, activation='relu'))
model.add(Conv2D(128, 3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
model.add(Flatten())
model.add(Dense(num_classes, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


opt = Adam(lr=0.01)

model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

  
callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=VERBOSEFLAG),
        ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=VERBOSEFLAG),
                ]
   
model.fit_generator( generator=batch_generator_train(images_build, batch_size, shuffle=False),
                             samples_per_epoch = len(build_index), nb_epoch = nb_epoch, callbacks = callbacks,
                             validation_data=batch_generator_valid(images_valid, batch_size, shuffle=False), 
                             nb_val_samples=len(valid_index), 
                             verbose = VERBOSEFLAG )

model.load_weights(MODEL_WEIGHTS_FILE)


num_cpus = 24
def imread(buf):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_ANYCOLOR)

def img2feat(im):
    x = cv2.resize(im, (ROWS, COLUMNS), interpolation=cv2.INTER_AREA)
    return np.float32(x) / 255

def load_image(pic, target, bar):
    picture = imread(pic)
    x = img2feat(picture)
    bar.update()
    
    return x, target
    
   
submission = pd.read_csv(inDir + '/input/sample_submission.csv', index_col='_id')

most_frequent_guess = 1000018296
submission['category_id'] = most_frequent_guess # Most frequent guess


num_images_test = 1768182  # We only have time for a few test images..

bar = tqdm_notebook(total=num_images_test * 2)
with open(inDir+'/input/test.bson', 'rb') as f, \
         concurrent.futures.ThreadPoolExecutor(num_cpus) as executor:

    data = bson.decode_file_iter(f)

    future_load = []
    
    for i,d in enumerate(data):
        if i >= num_images_test:
            break
        future_load.append(executor.submit(load_image, d['imgs'][0]['picture'], d['_id'], bar))

    print("Starting future processing")
    for future in concurrent.futures.as_completed(future_load):
        x, _id = future.result()
        
        y_cat = rev_labels[np.argmax(model.predict(x[None].transpose((0, 3, 1, 2)))[0])]
        if y_cat == -1:
            y_cat = most_frequent_guess

        bar.update()
        submission.loc[_id, 'category_id'] = y_cat
print('Finished')

submission.to_csv(inDir+'/submissions/Prav_CNN05.csv')



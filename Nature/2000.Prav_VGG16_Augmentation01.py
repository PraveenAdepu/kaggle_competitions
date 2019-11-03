# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:54:29 2017

@author: SriPrav
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import h5py
import cv2
import os
import glob
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adadelta, Adam, SGD
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, History
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from __future__ import division, print_function
from collections import Counter
%matplotlib inline

inDir = 'C:/Users/SriPrav/Documents/R/18Nature'
vgg16_weights = inDir + '/input/preModels/vgg16_weights.h5'

train_files = inDir + '/input/train_split/*/*.jpg'
validation_files = inDir + '/input/val_split/*/*.jpg'
train_generator_path =  inDir + '/input/train_split/'
validation_generator_path = inDir +  '/input/val_split/'
test_generator_path = inDir +  '/input/test/' # Move out test_stg1 folder from test folder
submission_file_path = inDir +  '/submissions/Prav_VGGAugmentation01_stg2.csv'

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1]

ROWS     = 224
COLUMNS  = 224
CHANNELS = 3
VERBOSEFLAG = 1


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
    


model = vgg_std16_model(ROWS, COLUMNS, CHANNELS)
model.pop(); model.pop(); model.pop(); model.pop(); model.pop();

for layer in model.layers:
    layer.trainable = False

model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(8, activation='softmax'))    
  
nb_train_samples = len(glob.glob(train_files))
nb_validation_samples = len(glob.glob(validation_files))
nb_epoch = 50
bath_size = 16
nb_test_samples = 12153 # test sample size
size=(224, 224)
classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_generator_path,
        target_size=size,
        batch_size=bath_size,
        shuffle = True,
        classes=classes,
        class_mode='categorical')

valid_datagen = ImageDataGenerator()

validation_generator = valid_datagen.flow_from_directory(
        validation_generator_path,
        target_size=size,
        batch_size=bath_size,
        shuffle = True,
        classes=classes,
        class_mode='categorical')

callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=0)]
model.compile(loss='categorical_crossentropy', optimizer="adadelta", \
              metrics=["accuracy"])

# fine-tune the model
hist = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=callbacks)

## summarize history for accuracy
#plt.figure(figsize=(15, 5))
#plt.subplot(1, 2, 1)
#plt.plot(hist.history['acc']); plt.plot(hist.history['val_acc']);
#plt.title('model accuracy'); plt.ylabel('accuracy');
#plt.xlabel('epoch'); plt.legend(['train', 'valid'], loc='upper left');
#
## summarize history for loss
#plt.subplot(1, 2, 2)
#plt.plot(hist.history['loss']); plt.plot(hist.history['val_loss']);
#plt.title('model loss'); plt.ylabel('loss');
#plt.xlabel('epoch'); plt.legend(['train', 'valid'], loc='upper left');
#plt.show()
#
#model.save_weights('C:/Users/SriPrav/Documents/R/18Nature/input/vgg16_aug_lessdrop.pkl')

test_aug = 5
test_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

for aug in range(test_aug):
    print('Predictions for Augmented -', aug)
    random_seed = np.random.random_integers(0, 100000)

    test_generator = test_datagen.flow_from_directory(
            test_generator_path,
            target_size=size,
            batch_size=bath_size,
            shuffle = False,
            seed = random_seed,
            classes = None,
            class_mode = None)

    test_image_list = test_generator.filenames
    if aug == 0:
        predictions = model.predict_generator(test_generator, nb_test_samples)
    else:
        predictions += model.predict_generator(test_generator, nb_test_samples)

predictions /= test_aug
# clip predictions
c = 0
preds = np.clip(predictions, c, 1-c)

print('Begin to write submission file ..')
f_submit = open(os.path.join(submission_file_path), 'w')
f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
for i, image_name in enumerate(test_generator.filenames):
    pred = ['%.6f' % p for p in preds[i, :]]
    if i%100 == 0:
        print(i, '/', 1000)
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

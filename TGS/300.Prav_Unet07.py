# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 10:27:59 2018

@author: SriPrav
"""

import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import cv2

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Input, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.optimizers import SGD, Adam

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, add, concatenate
from keras.models import Model
from keras.optimizers import RMSprop

from keras.losses import binary_crossentropy
import keras.backend as K

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pickle

# Set some parameters
inDir = 'C:/Users/SriPrav/Documents/R/51TGS'

im_width = 128
im_height = 128
border = 5
im_chan = 3 # Number of channels: first is original and second cumsum(axis=0)
n_features = 1 # Number of extra features, like depth
path_train = inDir+'/input/train/'
path_test = inDir+'/input/test/'

df_depths = pd.read_csv(inDir+'/input/depths.csv', index_col='id')
df_depths.head()

#df_depths = pd.read_csv(inDir+'/input/depths.csv')
#train_imgs = next(os.walk(path_train+"images"))[2]

df_depths = pd.read_csv(inDir+'/input/depths.csv', index_col='id')
df_depths.head()

train_imgs_CVindices = pd.read_csv(inDir+'/input/Prav_10folds_CVindices_CoverageClassStratified.csv')
del train_imgs_CVindices['CVindices']

train_imgs_CVindices.rename(columns={'CVindices_class': 'CVindices'}, inplace=True)

#train_imgs_CVindices['CVindices'] = train_imgs_CVindices['CVindices'] + 1

train_ids = train_imgs_CVindices["img"].tolist()

#train_ids = next(os.walk(path_train+"images"))[2]
test_ids = next(os.walk(path_test+"images"))[2]

df_depths.head()

df_depths.hist()

#img = load_img(inDir+'/input/train/masks/' + '1f1cc6b3a4' + '.png')
#img = np.array(img)

ids= ['1f1cc6b3a4','5b7c160d0d','6c40978ddf','7dfdf6eeb8','7e5a6e5013']
plt.figure(figsize=(30,15))
for j, img_name in enumerate(ids):
    q = j+1
    img = load_img(inDir+'/input/train/images/' + img_name + '.png', grayscale=True)
    img_mask = load_img(inDir+'/input/train/masks/' + img_name + '.png', grayscale=True)
    
    img = np.array(img)
    img_cumsum = (np.float32(img)-img.mean()).cumsum(axis=0)
    img_mask = np.array(img_mask)
    
    plt.subplot(1,3*(1+len(ids)),q*3-2)
    plt.imshow(img, cmap='seismic')
    plt.subplot(1,3*(1+len(ids)),q*3-1)
    plt.imshow(img_cumsum, cmap='seismic')
    plt.subplot(1,3*(1+len(ids)),q*3)
    plt.imshow(img_mask)
plt.show()




# Get and resize train images and masks
X = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.float32)
y = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.float32)
X_feat = np.zeros((len(train_ids), n_features), dtype=np.float32)

print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm_notebook(enumerate(train_ids), total=len(train_ids)):
    path = path_train
    
    # Depth
    X_feat[n] = df_depths.loc[id_.replace('.png', ''), 'z']
    
    # Load X
    img = load_img(path + '/images/' + id_, grayscale=False)
    x_img = img_to_array(img)
    x_img = resize(x_img, (128, 128, 3), mode='constant', preserve_range=True)
    
    # Create cumsum x
#    x_center_mean = x_img[border:-border, border:-border].mean()
#    x_csum = (np.float32(x_img)-x_center_mean).cumsum(axis=0)
#    x_csum -= x_csum[border:-border, border:-border].mean()
#    x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

    # Load Y
    mask = img_to_array(load_img(path + '/masks/' + id_, grayscale=True))
    mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

    # Save images
    X[n] = x_img / 255
#    X[n, ..., 1] = x_csum.squeeze()
    y[n] = mask / 255

print('Done!')

# Get and resize test images
X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.float32)
X_feat_test = np.zeros((len(test_ids), n_features), dtype=np.float32)
sizes_test = []

print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm_notebook(enumerate(test_ids), total=len(test_ids)):
    path = path_test
    
    # Depth
    X_feat_test[n] = df_depths.loc[id_.replace('.png', ''), 'z']
    
    # Load X
    img = load_img(path + '/images/' + id_, grayscale=False)
    x = img_to_array(img)
    sizes_test.append([x.shape[0], x.shape[1]])
    x = resize(x, (128, 128, 3), mode='constant', preserve_range=True)
    
#    # Create cumsum x
#    x_center_mean = x[border:-border, border:-border].mean()
#    x_csum = (np.float32(x)-x_center_mean).cumsum(axis=0)
#    x_csum -= x_csum[border:-border, border:-border].mean()
#    x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

    # Save images
    X_test[n] = x / 255
#    X_test[n, ..., 1] = x_csum.squeeze()

print('Done!')

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

# Build U-Net model
#def unet():
#    input_img = Input((im_height, im_width, im_chan), name='img')
#    input_features = Input((n_features, ), name='feat')
#    
#    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (input_img)
#    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
#    p1 = MaxPooling2D((2, 2)) (c1)
#    
#    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
#    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
#    p2 = MaxPooling2D((2, 2)) (c2)
#    
#    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
#    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
#    p3 = MaxPooling2D((2, 2)) (c3)
#    
#    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
#    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
#    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
#    
#    # Join features information in the depthest layer
#    f_repeat = RepeatVector(8*8)(input_features)
#    f_conv = Reshape((8, 8, n_features))(f_repeat)
#    p4_feat = concatenate([p4, f_conv], -1)
#    
#    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4_feat)
#    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)
#    
#    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
#    u6 = concatenate([u6, c4])
#    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
#    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)
#    
#    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
#    u7 = concatenate([u7, c3])
#    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
#    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)
#    
#    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
#    u8 = concatenate([u8, c2])
#    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
#    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)
#    
#    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
#    u9 = concatenate([u9, c1], axis=3)
#    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
#    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)
#    
#    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
#    
#    model = Model(inputs=[input_img, input_features], outputs=[outputs])
#    return model

#def unet():
#    start_neurons = 16
#    input_img = Input((im_height, im_width, im_chan), name='img')
#    # 128 -> 64
#    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_img)
#    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
#    pool1 = MaxPooling2D((2, 2))(conv1)
#    pool1 = Dropout(0.25)(pool1)
#
#    # 64 -> 32
#    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
#    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
#    pool2 = MaxPooling2D((2, 2))(conv2)
#    pool2 = Dropout(0.5)(pool2)
#
#    # 32 -> 16
#    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
#    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
#    pool3 = MaxPooling2D((2, 2))(conv3)
#    pool3 = Dropout(0.5)(pool3)
#
#    # 16 -> 8
#    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
#    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
#    pool4 = MaxPooling2D((2, 2))(conv4)
#    pool4 = Dropout(0.5)(pool4)
#
#    # Middle
#    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
#    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
#
#    # 8 -> 16
#    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
#    uconv4 = concatenate([deconv4, conv4])
#    uconv4 = Dropout(0.5)(uconv4)
#    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
#    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
#
#    # 16 -> 32
#    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
#    uconv3 = concatenate([deconv3, conv3])
#    uconv3 = Dropout(0.5)(uconv3)
#    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
#    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
#
#    # 32 -> 64
#    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
#    uconv2 = concatenate([deconv2, conv2])
#    uconv2 = Dropout(0.5)(uconv2)
#    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
#    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
#
#    # 64 -> 128
#    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
#    uconv1 = concatenate([deconv1, conv1])
#    uconv1 = Dropout(0.5)(uconv1)
#    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
#    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
#
#    uncov1 = Dropout(0.5)(uconv1)
#    outputs = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
#    
#    model = Model(inputs=input_img, outputs=[outputs])
#    return model




def encoder(x, filters=44, n_block=3, kernel_size=(3, 3), activation='relu'):
    skip = []
    for i in range(n_block):
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        skip.append(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x, skip


def bottleneck(x, filters_bottleneck, mode='cascade', depth=6,
               kernel_size=(3, 3), activation='relu'):
    dilated_layers = []
    if mode == 'cascade':  # used in the competition
        for i in range(depth):
            x = Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            dilated_layers.append(x)
        return add(dilated_layers)
    elif mode == 'parallel':  # Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            dilated_layers.append(
                Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            )
        return add(dilated_layers)


def decoder(x, skip, filters, n_block=3, kernel_size=(3, 3), activation='relu'):
    for i in reversed(range(n_block)):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = concatenate([skip[i], x])
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
    return x


def get_dilated_unet(
        input_shape=(128, 128, 3),
        mode='cascade',
        filters=16,
        n_block=3,
#        lr=0.0001,
#        loss=bce_dice_loss,
        n_class=1
):
    inputs = Input(input_shape)
    
    enc, skip = encoder(inputs, filters, n_block)
    bottle = bottleneck(enc, filters_bottleneck=filters * 2**n_block, mode=mode)
    dec = decoder(bottle, skip, filters, n_block)
    classify = Conv2D(n_class, (1, 1), activation='sigmoid')(dec)

    model = Model(inputs=inputs, outputs=classify)
#    model.compile(optimizer=RMSprop(lr), loss=loss, metrics=[dice_coef])

    return model

def unet_vgg16():
    start_neurons = 16
    input_img = Input((im_height, im_width, im_chan), name='img')
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=None, pooling=None, classes=1000)
#    base_model.summary()
    
        # => 128 * 64
    c1 = base_model.get_layer('block1_conv2').output
    p1 = base_model.get_layer('block1_pool').output
    # => 64 * 128
    c2 = base_model.get_layer('block2_conv2').output
    p2 = base_model.get_layer('block2_pool').output
    
    # => 32 * 256
    c3 = base_model.get_layer('block3_conv3').output
    p3 = base_model.get_layer('block3_pool').output
    
    # => 16 * 512
    c4 = base_model.get_layer('block4_conv3').output
    
#    input_img = Input((im_height, im_width, im_chan), name='img')
#    
#    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_img)
#    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
#    pool1 = MaxPooling2D((2, 2))(conv1)
#    pool1 = Dropout(0.25)(pool1)   
    
    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(c4)
    uconv3 = concatenate([deconv3, c3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(uconv3)
    #uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    
    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, c2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv2)
    #uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    
    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, c1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv1)
    #uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    
    #uncov1 = Dropout(0.5)(uconv1)
    outputs = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    model = Model(inputs=base_model.input, outputs=[outputs])
    return model

#model.summary()

train_datagen = ImageDataGenerator(        
                                  
                                  )

valid_datagen = ImageDataGenerator(
                                   
                                   )

test_datagen = ImageDataGenerator(        
                                 
                                 )

batch_size = 16
nb_epoch = 150
random_state = 2017
patience = 20
optim_type = 'Adam'
learning_rate = 1e-3
VERBOSEFLAG = 2
MODEL_WEIGHTS_FILE = inDir + '/Unet_CCN07_weights.h5'


i= 10

def train_nn(i):
    fileNo = i
    trainindex = train_imgs_CVindices[train_imgs_CVindices['CVindices'] != i].index.tolist()
    valindex   = train_imgs_CVindices[train_imgs_CVindices['CVindices'] == i].index.tolist()
    X_val_df = train_imgs_CVindices.iloc[valindex,:]
    X_build , X_valid,X_build_feat , X_valid_feat = X[trainindex,:], X[valindex,:], X_feat[trainindex,:], X_feat[valindex,:]
    y_build , y_valid = y[trainindex,:], y[valindex,:] 
    
    x_feat_mean = X_build_feat.mean(axis=0, keepdims=True)
    x_feat_std = X_build_feat.std(axis=0, keepdims=True)
    X_build_feat -= x_feat_mean
    X_build_feat /= x_feat_std
    
    X_valid_feat -= x_feat_mean
    X_valid_feat /= x_feat_std
    
    X_test_current = X_test.copy()
    X_feat_test_current = X_feat_test.copy()
    
    # Normalize X_test_feats
    X_feat_test_current -= x_feat_mean
    X_feat_test_current /= x_feat_std
    
    X_build = np.append(X_build, [np.fliplr(x) for x in X_build], axis=0)
    y_build = np.append(y_build, [np.fliplr(x) for x in y_build], axis=0)
    X_build_feat = np.concatenate((X_build_feat, X_build_feat), axis=0)
    
##    X_build = np.append(X_build, [np.fliplr(x) for x in X_build], axis=0)
#    X_build_lr = [np.fliplr(x) for x in X_build]
#    X_build_ud = [np.flipud(x) for x in X_build]
#    X_build = np.append(np.append(X_build,X_build_lr,axis=0),X_build_ud,axis=0)
#    
#    y_build_lr = [np.fliplr(x) for x in y_build]
#    y_build_ud = [np.flipud(x) for x in y_build]
#    y_build    = np.append(np.append(y_build, y_build_lr, axis=0),y_build_ud, axis=0)
#    X_build_feat = np.concatenate((X_build_feat, X_build_feat,X_build_feat), axis=0)
         
    print('Split train: ', len(X_build), len(y_build), len(X_build_feat)) 
    print('Split valid: ', len(X_valid), len(y_valid), len(X_valid_feat)) 
    print('Split test: ', len(X_test), len(X_feat_test))
    model = unet_vgg16()
    
#    model = get_dilated_unet(input_shape=(128, 128, 3),
#        mode='cascade',
#        filters=16,
#        n_block=4,
##        lr=0.0001,
##        loss=bce_dice_loss,
#        n_class=1)
#    model = unet() 
#    model.summary()
    
#    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou]) #, metrics=[mean_iou]) # The mean_iou metrics seens to leak train and test values...

    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[mean_iou])
    
    callbacks = [
                EarlyStopping(patience=patience, verbose=VERBOSEFLAG),
                ReduceLROnPlateau(patience=5, verbose=VERBOSEFLAG,factor=0.1,min_lr=1e-7),
                ModelCheckpoint(MODEL_WEIGHTS_FILE, verbose=1, save_best_only=True, save_weights_only=True)
                ]
#    callbacks = [
#    EarlyStopping(monitor='val_loss', patience=patience, verbose=VERBOSEFLAG),
#    ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=2),
#            ]

    results = model.fit(X_build, y_build, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks, verbose=2,
                        validation_data=(X_valid, y_valid))
    model.load_weights(MODEL_WEIGHTS_FILE)
    model.evaluate(X_valid, y_valid, verbose=1)
    
    preds_val = model.predict(X_valid, verbose=1)
    preds_test = model.predict(X_test_current, verbose=1)
    
    # Threshold predictions
#    preds_train_t = (preds_train > 0.5).astype(np.uint8)
#    preds_val_t = (preds_val > 0.5).astype(np.uint8)
#    preds_test_t = (preds_test > 0.5).astype(np.uint8)
    
    # Create list of upsampled test masks
    preds_test_upsampled = []
    for i in tnrange(len(preds_test)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), (101,101),
#                                           (sizes_test[i][0], sizes_test[i][1]), 
                                           mode='constant', preserve_range=True))
    preds_test_upsampled[0].shape
    
    pred_test_pkl = 'preds_test_unet07'+str(fileNo)+'.pkl'
    y_val_pkl = 'y_val_unet07'+str(fileNo)+'.pkl'
    pred_val_pkl = 'preds_val_unet07'+str(fileNo)+'.pkl'
    
    with open(pred_test_pkl, 'wb') as f:
        pickle.dump(preds_test_upsampled, f)
    with open(y_val_pkl, 'wb') as f:
        pickle.dump(y_valid, f)
    with open(pred_val_pkl, 'wb') as f:
        pickle.dump(preds_val, f)
    del model

#https://www.kaggle.com/takuok/keras-generator-starter-lb-0-326
    
#    model.fit_generator( train_datagen.flow( {'img': X_build, 'feat': X_build_feat}, y_build, batch_size = batch_size,shuffle=True),
#                         samples_per_epoch = len(X_build), nb_epoch = nb_epoch, callbacks = callbacks,
#                         validation_data=valid_datagen.flow({'img': X_valid, 'feat': X_valid_feat}, y_valid, batch_size=batch_size), 
#                         nb_val_samples=X_valid.shape[0], verbose = VERBOSEFLAG )

    
    #'categorical_crossentropy'dice_coef_loss , metrics=[dice_coef]
#    model.summary()
#        model.compile(loss='categorical_crossentropy', optimizer="adadelta", \
#              metrics=["accuracy"])
#    model.fit(train_data_224_3, train_target_224_3, batch_size=32, nb_epoch=20, shuffle=True,
#              validation_split=0.1,
#              callbacks = callbacks, verbose = VERBOSEFLAG)

#    min_loss = min(model.model['val_loss'])
#    print('Minimum loss for given fold: ', min_loss)
#    model.load_weights(MODEL_WEIGHTS_FILE)
        
#    mask_val = 'imgs_mask_val_Unet0'+str(i)+'.npy'
#    imgs_mask_val = model.predict(X_valid, verbose = True)
#    np.save(mask_val, imgs_mask_val)
#    
#    imgs_mask_test = model.predict(test_data_224_3, verbose = True)
#    
#    mask_test = 'imgs_mask_test_Unet0'+str(i)+'.npy'
#    np.save(mask_test, imgs_mask_test)
    
   
folds = 10

if __name__ == '__main__':      
    for i in range(1, folds+1):
        print("processing fold - ",i)
        train_nn(i)
        
#
## Split train and valid
#X_train, X_valid, X_feat_train, X_feat_valid, y_train, y_valid = train_test_split(X, X_feat, y, test_size=0.10, random_state=201801)
## Normalize X_feat
#x_feat_mean = X_feat_train.mean(axis=0, keepdims=True)
#x_feat_std = X_feat_train.std(axis=0, keepdims=True)
#X_feat_train -= x_feat_mean
#X_feat_train /= x_feat_std
#
#X_feat_valid -= x_feat_mean
#X_feat_valid /= x_feat_std
#
## Normalize X_test_feats
#X_feat_test -= x_feat_mean
#X_feat_test /= x_feat_std
#
### Check if training data looks all right
##ix = random.randint(0, len(X_train))
##
##has_mask = y_train[ix].max() > 0
##
##fig, ax = plt.subplots(1, 3, figsize=(20, 10))
##ax[0].imshow(X_train[ix, ..., 0], cmap='seismic', interpolation='bilinear')
##if has_mask:
##    ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
##ax[0].set_title('Seismic')
##
##ax[1].imshow(X_train[ix, ..., 1], cmap='seismic', interpolation='bilinear')
##if has_mask:
##    ax[1].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
##ax[1].set_title('Seismic cumsum')
##
##ax[2].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
##ax[2].set_title('Salt');
#
#
#
#
## Build U-Net model
#input_img = Input((im_height, im_width, im_chan), name='img')
#input_features = Input((n_features, ), name='feat')
#
#c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (input_img)
#c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
#p1 = MaxPooling2D((2, 2)) (c1)
#
#c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
#c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
#p2 = MaxPooling2D((2, 2)) (c2)
#
#c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
#c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
#p3 = MaxPooling2D((2, 2)) (c3)
#
#c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
#c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
#p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
#
## Join features information in the depthest layer
#f_repeat = RepeatVector(8*8)(input_features)
#f_conv = Reshape((8, 8, n_features))(f_repeat)
#p4_feat = concatenate([p4, f_conv], -1)
#
#c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4_feat)
#c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)
#
#u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
#u6 = concatenate([u6, c4])
#c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
#c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)
#
#u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
#u7 = concatenate([u7, c3])
#c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
#c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)
#
#u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
#u8 = concatenate([u8, c2])
#c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
#c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)
#
#u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
#u9 = concatenate([u9, c1], axis=3)
#c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
#c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)
#
#outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
#
#model = Model(inputs=[input_img, input_features], outputs=[outputs])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou]) #, metrics=[mean_iou]) # The mean_iou metrics seens to leak train and test values...
#model.summary()
#
#callbacks = [
#    EarlyStopping(patience=5, verbose=1),
#    ReduceLROnPlateau(patience=3, verbose=1),
#    ModelCheckpoint('model-tgs-salt-2.h5', verbose=1, save_best_only=True, save_weights_only=True)
#]
#
#results = model.fit({'img': X_train, 'feat': X_feat_train}, y_train, batch_size=16, epochs=50, callbacks=callbacks,
#                    validation_data=({'img': X_valid, 'feat': X_feat_valid}, y_valid))
#
#
#
#
## Load best model
#model.load_weights('model-tgs-salt-2.h5')
## Evaluate on validation set (this must be equals to the best log_loss)
#model.evaluate({'img': X_valid, 'feat': X_feat_valid}, y_valid, verbose=1)
#
## Predict on train, val and test
#preds_train = model.predict({'img': X_train, 'feat': X_feat_train}, verbose=1)
#preds_val = model.predict({'img': X_valid, 'feat': X_feat_valid}, verbose=1)
#preds_test = model.predict({'img': X_test, 'feat': X_feat_test}, verbose=1)
#
## Threshold predictions
#preds_train_t = (preds_train > 0.5).astype(np.uint8)
#preds_val_t = (preds_val > 0.5).astype(np.uint8)
#preds_test_t = (preds_test > 0.5).astype(np.uint8)
#
## Create list of upsampled test masks
#preds_test_upsampled = []
#for i in tnrange(len(preds_test)):
#    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
#                                       (sizes_test[i][0], sizes_test[i][1]), 
#                                       mode='constant', preserve_range=True))
#preds_test_upsampled[0].shape
#
#
#
#
#def plot_sample(X, y, preds):
#    ix = random.randint(0, len(X))
#
#    has_mask = y[ix].max() > 0
#
#    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
#    ax[0].imshow(X[ix, ..., 0], cmap='seismic')
#    if has_mask:
#        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
#    ax[0].set_title('Seismic')
#
#    ax[1].imshow(X[ix, ..., 1], cmap='seismic')
#    if has_mask:
#        ax[1].contour(y[ix].squeeze(), colors='k', levels=[0.5])
#    ax[1].set_title('Seismic cumsum')
#
#    ax[2].imshow(y[ix].squeeze())
#    ax[2].set_title('Salt')
#
#    ax[3].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
#    if has_mask:
#        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
#    ax[3].set_title('Salt Pred');
#    
## Check if training data looks all right
#plot_sample(X_train, y_train, preds_train)
#
## Check if valid data looks all right
#plot_sample(X_valid, y_valid, preds_val)   
#
## src: https://www.kaggle.com/aglotero/another-iou-metric
#def iou_metric(y_true_in, y_pred_in, print_table=False):
#    labels = y_true_in
#    y_pred = y_pred_in
#    
#    true_objects = 2
#    pred_objects = 2
#
#    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]
#
#    # Compute areas (needed for finding the union between all objects)
#    area_true = np.histogram(labels, bins = true_objects)[0]
#    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
#    area_true = np.expand_dims(area_true, -1)
#    area_pred = np.expand_dims(area_pred, 0)
#
#    # Compute union
#    union = area_true + area_pred - intersection
#
#    # Exclude background from the analysis
#    intersection = intersection[1:,1:]
#    union = union[1:,1:]
#    union[union == 0] = 1e-9
#
#    # Compute the intersection over union
#    iou = intersection / union
#
#    # Precision helper function
#    def precision_at(threshold, iou):
#        matches = iou > threshold
#        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
#        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
#        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
#        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
#        return tp, fp, fn
#
#    # Loop over IoU thresholds
#    prec = []
#    if print_table:
#        print("Thresh\tTP\tFP\tFN\tPrec.")
#    for t in np.arange(0.5, 1.0, 0.05):
#        tp, fp, fn = precision_at(t, iou)
#        if (tp + fp + fn) > 0:
#            p = tp / (tp + fp + fn)
#        else:
#            p = 0
#        if print_table:
#            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
#        prec.append(p)
#    
#    if print_table:
#        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
#    return np.mean(prec)
#
#def iou_metric_batch(y_true_in, y_pred_in):
#    batch_size = y_true_in.shape[0]
#    metric = []
#    for batch in range(batch_size):
#        value = iou_metric(y_true_in[batch], y_pred_in[batch])
#        metric.append(value)
#    return np.mean(metric)
#
#thres = np.linspace(0.25, 0.75, 20)
#thres_ioc = [iou_metric_batch(y_valid, np.int32(preds_val > t)) for t in tqdm_notebook(thres)]
#
#best_thres = thres[np.argmax(thres_ioc)]
#best_thres, max(thres_ioc)
#
#def RLenc(img, order='F', format=True):
#    """
#    img is binary mask image, shape (r,c)
#    order is down-then-right, i.e. Fortran
#    format determines if the order needs to be preformatted (according to submission rules) or not
#
#    returns run length as an array or string (if format is True)
#    """
#    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
#    runs = []  ## list of run lengths
#    r = 0  ## the current run length
#    pos = 1  ## count starts from 1 per WK
#    for c in bytes:
#        if (c == 0):
#            if r != 0:
#                runs.append((pos, r))
#                pos += r
#                r = 0
#            pos += 1
#        else:
#            r += 1
#
#    # if last run is unsaved (i.e. data ends with 1)
#    if r != 0:
#        runs.append((pos, r))
#        pos += r
#        r = 0
#
#    if format:
#        z = ''
#
#        for rr in runs:
#            z += '{} {} '.format(rr[0], rr[1])
#        return z[:-1]
#    else:
#        return runs
#
#pred_dict = {id_[:-4]:RLenc(np.round(preds_test_upsampled[i] > best_thres)) for i,id_ in tqdm_notebook(enumerate(test_ids))}
#sub = pd.DataFrame.from_dict(pred_dict,orient='index')
#sub.index.names = ['id']
#sub.columns = ['rle_mask']
#sub_file = inDir + "/submissions/Prav_Unet02.csv"
#sub.to_csv(sub_file)
#
##sub.to_csv('submission.csv')
#
#import pickle
#with open('preds_test_upsampled02.pkl', 'wb') as f:
#    pickle.dump(preds_test_upsampled, f)
#with open('y_valid02.pkl', 'wb') as f:
#    pickle.dump(y_valid, f)
#with open('preds_val02.pkl', 'wb') as f:
#    pickle.dump(preds_val, f)
##    
##with open('preds_test_upsampled01.pkl', 'rb') as f:
##    mynewlist = pickle.load(f)
##preds_test_upsampled[0]
##mynewlist[0]
##preds_test_upsampled_Ensemble = [0.5*x + 0.5*y for x, y in zip(preds_test_upsampled, preds_test_upsampled)] 
##
##
##preds_test_upsampled[0]
##preds_test_upsampled_Ensemble[0]
from keras.models import *
from keras.layers import *

import os
file_path = os.path.dirname( os.path.abspath(__file__) )


VGG_Weights_path = inDir+"/FeatureFiles/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

IMAGE_ORDERING = 'channels_first'


def VGGUnet( n_classes ,  input_height=128, input_width=128 , vgg_level=3):

	assert input_height%32 == 0
	assert input_width%32 == 0

	# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
	img_input = Input(shape=(input_height,input_width,1))

	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool' )(x)
	f1 = x
	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1' )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool' )(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1' )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2' )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool' )(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool' )(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool' )(x)
	f5 = x

	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense( 1000 , activation='softmax', name='predictions')(x)

	vgg  = Model(  img_input , x  )
#	vgg.load_weights(VGG_Weights_path)

	levels = [f1 , f2 , f3 , f4 , f5 ]

	o = f4

	o = ( ZeroPadding2D( (1,1)  ))(o)
	o = ( Conv2D(512, (3, 3), padding='valid'))(o)
	o = ( BatchNormalization())(o)

	o = (UpSampling2D( (2,2)))(o)
	o = ( concatenate([ o ,f3],axis=1 )  )
	o = ( ZeroPadding2D( (1,1)))(o)
	o = ( Conv2D( 256, (3, 3), padding='valid'))(o)
	o = ( BatchNormalization())(o)

	o = (UpSampling2D( (2,2)))(o)
	o = ( concatenate([o,f2],axis=1 ) )
	o = ( ZeroPadding2D((1,1) ))(o)
	o = ( Conv2D( 128 , (3, 3), padding='valid'  ) )(o)
	o = ( BatchNormalization())(o)

	o = (UpSampling2D( (2,2)))(o)
	o = ( concatenate([o,f1],axis=1 ) )
	o = ( ZeroPadding2D((1,1)   ))(o)
	o = ( Conv2D( 64 , (3, 3), padding='valid'  ))(o)
	o = ( BatchNormalization())(o)


	o =  Conv2D( n_classes , (3, 3) , padding='same' )( o )
	o_shape = Model(img_input , o ).output_shape
	outputHeight = o_shape[2]
	outputWidth = o_shape[3]

	o = (Reshape((  n_classes , outputHeight*outputWidth   )))(o)
	o = (Permute((2, 1)))(o)
	o = (Activation('softmax'))(o)
	model = Model( img_input , o )
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight
	return model

model = VGGUnet( n_classes=1 ,  input_height=128, input_width=128 , vgg_level=3)

def VGGUnet2( n_classes ,  input_height=416, input_width=608 , vgg_level=3):

	assert input_height%32 == 0
	assert input_width%32 == 0

	# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
	img_input = Input(shape=(3,input_height,input_width))

	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
	f1 = x
	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
	f5 = x

	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense( 1024 , activation='softmax', name='predictions')(x)

	vgg  = Model(  img_input , x  )
	vgg.load_weights(VGG_Weights_path)

	levels = [f1 , f2 , f3 , f4 , f5 ]

	o = f4

	o = ( ZeroPadding2D( (1,1) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	o = (UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([ o ,f3],axis=1 )  )
	o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
	o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	o = (UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([o,f2],axis=1 ) )
	o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format=IMAGE_ORDERING ) )(o)
	o = ( BatchNormalization())(o)

	o = (UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	# o = ( concatenate([o,f1],axis=1 ) )
	o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
	o = ( BatchNormalization())(o)


	o =  Conv2D( n_classes , (3, 3) , padding='same', data_format=IMAGE_ORDERING )( o )
	o_shape = Model(img_input , o ).output_shape
	outputHeight = o_shape[2]
	outputWidth = o_shape[3]

	o = (Reshape((  n_classes , outputHeight*outputWidth   )))(o)
	o = (Permute((2, 1)))(o)
	o = (Activation('softmax'))(o)
	model = Model( img_input , o )
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight



	return model


from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16


# The number of output labels
nb_labels = 1
# The dimensions of the input images
nb_rows = 224
nb_cols = 224
input_tensor = Input(shape=(128,128,3))
im_width = 128
im_height = 128
border = 5
im_chan = 3 

start_neurons = 16
def unet_vgg16():
    start_neurons = 16
    input_img = Input((im_height, im_width, im_chan), name='img')
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=None, pooling=None, classes=1000)
#    base_model.summary()
    
        # => 128 * 64
    c1 = base_model.get_layer('block1_conv2').output
    p1 = base_model.get_layer('block1_pool').output
    # => 64 * 128
    c2 = base_model.get_layer('block2_conv2').output
    p2 = base_model.get_layer('block2_pool').output
    
    # => 32 * 256
    c3 = base_model.get_layer('block3_conv3').output
    p3 = base_model.get_layer('block3_pool').output
    
    # => 16 * 512
    c4 = base_model.get_layer('block4_conv3').output
    
#    input_img = Input((im_height, im_width, im_chan), name='img')
#    
#    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_img)
#    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
#    pool1 = MaxPooling2D((2, 2))(conv1)
#    pool1 = Dropout(0.25)(pool1)   
    
    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(c4)
    uconv3 = concatenate([deconv3, c3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(uconv3)
    #uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    
    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, c2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv2)
    #uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    
    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, c1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv1)
    #uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    
    #uncov1 = Dropout(0.5)(uconv1)
    outputs = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    model = Model(inputs=base_model.input, outputs=[outputs])
    return model

model = unet_vgg16()
model.summary()
# Get final 32x32, 16x16, and 8x8 layers in the original
# ResNet by that layers's name.
x32 = base_model.get_layer('final_32').output
x16 = base_model.get_layer('final_16').output
x8 = base_model.get_layer('final_x8').output
# Compress each skip connection so it has nb_labels channels.
c32 = Convolution2D(nb_labels, (1, 1))(x32)
c16 = Convolution2D(nb_labels, (1, 1))(x16)
c8 = Convolution2D(nb_labels, (1, 1))(x8)
# Resize each compressed skip connection using bilinear interpolation.
# This operation isn't built into Keras, so we use a LambdaLayer
# which allows calling a Tensorflow operation.
def resize_bilinear(images):
    return tf.image.resize_bilinear(images, [nb_rows, nb_cols])
r32 = Lambda(resize_bilinear)(c32)
r16 = Lambda(resize_bilinear)(c16)
r8 = Lambda(resize_bilinear)(c8)
# Merge the three layers together using summation.
m = Add()([r32, r16, r8])
# Add softmax layer to get probabilities as output. We need to reshape
# and then un-reshape because Keras expects input to softmax to
# be 2D.
x = Reshape((nb_rows * nb_cols, nb_labels))(m)
x = Activation('softmax')(x)
x = Reshape((nb_rows, nb_cols, nb_labels))(x)
fcn_model = Model(input=input_tensor, output=x)

fcn_model.summary()
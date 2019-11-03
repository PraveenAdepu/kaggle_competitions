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


from sklearn.metrics import fbeta_score

import gc
from skimage.transform import resize
    
np.random.seed(2017)

inDir = 'C:/Users/SriPrav/Documents/R/29Carvana'

MODEL_WEIGHTS_FILE = inDir + '/Unet_CCN01_weights.h5'

train_file = inDir + "/CVSchema/Prav_CVindices_10folds.csv"
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


def ZF_UNET_128(dropout_val=0.05, batch_norm=True):
    from keras.models import Model
    from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.core import Dropout, Activation
    inputs = Input((1, 128, 128))
    conv1 = double_conv_layer(inputs, 32, dropout_val, batch_norm)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = double_conv_layer(pool1, 64, dropout_val, batch_norm)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = double_conv_layer(pool2, 128, dropout_val, batch_norm)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = double_conv_layer(pool3, 256, dropout_val, batch_norm)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = double_conv_layer(pool4, 512, dropout_val, batch_norm)

    up5 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = double_conv_layer(up5, 256, dropout_val, batch_norm)

    up6 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = double_conv_layer(up6, 128, dropout_val, batch_norm)

    up7 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = double_conv_layer(up7, 64, dropout_val, batch_norm)

    up8 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = double_conv_layer(up8, 32, 0, batch_norm)

    conv10 = Convolution2D(1, 1, 1, border_mode='same')(conv9)
#    conv10 = BatchNormalization(mode=0, axis=1)(conv10)
    conv10 = Activation('sigmoid')(conv10)

    model = Model(input=inputs, output=conv10)
    return model
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
#
#   

train_data_224_3 = train_data_224_3.astype('float32')
mean = np.mean(train_data_224_3)  # mean for data centering
std = np.std(train_data_224_3)  # std for data normalization

train_data_224_3 -= mean
train_data_224_3 /= std

train_target_224_3 = train_target_224_3.astype('float32')
train_target_224_3 /= 255.  # scale masks to [0, 1]

test_data_224_3 = test_data_224_3.astype('float32')
mean = np.mean(test_data_224_3)  # mean for data centering
std = np.std(test_data_224_3)  # std for data normalization

test_data_224_3 -= mean
test_data_224_3 /= std

train_datagen = ImageDataGenerator(        
                                  
                                  )

valid_datagen = ImageDataGenerator(
                                   
                                   )

test_datagen = ImageDataGenerator(        
                                 
                                 )

batch_size = 32
nb_epoch = 60
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
    model = ZF_UNET_128()    
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
    model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])#'categorical_crossentropy'
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
        
#    pred_cv = model.predict_generator(valid_datagen.flow(X_valid, batch_size=batch_size,shuffle=False),val_samples=X_valid.shape[0])
#    print('F2 Score : ',fbeta_score(y_valid, np.array(pred_cv) > 0.2, beta=2, average='samples'))
#    pred_cv = pd.DataFrame(pred_cv)
#    pred_cv.head()
#    pred_cv.columns = ["slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]
#    pred_cv["image_name"] = X_val_df.image_name.values
#    
#    sub_valfile = inDir + '/submissions/Prav.Unet_01.fold' + str(i) + '.csv'
#    pred_cv = pred_cv[["image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]]
#    pred_cv.to_csv(sub_valfile, index=False)
#    pred_test = model.predict_generator(test_datagen.flow(test_data_224_3, batch_size=batch_size,shuffle=False),val_samples=test_data_224_3.shape[0])
#    pred_test = pd.DataFrame(pred_test)
#    pred_test.columns = ["slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]
#    pred_test["image_name"] = test_all.image_name.values
#    pred_test = pred_test[["image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]]
#    sub_file = inDir + '/submissions/Prav.Unet_01.fold' + str(i) + '-test'+'.csv'
#    pred_test.to_csv(sub_file, index=False)  
    
    imgs_mask_val = model.predict_generator(valid_datagen.flow(X_valid, batch_size=batch_size,shuffle=False),val_samples=X_valid.shape[0])
    np.save('imgs_mask_val10_Unet_CNN01.npy', imgs_mask_val)
    


#    val_image = X_valid[0]
#    plt.imshow((val_image[0, :, :]).astype(np.uint8))
#    val_image = imgs_mask_val10[0]
#    val_image = prep(val_image)
#    val_image = (val_image[0, :, :] * 255.).astype(np.uint8)
#    plt.imshow(val_image)
#    plt.imshow(val_image)
#    val_image = (val_image[:, :, 0] * 255.).astype(np.uint8)
    #imgs_mask_test = model.predict(imgs_test, verbose=1)
    imgs_mask_test = model.predict_generator(test_datagen.flow(test_data_224_3, batch_size=batch_size,shuffle=False),val_samples=test_data_224_3.shape[0])
    np.save('imgs_mask_test10_Unet_CNN01.npy', imgs_mask_test)

#    print('-' * 30)
#    print('Saving predicted masks to files...')
#    print('-' * 30)
#    pred_dir = 'preds'
#    if not os.path.exists(pred_dir):
#        os.mkdir(pred_dir)
#    for image, image_id in zip(imgs_mask_test, imgs_id_test):
#        image = (image[:, :, 0] * 255.).astype(np.uint8)
#        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image) 



i = 10
train_nn(i)


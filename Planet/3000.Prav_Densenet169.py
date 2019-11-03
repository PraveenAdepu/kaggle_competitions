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
densenet169_weights = 'C:/Users/SriPrav/Documents/R' + '/imagenet_models/densenet169_weights_th.h5'
MODEL_WEIGHTS_FILE = inDir + '/Densenet169_CCN01_weights.h5'

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


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Convolution2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, 3, 3, name=conv_name_base+'_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


def densenet169_model(img_rows, img_cols, color_type=1, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, num_classes=None):
    '''
    DenseNet 169 Model for Keras
    Model Schema is based on 
    https://github.com/flyyufelix/DenseNet-Keras
    ImageNet Pretrained Weights 
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfN0d3T1F1MXg0NlU
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfSEc5UC1ROUFJdmM
    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      img_input = Input(shape=(224, 224, 3), name='data')
    else:
      concat_axis = 1
      img_input = Input(shape=(3, 224, 224), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6,12,32,32] # For DenseNet-169

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')
    weights_path = densenet169_weights
#    if K.image_dim_ordering() == 'th':
#      # Use pre-trained weights for Theano backend
#      weights_path = 'imagenet_models/densenet169_weights_th.h5'
#    else:
#      # Use pre-trained weights for Tensorflow backend
#      weights_path = 'imagenet_models/densenet169_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    x_newfc = Activation('sigmoid', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
#    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

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
nb_epoch = 3
random_state = 2017

    
def train_nn(i):
    trainindex = train_df[train_df['CVindices'] != i].index.tolist()
    valindex   = train_df[train_df['CVindices'] == i].index.tolist()
    X_val_df = train_df.iloc[valindex,:]
    X_build , X_valid = train_data_224_3[trainindex,:], train_data_224_3[valindex,:]
    y_build , y_valid = train_target_224_3[trainindex,:], train_target_224_3[valindex,:]  
    
    print('Split train: ', len(X_build), len(y_build))
    print('Split valid: ', len(X_valid), len(y_valid))
    model = densenet169_model(ROWS, COLUMNS, CHANNELS, num_classes=17)    
#        callbacks = [
#            EarlyStopping(monitor='val_loss', patience=3, verbose=VERBOSEFLAG),
#        ]
    callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=1)] 
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])#'categorical_crossentropy'
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
    
    sub_valfile = inDir + '/submissions/Prav.dense169.fold' + str(i) + '.csv'
    pred_cv = pred_cv[["image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]]
    pred_cv.to_csv(sub_valfile, index=False)
    pred_test = model.predict(test_data_224_3,verbose=1)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]
    pred_test["image_name"] = test_all.image_name.values
    pred_test = pred_test[["image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]]
    sub_file = inDir + '/submissions/Prav.dense169.fold' + str(i) + '-test'+'.csv'
    pred_test.to_csv(sub_file, index=False)   



i = 10
train_nn(i)


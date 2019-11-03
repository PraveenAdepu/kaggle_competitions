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
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint , ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.models import Model
from ipywidgets import IntProgress

from keras.utils import np_utils

from keras.models import Model
from keras import backend as K

import time
import glob
import math


#from keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg19 import VGG19
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
               
inDir = 'C:/Users/SriPrav/Documents/R/43IEEE'

images_train1 = pd.read_csv(inDir + '/input/Prav_5folds_CVindices_v5.csv')
images_train1 = images_train1[images_train1['image_patch_rank']<=40]

images_train2 = pd.read_csv(inDir + '/input/Prav_5folds_CVindices_v51.csv')
images_train2 = images_train2[images_train2['image_patch_rank']<=8]

images_train = pd.concat([images_train1, images_train2])
images_train = images_train.reset_index(drop=True)

images_train.groupby(['CVindices','image_category'])[['image_category']].size()

images_train.head()

images_train['image_category_lower'] = images_train['image_category'].str.lower()

images_train = images_train.sort_values('image_category_lower')

del images_train['image_category_lower']

y, rev_labels = pd.factorize(images_train['image_category'])

images_train['y'] = y


test = pd.read_csv(inDir+'/input/images_test_v2.csv')


test.head()


num_classes = 10  # This will reduce the max accuracy to about 0.75
CHANNELS = 3
ROWS = 224
COLUMNS = 224
nb_epoch = 20
VERBOSEFLAG = 2
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
#    x[:, :, :, 0] -= 103.939
#    x[:, :, :, 1] -= 116.779
#    x[:, :, :, 2] -= 123.68
    x = x / 255
    x -= 0.5
    x *= 2.
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

            

#def model_ResNet50(num_classes):
#    base_model = ResNet50(weights='imagenet')
#    # Freeze layers not in classifier due to loading imagenet weights
#    for layer in base_model.layers:
#        layer.trainable = False
#    x = base_model.output
##    x = GlobalAveragePooling2D()(x)
##    x = BatchNormalization()(x)
#    x = Dense(128, activation='relu')(x)
##    x = Dropout(0.2)(x)
##    x = BatchNormalization()(x)
#    x = Dense(num_classes, activation='softmax', name='predictions')(x)
#    model = Model(input=base_model.input, output=x)    
#    # print(model.summary())
#    return model

###########################################################################################################################
    
from __future__ import print_function
from __future__ import absolute_import

import warnings
from functools import partial

from keras.models import Model
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K

BASE_WEIGHT_URL = 'https://github.com/myutwo150/keras-inception-resnet-v2/releases/download/v0.1/'


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
    This function applies the "Inception" preprocessing which converts
    the RGB values from [0, 255] to [-1, 1]. Note that this preprocessing
    function is different from `imagenet_utils.preprocess_input()`.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Preprocessed array.
    """
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_Activation'`
            for the activation and `name + '_BatchNorm'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = _generate_layer_name('BatchNorm', prefix=name)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = _generate_layer_name('Activation', prefix=name)
        x = Activation(activation, name=ac_name)(x)
    return x


def _generate_layer_name(name, branch_idx=None, prefix=None):
    """Utility function for generating layer names.
    If `prefix` is `None`, returns `None` to use default automatic layer names.
    Otherwise, the returned layer name is:
        - PREFIX_NAME if `branch_idx` is not given.
        - PREFIX_Branch_0_NAME if e.g. `branch_idx=0` is given.
    # Arguments
        name: base layer name string, e.g. `'Concatenate'` or `'Conv2d_1x1'`.
        branch_idx: an `int`. If given, will add e.g. `'Branch_0'`
            after `prefix` and in front of `name` in order to identify
            layers in the same block but in different branches.
        prefix: string prefix that will be added in front of `name` to make
            all layer names unique (e.g. which block this layer belongs to).
    # Returns
        The layer name.
    """
    if prefix is None:
        return None
    if branch_idx is None:
        return '_'.join((prefix, name))
    return '_'.join((prefix, 'Branch', str(branch_idx), name))


def _inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block.
    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='Block35'`
        - Inception-ResNet-B: `block_type='Block17'`
        - Inception-ResNet-C: `block_type='Block8'`
    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals before adding
            them to the shortcut branch.
        block_type: `'Block35'`, `'Block17'` or `'Block8'`, determines
            the network structure in the residual branch.
        block_idx: used for generating layer names.
        activation: name of the activation function to use at the end
            of the block (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).
    # Returns
        Output tensor for the block.
    # Raises
        ValueError: if `block_type` is not one of `'Block35'`,
            `'Block17'` or `'Block8'`.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    if block_idx is None:
        prefix = None
    else:
        prefix = '_'.join((block_type, str(block_idx)))
    name_fmt = partial(_generate_layer_name, prefix=prefix)

    if block_type == 'Block35':
        branch_0 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 32, 3, name=name_fmt('Conv2d_0b_3x3', 1))
        branch_2 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 2))
        branch_2 = conv2d_bn(branch_2, 48, 3, name=name_fmt('Conv2d_0b_3x3', 2))
        branch_2 = conv2d_bn(branch_2, 64, 3, name=name_fmt('Conv2d_0c_3x3', 2))
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'Block17':
        branch_0 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 160, [1, 7], name=name_fmt('Conv2d_0b_1x7', 1))
        branch_1 = conv2d_bn(branch_1, 192, [7, 1], name=name_fmt('Conv2d_0c_7x1', 1))
        branches = [branch_0, branch_1]
    elif block_type == 'Block8':
        branch_0 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 224, [1, 3], name=name_fmt('Conv2d_0b_1x3', 1))
        branch_1 = conv2d_bn(branch_1, 256, [3, 1], name=name_fmt('Conv2d_0c_3x1', 1))
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "Block35", "Block17" or "Block8", '
                         'but got: ' + str(block_type))

    mixed = Concatenate(axis=channel_axis, name=name_fmt('Concatenate'))(branches)
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=name_fmt('Conv2d_1x1'))
    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=name_fmt('ScaleSum'))([x, up])
    if activation is not None:
        x = Activation(activation, name=name_fmt('Activation'))(x)
    return x


def InceptionResNetV2(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000,
                      dropout_keep_prob=0.8):
    """Instantiates the Inception-ResNet v2 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that when using TensorFlow, for best performance you should
    set `"image_data_format": "channels_last"` in your Keras config
    at `~/.keras/keras.json`.
    The model and the weights are compatible with both TensorFlow and Theano.
    The data format convention used by the model is the one specified in your
    Keras config file.
    Note that the default input image size for this model is 299x299, instead
    of 224x224 as in the VGG16 and ResNet models. Also, the input preprocessing
    function is different (i.e., do not use `imagenet_utils.preprocess_input()`
    with this model. Use `preprocess_input()` defined in this module instead).
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or `'imagenet'` (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional layer.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.
        dropout_keep_prob: dropout keep rate after pooling and before the
            classification layer, only to be specified if `include_top` is `True`.
    # Returns
        A Keras `Model` instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
#    input_shape = _obtain_input_shape(
#        input_shape,
#        default_size=299,
#        min_size=139,
#        data_format=K.image_data_format(),
#        require_flatten=False,
#        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid', name='Conv2d_1a_3x3')
    x = conv2d_bn(x, 32, 3, padding='valid', name='Conv2d_2a_3x3')
    x = conv2d_bn(x, 64, 3, name='Conv2d_2b_3x3')
    x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
    x = conv2d_bn(x, 80, 1, padding='valid', name='Conv2d_3b_1x1')
    x = conv2d_bn(x, 192, 3, padding='valid', name='Conv2d_4a_3x3')
    x = MaxPooling2D(3, strides=2, name='MaxPool_5a_3x3')(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    name_fmt = partial(_generate_layer_name, prefix='Mixed_5b')
    branch_0 = conv2d_bn(x, 96, 1, name=name_fmt('Conv2d_1x1', 0))
    branch_1 = conv2d_bn(x, 48, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 64, 5, name=name_fmt('Conv2d_0b_5x5', 1))
    branch_2 = conv2d_bn(x, 64, 1, name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 96, 3, name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2 = conv2d_bn(branch_2, 96, 3, name=name_fmt('Conv2d_0c_3x3', 2))
    branch_pool = AveragePooling2D(3,
                                   strides=1,
                                   padding='same',
                                   name=name_fmt('AvgPool_0a_3x3', 3))(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, name=name_fmt('Conv2d_0b_1x1', 3))
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_5b')(branches)

    # 10x Block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = _inception_resnet_block(x,
                                    scale=0.17,
                                    block_type='Block35',
                                    block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    name_fmt = partial(_generate_layer_name, prefix='Mixed_6a')
    branch_0 = conv2d_bn(x,
                         384,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 256, 3, name=name_fmt('Conv2d_0b_3x3', 1))
    branch_1 = conv2d_bn(branch_1,
                         384,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 1))
    branch_pool = MaxPooling2D(3,
                               strides=2,
                               padding='valid',
                               name=name_fmt('MaxPool_1a_3x3', 2))(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_6a')(branches)

    # 20x Block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = _inception_resnet_block(x,
                                    scale=0.1,
                                    block_type='Block17',
                                    block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    name_fmt = partial(_generate_layer_name, prefix='Mixed_7a')
    branch_0 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 0))
    branch_0 = conv2d_bn(branch_0,
                         384,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1,
                         288,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 1))
    branch_2 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 288, 3, name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2 = conv2d_bn(branch_2,
                         320,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 2))
    branch_pool = MaxPooling2D(3,
                               strides=2,
                               padding='valid',
                               name=name_fmt('MaxPool_1a_3x3', 3))(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_7a')(branches)

    # 10x Block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = _inception_resnet_block(x,
                                    scale=0.2,
                                    block_type='Block8',
                                    block_idx=block_idx)
    x = _inception_resnet_block(x,
                                scale=1.,
                                activation=None,
                                block_type='Block8',
                                block_idx=10)

    # Final convolution block
    x = conv2d_bn(x, 1536, 1, name='Conv2d_7b_1x1')

    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='AvgPool')(x)
        x = Dropout(1.0 - dropout_keep_prob, name='Dropout')(x)
        x = Dense(classes, name='Logits')(x)
        x = Activation('softmax', name='Predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='AvgPool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='MaxPool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = Model(inputs, x, name='inception_resnet_v2')

    # Load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_filename = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = get_file(weights_filename,
                                    BASE_WEIGHT_URL + weights_filename,
                                    cache_subdir='models',
                                    md5_hash='e693bd0210a403b3192acc6073ad2e96')
        else:
            weights_filename = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
            weights_path = get_file(weights_filename,
                                    BASE_WEIGHT_URL + weights_filename,
                                    cache_subdir='models',
                                    md5_hash='d19885ff4a710c122648d3b5c3b684e4')
        model.load_weights(weights_path)

    return model
##############################################################################################################################
def model_InceptionResNetv2(num_classes):
    base_model = InceptionResNetV2( include_top=False,weights='imagenet',input_shape=( ROWS, COLUMNS,CHANNELS),pooling='avg')

    # Freeze layers not in classifier due to loading imagenet weights
    for layer in base_model.layers:
        layer.trainable = True
#    for layer in base_model.layers[1:]:
#        layer.trainable = False
    x = base_model.output
#    x = Flatten(name='flatten')(x)
#    x = GlobalAveragePooling2D()(x)
#    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
#    x = Dropout(0.2)(x)
#    x = BatchNormalization()(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=x)    
    # print(model.summary())
    return model


 
ModelName= 'InceptionResnet_04'
i=1
def dlnet(i):
    MODEL_WEIGHTS_FILE = inDir + '/Prav_01_InceptionResnet04_1'+str(i)+'.h5'
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
        model = model_InceptionResNetv2(num_classes=num_classes)
#        callbacks = [
#                EarlyStopping(monitor='val_acc', patience=patience, verbose=VERBOSEFLAG, mode='max'),
#                ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True, verbose=VERBOSEFLAG, mode='max'),
#                        ]
        save_checkpoint = ModelCheckpoint(
            MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True, verbose=VERBOSEFLAG, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=1e-7, epsilon = 0.00001, verbose=1, mode='max')
        nb_epoch = 500
        learning_rate = 1e-3
        batch_size  = 16
        #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        if optim_type == 'SGD':
            optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            optim = Adam(lr=learning_rate)
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'])
#        model.summary()
           
        
        model.fit_generator( generator=batch_generator_X_build(images_build,X_build, y_build, batch_size, shuffle=True),
                                 #samples_per_epoch = len(build_index), 
                                 steps_per_epoch = math.ceil((len(images_build)/10) / batch_size), #int(len(build_index)/float(batch_size)),
                                 nb_epoch = nb_epoch, 
                                 callbacks = [save_checkpoint, reduce_lr],
                                 validation_data=batch_generator_X_valid(images_valid,X_valid,y_valid, batch_size, shuffle=True), 
                                 #nb_val_samples=len(valid_index), 
                                 validation_steps = math.ceil((len(images_valid)/4) / batch_size), #int(len(valid_index)/float(batch_size)),
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
        
#        for layer in model.layers:
#            layer.trainable = True
#            
#        batch_size  = 16   
#        nb_epoch = 15
#        initial_epoch = 5
#        learning_rate = 1e-4
#        optim = Adam(lr=learning_rate,decay=0.0005)
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
#        model.load_weights(MODEL_WEIGHTS_FILE)
        
#        for layer in model.layers:
#            layer.trainable = True
            
#        nb_epoch = 25
#        initial_epoch = 15
#        learning_rate = 1e-4
#        optim = Adam(lr=learning_rate,decay=0.0005)
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
        
#        nb_epoch = 30
#        initial_epoch = 25
#        learning_rate = 1e-5
#        optim = Adam(lr=learning_rate,decay=0.0005)
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
#        
#        model.load_weights(MODEL_WEIGHTS_FILE)
        
        #bag_cv  += model.predict(X_valid_data, batch_size=batch_size, verbose=VERBOSEFLAG)        
        pred_cv += model.predict(X_valid_data, batch_size=batch_size, verbose=VERBOSEFLAG)        
        pred_test += model.predict(X_test, batch_size=batch_size, verbose=VERBOSEFLAG)

    pred_cv /= nbags
    
    pred_test/= nbags

    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = rev_labels
    pred_cv["fname"] = images_valid.image_name.values
    pred_cv["fname_patch"] = images_valid.image_name_patch.values
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
#    os.remove(MODEL_WEIGHTS_FILE)
    
    
nbags = 1
folds = 5

dlnet(1)

if __name__ == '__main__':
    for i in range(2, folds+1):        
        dlnet(i)
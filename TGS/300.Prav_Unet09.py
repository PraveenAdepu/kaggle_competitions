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



#import tensorflow as tf
#
#print(tf.__version__)
#import keras as k
#print(k.__version__)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pickle



import six

from random import randint

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split

from skimage.transform import resize

#from keras import Model
from keras.models import load_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.preprocessing.image import load_img
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from tqdm import tqdm_notebook
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras import backend as K
#from keras.applications.imagenet_utils import _obtain_input_shape
from keras.regularizers import l2
from keras import optimizers


from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate,add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D

import tensorflow as tf

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import KFold

# Set some parameters
inDir = 'C:/Users/SriPrav/Documents/R/51TGS'

im_width = 101
im_height = 101
border = 5
im_chan = 1 # Number of channels: first is original and second cumsum(axis=0)
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
    img = load_img(path + '/images/' + id_, grayscale=True)
    x_img = img_to_array(img)
    x_img = resize(x_img, (im_width, im_height, 1), mode='constant', preserve_range=True)
    
    # Create cumsum x
    x_center_mean = x_img[border:-border, border:-border].mean()
    x_csum = (np.float32(x_img)-x_center_mean).cumsum(axis=0)
    x_csum -= x_csum[border:-border, border:-border].mean()
    x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

    # Load Y
    mask = img_to_array(load_img(path + '/masks/' + id_, grayscale=True))
    mask = resize(mask, (im_width, im_height, 1), mode='constant', preserve_range=True)

    # Save images
    X[n, ..., 0] = x_img.squeeze() / 255
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
    img = load_img(path + '/images/' + id_, grayscale=True)
    x = img_to_array(img)
    sizes_test.append([x.shape[0], x.shape[1]])
    x = resize(x, (im_width, im_height, 1), mode='constant', preserve_range=True)
    
    # Create cumsum x
    x_center_mean = x[border:-border, border:-border].mean()
    x_csum = (np.float32(x)-x_center_mean).cumsum(axis=0)
    x_csum -= x_csum[border:-border, border:-border].mean()
    x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

    # Save images
    X_test[n, ..., 0] = x.squeeze() / 255
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

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x

size = (3, 3)
def build_model(input_layer, start_neurons, DropoutRatio = 0.5):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, size, activation="relu", padding="same")(input_layer)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, size, activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, size, activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, size, activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, size, activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16, True)
    
    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, size, strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8, True)
    
    # 12 -> 25
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = Conv2DTranspose(start_neurons * 4, size, strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(DropoutRatio)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4, True)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, size, strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2, True)
    
    # 50 -> 101
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, size, strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1, True)
    
    #uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None)(uconv1)
    output_layer =  Activation('sigmoid')(output_layer_noActi)
    
    return output_layer

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
#         if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
#             metric.append(0)
#             continue
#         if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
#             metric.append(0)
#             continue
#         if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
#             metric.append(1)
#             continue
        
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.75], tf.float64)#change 0.5 >> 0.7

def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.75], tf.float64)#change 0.0 >> 0.5

def my_iou_metric_3(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.75], tf.float64)#change 0.0 >> 0.5

# code download from: https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
#        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        loss = tf.reduce_sum(tf.multiply(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), name="loss_non_void"))
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
#                   strict=True,
                   name="loss"
                   )
    return loss

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss




def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss

# https://github.com/raghakot/keras-resnet/blob/master/resnet.py
def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f

def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f

def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, block_fn, repetitions,input_tensor):
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)
        
        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
                
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(img_input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        model = Model(inputs=img_input, outputs=block)
        return model

    @staticmethod
    def build_resnet_34(input_shape,input_tensor):
        return ResnetBuilder.build(input_shape, basic_block, [3, 4, 6, 3],input_tensor)
    
def UResNet34(input_shape=(128, 128, 1), classes=1, decoder_filters=16, decoder_block_type='upsampling',
                       encoder_weights="imagenet", input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResnetBuilder.build_resnet_34(input_shape=input_shape,input_tensor=input_tensor)
    
    input_layer = backbone.input #input = backbone.input
    output_layer = build_model(input_layer, 16,0.5) #x
    model = Model(input_layer, output_layer)
    c = optimizers.adam(lr = 0.01)

    model.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])
    model.name = 'u-resnet34'

    return model

img_size_ori = 101
img_size_target = 101

epochs = 65
batch_size = 32

model1 = UResNet34( input_shape = (1,img_size_target,img_size_target))

model1.summary()

version = 9
basic_name = 'Unet_resnet_version_09'
save_model_name = basic_name + '.model'
submission_file = basic_name + '.csv'

model_checkpoint = ModelCheckpoint(save_model_name,monitor='my_iou_metric', 
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)

i = 10
fileNo = 10
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

   
print('Split train: ', len(X_build), len(y_build), len(X_build_feat)) 
print('Split valid: ', len(X_valid), len(y_valid), len(X_valid_feat)) 
print('Split test: ', len(X_test), len(X_feat_test))
    
history = model1.fit(X_build, y_build,
                    validation_data=[X_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[ model_checkpoint,reduce_lr], 
                    verbose=1)

model1 = load_model(save_model_name,custom_objects={'my_iou_metric': my_iou_metric})
# remove layter activation layer and use losvasz loss
input_x = model1.layers[0].input

output_layer = model1.layers[-1].input
model = Model(input_x, output_layer)
c = optimizers.adam(lr = 0.01)

# lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation  
# Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])

model.summary()
early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=20, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric_2', 
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)
#epochs = 50
#batch_size = 32

history = model.fit(X_build, y_build,
                    validation_data=[X_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[ model_checkpoint,reduce_lr,early_stopping], 
                    verbose=1)

model = load_model(save_model_name,custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                                   'lovasz_loss': lovasz_loss})
    
def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2

preds_val = predict_result(model,X_valid,img_size_target)
preds_test = predict_result(model,X_test_current,img_size_target)

pred_test_pkl = 'preds_test_unet09'+str(fileNo)+'.pkl'
y_val_pkl = 'y_val_unet09'+str(fileNo)+'.pkl'
pred_val_pkl = 'preds_val_unet09'+str(fileNo)+'.pkl'
    
with open(pred_test_pkl, 'wb') as f:
    pickle.dump(preds_test, f)
with open(y_val_pkl, 'wb') as f:
    pickle.dump(y_valid, f)
with open(pred_val_pkl, 'wb') as f:
    pickle.dump(preds_val, f)
#    del model
    
    
#####################################################################################################################################
#####################################################################################################################################
    
#####################################################################################################################################
#####################################################################################################################################

    
train_datagen = ImageDataGenerator(        
                                  
                                  )

valid_datagen = ImageDataGenerator(
                                   
                                   )

test_datagen = ImageDataGenerator(        
                                 
                                 )

batch_size = 16
nb_epoch = 50
random_state = 2017
patience = 10
optim_type = 'Adam'
learning_rate = 1e-3
VERBOSEFLAG = 2
MODEL_WEIGHTS_FILE = inDir + '/Unet_CCN05_weights.h5'


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

       
    print('Split train: ', len(X_build), len(y_build), len(X_build_feat)) 
    print('Split valid: ', len(X_valid), len(y_valid), len(X_valid_feat)) 
    print('Split test: ', len(X_test), len(X_feat_test))
    model = unet() 
    
#    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou]) #, metrics=[mean_iou]) # The mean_iou metrics seens to leak train and test values...

    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[mean_iou])
    
    callbacks = [
                EarlyStopping(patience=patience, verbose=VERBOSEFLAG),
                ReduceLROnPlateau(patience=5, verbose=VERBOSEFLAG,factor=0.1,min_lr=1e-5),
                ModelCheckpoint(MODEL_WEIGHTS_FILE, verbose=1, save_best_only=True, save_weights_only=True)
                ]
#    callbacks = [
#    EarlyStopping(monitor='val_loss', patience=patience, verbose=VERBOSEFLAG),
#    ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=2),
#            ]

    results = model.fit({'img': X_build}, y_build, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks, verbose=2,
                        validation_data=({'img': X_valid}, y_valid))
    model.load_weights(MODEL_WEIGHTS_FILE)
    model.evaluate({'img': X_valid}, y_valid, verbose=1)
    
    preds_val = model.predict({'img': X_valid}, verbose=1)
    preds_test = model.predict({'img': X_test_current}, verbose=1)
    
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
    
    pred_test_pkl = 'preds_test_unet06'+str(fileNo)+'.pkl'
    y_val_pkl = 'y_val_unet06'+str(fileNo)+'.pkl'
    pred_val_pkl = 'preds_val_unet06'+str(fileNo)+'.pkl'
    
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
        

# Split train and valid
X_train, X_valid, X_feat_train, X_feat_valid, y_train, y_valid = train_test_split(X, X_feat, y, test_size=0.10, random_state=201801)
# Normalize X_feat
x_feat_mean = X_feat_train.mean(axis=0, keepdims=True)
x_feat_std = X_feat_train.std(axis=0, keepdims=True)
X_feat_train -= x_feat_mean
X_feat_train /= x_feat_std

X_feat_valid -= x_feat_mean
X_feat_valid /= x_feat_std

# Normalize X_test_feats
X_feat_test -= x_feat_mean
X_feat_test /= x_feat_std

## Check if training data looks all right
#ix = random.randint(0, len(X_train))
#
#has_mask = y_train[ix].max() > 0
#
#fig, ax = plt.subplots(1, 3, figsize=(20, 10))
#ax[0].imshow(X_train[ix, ..., 0], cmap='seismic', interpolation='bilinear')
#if has_mask:
#    ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
#ax[0].set_title('Seismic')
#
#ax[1].imshow(X_train[ix, ..., 1], cmap='seismic', interpolation='bilinear')
#if has_mask:
#    ax[1].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
#ax[1].set_title('Seismic cumsum')
#
#ax[2].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
#ax[2].set_title('Salt');




# Build U-Net model
input_img = Input((im_height, im_width, im_chan), name='img')
input_features = Input((n_features, ), name='feat')

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (input_img)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

# Join features information in the depthest layer
f_repeat = RepeatVector(8*8)(input_features)
f_conv = Reshape((8, 8, n_features))(f_repeat)
p4_feat = concatenate([p4, f_conv], -1)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4_feat)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[input_img, input_features], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou]) #, metrics=[mean_iou]) # The mean_iou metrics seens to leak train and test values...
model.summary()

callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint('model-tgs-salt-2.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

results = model.fit({'img': X_train, 'feat': X_feat_train}, y_train, batch_size=16, epochs=50, callbacks=callbacks,
                    validation_data=({'img': X_valid, 'feat': X_feat_valid}, y_valid))




# Load best model
model.load_weights('model-tgs-salt-2.h5')
# Evaluate on validation set (this must be equals to the best log_loss)
model.evaluate({'img': X_valid, 'feat': X_feat_valid}, y_valid, verbose=1)

# Predict on train, val and test
preds_train = model.predict({'img': X_train, 'feat': X_feat_train}, verbose=1)
preds_val = model.predict({'img': X_valid, 'feat': X_feat_valid}, verbose=1)
preds_test = model.predict({'img': X_test, 'feat': X_feat_test}, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in tnrange(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))
preds_test_upsampled[0].shape




def plot_sample(X, y, preds):
    ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='seismic')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Seismic')

    ax[1].imshow(X[ix, ..., 1], cmap='seismic')
    if has_mask:
        ax[1].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[1].set_title('Seismic cumsum')

    ax[2].imshow(y[ix].squeeze())
    ax[2].set_title('Salt')

    ax[3].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Salt Pred');
    
# Check if training data looks all right
plot_sample(X_train, y_train, preds_train)

# Check if valid data looks all right
plot_sample(X_valid, y_valid, preds_val)   

# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

thres = np.linspace(0.25, 0.75, 20)
thres_ioc = [iou_metric_batch(y_valid, np.int32(preds_val > t)) for t in tqdm_notebook(thres)]

best_thres = thres[np.argmax(thres_ioc)]
best_thres, max(thres_ioc)

def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

pred_dict = {id_[:-4]:RLenc(np.round(preds_test_upsampled[i] > best_thres)) for i,id_ in tqdm_notebook(enumerate(test_ids))}
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub_file = inDir + "/submissions/Prav_Unet02.csv"
sub.to_csv(sub_file)

#sub.to_csv('submission.csv')

import pickle
with open('preds_test_upsampled02.pkl', 'wb') as f:
    pickle.dump(preds_test_upsampled, f)
with open('y_valid02.pkl', 'wb') as f:
    pickle.dump(y_valid, f)
with open('preds_val02.pkl', 'wb') as f:
    pickle.dump(preds_val, f)
#    
#with open('preds_test_upsampled01.pkl', 'rb') as f:
#    mynewlist = pickle.load(f)
#preds_test_upsampled[0]
#mynewlist[0]
#preds_test_upsampled_Ensemble = [0.5*x + 0.5*y for x, y in zip(preds_test_upsampled, preds_test_upsampled)] 
#
#
#preds_test_upsampled[0]
#preds_test_upsampled_Ensemble[0]

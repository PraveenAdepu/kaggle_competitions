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


from keras.utils import np_utils

import time
import glob
import math

from tqdm import tqdm
from PIL import Image
from io import BytesIO
import jpeg4py as jpeg
import random

import warnings
warnings.filterwarnings("ignore")
from sklearn.utils import shuffle

num_classes = 10  # This will reduce the max accuracy to about 0.75
CHANNELS = 3
ROWS = 224
COLUMNS = 224
nb_epoch = 20
VERBOSEFLAG = 2
batch_size  = 16
patience = 10
optim_type = 'Adam'
learning_rate = 1e-3
crop_size = 224

MANIPULATIONS = ['gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0']

def get_crop(img, crop_size, random_crop=False):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    half_crop = crop_size // 2
    pad_x = max(0, crop_size - img.shape[1])
    pad_y = max(0, crop_size - img.shape[0])
    if (pad_x > 0) or (pad_y > 0):
        img = np.pad(img, ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2), (0,0)), mode='wrap')
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    if random_crop:
        freedom_x, freedom_y = img.shape[1] - crop_size, img.shape[0] - crop_size
        if freedom_x > 0:
            center_x += np.random.randint(math.ceil(-freedom_x/2), freedom_x - math.floor(freedom_x/2) )
        if freedom_y > 0:
            center_y += np.random.randint(math.ceil(-freedom_y/2), freedom_y - math.floor(freedom_y/2) )

    return img[center_y - half_crop : center_y + crop_size - half_crop, center_x - half_crop : center_x + crop_size - half_crop]

def random_manipulation(img, manipulation=None):

    if manipulation == None:
        manipulation = random.choice(MANIPULATIONS)

    if manipulation.startswith('jpg'):
        quality = int(manipulation[3:])
        out = BytesIO()
        im = Image.fromarray(img)
        im.save(out, format='jpeg', quality=quality)
        im_decoded = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()
        del out
        del im
    elif manipulation.startswith('gamma'):
        gamma = float(manipulation[5:])
        # alternatively use skimage.exposure.adjust_gamma
        # img = skimage.exposure.adjust_gamma(img, gamma)
        im_decoded = np.uint8(cv2.pow(img / 255., gamma)*255.)
    elif manipulation.startswith('bicubic'):
        scale = float(manipulation[7:])
        im_decoded = cv2.resize(img,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    else:
        assert False
    return im_decoded

def get_im_cv2(path,use_manipulation=True,crop_size=224,random_crop=True):
    img = cv2.imread(path)
    if use_manipulation and np.random.rand() < 0.35:
        img = random_manipulation(img, manipulation=None)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = get_crop(img,crop_size, random_crop=True)
    return img

   
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
    
def load_train_frombatch(fl):
    X_train = []      
    img = get_im_cv2(fl,use_manipulation=True,crop_size=224,random_crop=True)        
    X_train.append(img)

    return X_train

    
def save_image_numpy_array(fl):   
    #print(fl)
    X_build = load_train_frombatch(fl)
    return X_build


    


    
    

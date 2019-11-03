# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 14:02:27 2018

@author: SriPrav
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:32:27 2018

@author: SriPrav
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
import scipy.misc

inDir = 'C:/Users/SriPrav/Documents/R/43IEEE'

images_train = pd.read_csv(inDir + '/input/Prav_5folds_CVindices.csv')

def RowWiseOperation(x):
    x1 = x.replace('\\','\\\\')
    return x1

images_train['image_path'] = images_train['image_path'].apply(RowWiseOperation)


def get_im_cv2_centering(path, new_height=896, new_width=896):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)    
    left = (w - new_width)/2
    top = (h + new_height)/2
    right = (w + new_width)/2
    bottom = (h - new_height)/2
    img = img[int(bottom):int(top), int(left):int(right)]

    return img

def image_patches(img, M=224, N=224):
    img = np.array(img, dtype=np.uint8)
    tiles = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0, img.shape[1],N)]
    tiles = np.stack(tiles, axis=0)
    return tiles


images_train = images_train[images_train['image_name']!='(MotoNex6)8.jpg']
images_train_Non_Sony = images_train[images_train['image_category']!='Sony-NEX-7']
images_train_Sony = images_train[images_train['image_category']=='Sony-NEX-7']



def load_test_frombatch(images_batch):   
    for fl in images_batch.image_path.values:    
        img = get_im_cv2_centering(fl, new_height=896, new_width=896)  
        current_image_patches = image_patches(img,224,224)
        new_fl = fl.replace('train','train_3')
        for i in range(0,16):
            new_path = new_fl.replace('.jpg','_{}.jpg'.format(i))
            scipy.misc.imsave(new_path, current_image_patches[i])
            
load_test_frombatch(images_train_Non_Sony)        

def load_test_frombatch(images_batch):   
    for fl in images_batch.image_path.values:    
        img = get_im_cv2_centering(fl, new_height=896, new_width=896)  
        current_image_patches = image_patches(img,224,224)
        new_fl = fl.replace('train','train_3')
        for i in range(0,16):
            new_path = new_fl.replace('.JPG','_{}.JPG'.format(i))
            scipy.misc.imsave(new_path, current_image_patches[i])
            
load_test_frombatch(images_train_Sony) 


images_train = pd.read_csv(inDir + '/input/Prav_5folds_CVindices_v2.csv')

def train_replace(x):
    return x.replace('train_2','train_3')
images_train['image_path'] = images_train['image_path'].apply(train_replace)

images_train1 = images_train.copy()
images_train2 = images_train.copy()
images_train3 = images_train.copy()
images_train4 = images_train.copy()

def train_replace(x):
    x = x.replace('_0.','_4.')
    x = x.replace('_1.','_5.')
    x = x.replace('_2.','_6.')
    x = x.replace('_3.','_7.')
    return x

images_train2['image_path'] = images_train2['image_path'].apply(train_replace)

def train_replace(x):
    x = x.replace('_0.','_8.')
    x = x.replace('_1.','_9.')
    x = x.replace('_2.','_10.')
    x = x.replace('_3.','_11.')
    return x

images_train3['image_path'] = images_train3['image_path'].apply(train_replace)

def train_replace(x):
    x = x.replace('_0.','_12.')
    x = x.replace('_1.','_13.')
    x = x.replace('_2.','_14.')
    x = x.replace('_3.','_15.')
    return x

images_train4['image_path'] = images_train4['image_path'].apply(train_replace)

images_train_v3 = pd.concat([images_train1, images_train2,images_train3,images_train4] )

images_train_v3 = images_train_v3.reset_index(drop=True)

images_train_v3.groupby(['CVindices','image_category'])[['image_category']].size()

images_train_v3.to_csv(inDir+"/input/train_images_v3.csv", index=False)
images_train_v3.to_csv(inDir+"/input/Prav_5folds_CVindices_v3.csv", index=False)




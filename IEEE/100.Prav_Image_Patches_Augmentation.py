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

def train_replace(x):
    return x.replace('train','train_0')

images_train['image_path'] = images_train['image_path'].apply(train_replace)

def get_im_cv2_centering(path, new_height=448, new_width=448):
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

def load_test_frombatch(images_batch):   
    for fl in images_batch.image_path.values:    
        img = get_im_cv2_centering(fl, new_height=448, new_width=448)  
        current_image_patches = image_patches(img,224,224)
        new_fl = fl.replace('train_0','train_22')
        for i in range(0,4):
            new_path = new_fl.replace('.jpg','_{}.jpg'.format(i))
            scipy.misc.imsave(new_path, current_image_patches[i])
            
load_test_frombatch(images_train)           

images_train_Sony = images_train[images_train['image_category']=='Sony-NEX-7']

def load_test_frombatch(images_batch):   
    for fl in images_batch.image_path.values:    
        img = get_im_cv2_centering(fl, new_height=448, new_width=448)  
        current_image_patches = image_patches(img,224,224)
        new_fl = fl.replace('train_0','train_22')
        for i in range(0,4):
            new_path = new_fl.replace('.JPG','_{}.JPG'.format(i))
            scipy.misc.imsave(new_path, current_image_patches[i])
            
load_test_frombatch(images_train_Sony) 

images_train_v2 = pd.read_csv(inDir + '/input/Prav_5folds_CVindices_v2.csv')

def train_replace(x):
    return x.replace('train_2','train_22')

images_train_v2['image_path'] = images_train_v2['image_path'].apply(train_replace)

images_train_v2.to_csv(inDir+"/input/Prav_5folds_AugmentationCVindices_v2.csv", index=False)

images_train_v2 = pd.read_csv(inDir + '/input/Prav_5folds_CVindices_v2.csv')
images_train_Augmentation_v2 = pd.read_csv(inDir + '/input/Prav_5folds_AugmentationCVindices_v2.csv')

trainingSet = pd.read_csv(inDir + '/input/Prav_5folds_CVindices_Augmentationindices.csv')

trainingSet = trainingSet[['image_name','Augmentationindices']]

images_train_Augmentation_v2 = pd.merge(images_train_Augmentation_v2, trainingSet, on='image_name', how='left')

images_train_v2['Augmentationindices'] = 0

images_train_v2_combine = pd.concat([images_train_v2, images_train_Augmentation_v2] )

images_train_v2_combine = images_train_v2_combine.reset_index(drop=True)

images_train_v2_combine.groupby(['CVindices','image_category'])[['image_category']].size()

images_train_v2_combine.to_csv(inDir+"/input/Prav_5folds_CVindices_includingAugmentationPatches_v2.csv", index=False)


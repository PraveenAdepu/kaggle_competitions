# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 21:26:31 2018

@author: SriPrav
"""

import os

import numpy as np
random_state = 2017
np.random.seed(random_state)

from sklearn.model_selection import StratifiedKFold

import pandas as pd
import io
import bson
import cv2
import matplotlib.pyplot as plt
import scipy.misc

inDir = 'C:/Users/SriPrav/Documents/R/43IEEE'

images_train = pd.read_csv(inDir + '/input/Prav_5folds_CVindices.csv')

images_train.head()

def RowWiseOperation(x):
    x1 = x.replace('\\','\\\\')
    return x1

images_train['image_path'] = images_train['image_path'].apply(RowWiseOperation)

train = pd.read_csv(inDir + '/input/Prav_5folds_CVindices.csv')

train.groupby(['image_category']).size()

trainfoldSource = train[['image_path','image_category']]

folds = 8
skf = StratifiedKFold(n_splits=folds, random_state = random_state, shuffle = True)

skf.get_n_splits(trainfoldSource, trainfoldSource['image_category'])

print(skf) 

Prav_CVindices = pd.DataFrame(columns=['index','CVindices'])

count = 1
for train_index, test_index in skf.split(trainfoldSource, trainfoldSource['image_category']):           
       df_index = pd.DataFrame(test_index,columns=['index']) 
       df_index['CVindices'] = count
       Prav_CVindices = Prav_CVindices.append(df_index)       
       count+=1
       
Prav_CVindices.set_index('index', inplace=True)

trainfoldSource = pd.merge(trainfoldSource, Prav_CVindices, left_index=True, right_index=True)

trainfoldSource.groupby(['CVindices','image_category'])[['image_category']].size()

del trainfoldSource['image_category']

trainfoldSource.columns = ['image_path', 'Augmentationindices']

trainingSet = pd.merge(train, trainfoldSource, on='image_path', how='left')

trainingSet.to_csv(inDir+"/input/Prav_5folds_CVindices_Augmentationindices.csv", index=False)

trainingSet = pd.read_csv(inDir + '/input/Prav_5folds_CVindices_Augmentationindices.csv')


def RowWiseOperation(x):
    x1 = x.replace('\\','\\\\')
    return x1

trainingSet['image_path'] = trainingSet['image_path'].apply(RowWiseOperation)
trainingSet['image_augmentation_path'] = trainingSet['image_path']

def train_replace(x):
    return x.replace('train','train_0')

trainingSet['image_augmentation_path'] = trainingSet['image_augmentation_path'].apply(train_replace)

def get_im_cv2(path):
    img = cv2.imread(path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

    return img

def get_im_centering(img, new_height=512, new_width=512):
   
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)    
    left = (w - new_width)/2
    top = (h + new_height)/2
    right = (w + new_width)/2
    bottom = (h - new_height)/2
    img = img[int(bottom):int(top), int(left):int(right)]

    return img

def save_jpg_70(img, target_path, quality=70):
    cv2.imwrite(target_path,img,[int(cv2.IMWRITE_JPEG_QUALITY),quality])
    
def save_jpg_90(img, target_path, quality=90):
    cv2.imwrite(target_path,img,[int(cv2.IMWRITE_JPEG_QUALITY),quality])
    
def save_resize_factor(img, target_path, factor):
    img = cv2.resize(img, (512,512),factor, factor, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(target_path,img)

def save_gamma_correction(img, target_path, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    img = np.uint8(img*255)
    cv2.imwrite(target_path,img)
 
########################################################################################################      
Augmentation_fold1 = trainingSet[trainingSet['Augmentationindices']==1]

def load_test_frombatch(images_batch):   
    for fl in images_batch.image_path.values:    
        img = get_im_cv2(fl) 
        new_fl = fl.replace('train','train_0')
        save_jpg_70(img, new_fl, quality=70)        
            
load_test_frombatch(Augmentation_fold1) 
########################################################################################################    
Augmentation_fold2 = trainingSet[trainingSet['Augmentationindices']==2]

def load_test_frombatch(images_batch):   
    for fl in images_batch.image_path.values:    
        img = get_im_cv2(fl) 
        new_fl = fl.replace('train','train_0')
        save_jpg_90(img, new_fl, quality=90)        
            
load_test_frombatch(Augmentation_fold2)
########################################################################################################  
Augmentation_fold3 = trainingSet[trainingSet['Augmentationindices']==3]

def load_test_frombatch(images_batch):   
    for fl in images_batch.image_path.values:    
        img = get_im_cv2(fl) 
        new_fl = fl.replace('train','train_0')
        save_resize_factor(img, new_fl,factor=0.5)                
            
load_test_frombatch(Augmentation_fold3)
########################################################################################################  

Augmentation_fold4 = trainingSet[trainingSet['Augmentationindices']==4]

def load_test_frombatch(images_batch):   
    for fl in images_batch.image_path.values:    
        img = get_im_cv2(fl) 
        new_fl = fl.replace('train','train_0')
        save_resize_factor(img, new_fl,factor=0.8)                
            
load_test_frombatch(Augmentation_fold4)
########################################################################################################  
Augmentation_fold5 = trainingSet[trainingSet['Augmentationindices']==5]

def load_test_frombatch(images_batch):   
    for fl in images_batch.image_path.values:    
        img = get_im_cv2(fl) 
        new_fl = fl.replace('train','train_0')
        save_resize_factor(img, new_fl,factor=1.5)                
            
load_test_frombatch(Augmentation_fold5)
########################################################################################################  
Augmentation_fold6 = trainingSet[trainingSet['Augmentationindices']==6]

def load_test_frombatch(images_batch):   
    for fl in images_batch.image_path.values:    
        img = get_im_cv2(fl) 
        new_fl = fl.replace('train','train_0')
        save_resize_factor(img, new_fl,factor=2)                
            
load_test_frombatch(Augmentation_fold6)
########################################################################################################  

Augmentation_fold7 = trainingSet[trainingSet['Augmentationindices']==7]

def load_test_frombatch(images_batch):   
    for fl in images_batch.image_path.values:    
        img = get_im_cv2(fl) 
        new_fl = fl.replace('train','train_0')
        save_gamma_correction(img, new_fl,correction=1.25)     # 1/0.8           
            
load_test_frombatch(Augmentation_fold7)
########################################################################################################
Augmentation_fold8 = trainingSet[trainingSet['Augmentationindices']==8] 

def load_test_frombatch(images_batch):   
    for fl in images_batch.image_path.values:    
        img = get_im_cv2(fl) 
        new_fl = fl.replace('train','train_0')
        save_gamma_correction(img, new_fl,correction=0.833)   # 1/1.2              
            
load_test_frombatch(Augmentation_fold8)
########################################################################################################
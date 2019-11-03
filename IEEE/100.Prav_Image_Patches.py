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

test = pd.read_csv(inDir+'/input/images_test.csv')
test['image_path'] = test['image_path'].apply(RowWiseOperation)


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
        new_fl = fl.replace('train','train_2')
        for i in range(0,4):
            new_path = new_fl.replace('.jpg','_{}.jpg'.format(i))
            scipy.misc.imsave(new_path, current_image_patches[i])
            
load_test_frombatch(images_train)           

def load_test_frombatch(images_batch):   
    for fl in images_batch.image_path.values:    
        img = get_im_cv2_centering(fl, new_height=448, new_width=448)  
        current_image_patches = image_patches(img,224,224)
        new_fl = fl.replace('test','test_2')
        for i in range(0,4):
            new_path = new_fl.replace('.tif','_{}.tif'.format(i))
            scipy.misc.imsave(new_path, current_image_patches[i])
            
load_test_frombatch(test)

images_train_Sony = images_train[images_train['image_category']=='Sony-NEX-7']

def load_test_frombatch(images_batch):   
    for fl in images_batch.image_path.values:    
        img = get_im_cv2_centering(fl, new_height=448, new_width=448)  
        current_image_patches = image_patches(img,224,224)
        new_fl = fl.replace('train','train_2')
        for i in range(0,4):
            new_path = new_fl.replace('.JPG','_{}.JPG'.format(i))
            scipy.misc.imsave(new_path, current_image_patches[i])
            
load_test_frombatch(images_train_Sony) 

images_train = pd.read_csv(inDir + '/input/Prav_5folds_CVindices.csv')

def RowWiseOperation(x):
    x1 = x.replace('\\','\\\\')
    return x1

images_train['image_path'] = images_train['image_path'].apply(RowWiseOperation)
images_train1 = images_train


def train_replace(x):
    return x.replace('train','train_2')

images_train1['image_path'] = images_train1['image_path'].apply(train_replace)




def jpg_0_adjustment(x):
    return x.replace('.jpg','_0.jpg')

def jpg_1_adjustment(x):
    return x.replace('.jpg','_1.jpg')

def jpg_2_adjustment(x):
    return x.replace('.jpg','_2.jpg')

def jpg_3_adjustment(x):
    return x.replace('.jpg','_3.jpg')

def JPG_0_adjustment(x):
    return x.replace('.JPG','_0.JPG')

def JPG_1_adjustment(x):
    return x.replace('.JPG','_1.JPG')

def JPG_2_adjustment(x):
    return x.replace('.JPG','_2.JPG')

def JPG_3_adjustment(x):
    return x.replace('.JPG','_3.JPG')

images_train1['image_path'] = images_train1['image_path'].apply(jpg_0_adjustment)
images_train1['image_path'] = images_train1['image_path'].apply(JPG_0_adjustment)
images_train1['image_name1'] = images_train1['image_name']
images_train1['image_name1'] = images_train1['image_name1'].apply(jpg_0_adjustment)
images_train1['image_name1'] = images_train1['image_name1'].apply(JPG_0_adjustment)

images_train = pd.read_csv(inDir + '/input/Prav_5folds_CVindices.csv')

def RowWiseOperation(x):
    x1 = x.replace('\\','\\\\')
    return x1

images_train['image_path'] = images_train['image_path'].apply(RowWiseOperation)
images_train2 = images_train

images_train2['image_path'] = images_train2['image_path'].apply(train_replace)

images_train2['image_path']  = images_train2['image_path'].apply(jpg_1_adjustment)
images_train2['image_path']  = images_train2['image_path'].apply(JPG_1_adjustment)
images_train2['image_name1'] = images_train2['image_name']
images_train2['image_name1'] = images_train2['image_name1'].apply(jpg_1_adjustment)
images_train2['image_name1'] = images_train2['image_name1'].apply(JPG_1_adjustment)

images_train = pd.read_csv(inDir + '/input/Prav_5folds_CVindices.csv')

def RowWiseOperation(x):
    x1 = x.replace('\\','\\\\')
    return x1

images_train['image_path'] = images_train['image_path'].apply(RowWiseOperation)
images_train3 = images_train

images_train3['image_path'] = images_train3['image_path'].apply(train_replace)


images_train3['image_path']  = images_train3['image_path'].apply(jpg_2_adjustment)
images_train3['image_path']  = images_train3['image_path'].apply(JPG_2_adjustment)
images_train3['image_name1'] = images_train3['image_name']
images_train3['image_name1'] = images_train3['image_name1'].apply(jpg_2_adjustment)
images_train3['image_name1'] = images_train3['image_name1'].apply(JPG_2_adjustment)

images_train = pd.read_csv(inDir + '/input/Prav_5folds_CVindices.csv')

def RowWiseOperation(x):
    x1 = x.replace('\\','\\\\')
    return x1

images_train['image_path'] = images_train['image_path'].apply(RowWiseOperation)
images_train4 = images_train

images_train4['image_path'] = images_train4['image_path'].apply(train_replace)


images_train4['image_path']  = images_train4['image_path'].apply(jpg_3_adjustment)
images_train4['image_path']  = images_train4['image_path'].apply(JPG_3_adjustment)
images_train4['image_name1'] = images_train4['image_name']
images_train4['image_name1'] = images_train4['image_name1'].apply(jpg_3_adjustment)
images_train4['image_name1'] = images_train4['image_name1'].apply(JPG_3_adjustment)



images_train_v2 = pd.concat([images_train1, images_train2,images_train3,images_train4] )

images_train_v2 = images_train_v2.reset_index(drop=True)

images_train_v2.groupby(['CVindices','image_category'])[['image_category']].size()

images_train_v2.to_csv(inDir+"/input/train_images_v2.csv", index=False)
images_train_v2.to_csv(inDir+"/input/Prav_5folds_CVindices_v2.csv", index=False)



def test_replace(x):
    return x.replace('test','test_2')

def tif_0_adjustment(x):
    return x.replace('.tif','_0.tif')

def tif_1_adjustment(x):
    return x.replace('.tif','_1.tif')

def tif_2_adjustment(x):
    return x.replace('.tif','_2.tif')

def tif_3_adjustment(x):
    return x.replace('.tif','_3.tif')

test = pd.read_csv(inDir+'/input/images_test.csv')
test['image_path'] = test['image_path'].apply(RowWiseOperation)
del test['image_id']
test['_id1'] = test['_id']
test1 = test
test1['image_path'] = test1['image_path'].apply(test_replace)
test1['image_path'] = test1['image_path'].apply(tif_0_adjustment)
test1['_id1'] = test1['_id1'].apply(tif_0_adjustment)

test = pd.read_csv(inDir+'/input/images_test.csv')
test['image_path'] = test['image_path'].apply(RowWiseOperation)
del test['image_id']
test['_id1'] = test['_id']
test2 = test
test2['image_path'] = test2['image_path'].apply(test_replace)
test2['image_path'] = test2['image_path'].apply(tif_1_adjustment)
test2['_id1'] = test2['_id1'].apply(tif_1_adjustment)

test = pd.read_csv(inDir+'/input/images_test.csv')
test['image_path'] = test['image_path'].apply(RowWiseOperation)
del test['image_id']
test['_id1'] = test['_id']
test3 = test
test3['image_path'] = test3['image_path'].apply(test_replace)
test3['image_path'] = test3['image_path'].apply(tif_2_adjustment)
test3['_id1'] = test3['_id1'].apply(tif_2_adjustment)

test = pd.read_csv(inDir+'/input/images_test.csv')
test['image_path'] = test['image_path'].apply(RowWiseOperation)
del test['image_id']
test['_id1'] = test['_id']
test4 = test
test4['image_path'] = test4['image_path'].apply(test_replace)
test4['image_path'] = test4['image_path'].apply(tif_3_adjustment)
test4['_id1'] = test4['_id1'].apply(tif_3_adjustment)

test_v2 = pd.concat([test1, test2,test3,test4] )

test_v2 = test_v2.reset_index(drop=True)

test_v2.to_csv(inDir+"/input/images_test_v2.csv", index=False)

#path = "C:\\Users\\SriPrav\\Documents\\R\\43IEEE\\input\\train\\HTC-1-M7\\(HTC-1-M7)1.jpg"
#new_fl = path.replace('.jpg','_{}.jpg'.format(0))
#
#path = ('C:\\Users\\SriPrav\\Documents\\R\\43IEEE\\input\\train_{0}\\HTC-1-M7\\(HTC-1-M7)1_{1}.jpg'.format(2,1))
#    
#current_image = get_im_cv2_centering(path, new_height=448, new_width=448)
#current_image_patches = image_patches(current_image,224,224)
#scipy.misc.imsave("C:\\Users\\SriPrav\\Documents\\R\\43IEEE\\input\\train_2\\HTC-1-M7\\(HTC-1-M7)1_1.jpg", current_image_patches[0])

#plt.imshow(img)
#image_patches[0].save("C:\\Users\\SriPrav\\Documents\\R\\43IEEE\\input\\train_2\\HTC-1-M7\\(HTC-1-M7)1_1.jpg")


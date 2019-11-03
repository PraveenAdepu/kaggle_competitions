
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
from keras.layers import Input
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.optimizers import SGD, Adam

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pickle

# Set some parameters
inDir = 'C:/Users/SriPrav/Documents/R/51TGS'

im_width = 128
im_height = 128
border = 5
im_chan = 2 # Number of channels: first is original and second cumsum(axis=0)
n_features = 1 # Number of extra features, like depth
path_train = inDir+'/input/train/'
path_test = inDir+'/input/test/'

df_depths = pd.read_csv(inDir+'/input/depths.csv', index_col='id')
df_depths.head()

#df_depths = pd.read_csv(inDir+'/input/depths.csv')
#train_imgs = next(os.walk(path_train+"images"))[2]

df_depths = pd.read_csv(inDir+'/input/depths.csv', index_col='id')
df_depths.head()

train_imgs_CVindices = pd.read_csv(inDir+'/input/Prav_10fold_CVindices.csv')
train_ids = train_imgs_CVindices["img"].tolist()

#train_ids = next(os.walk(path_train+"images"))[2]
test_ids = next(os.walk(path_test+"images"))[2]

#train_imgs = pd.DataFrame(train_imgs)
#train_imgs.columns = ['img']
#train_imgs['id'] = train_imgs['img'].str.replace('.png','')
#train_imgs = pd.merge(train_imgs,df_depths, on='id', how='left')
#
#
#from sklearn.model_selection import KFold
#DATA_ROOT = inDir+'/input/'
#
#
#def main():
#    n_fold = 10
#    depths = train_imgs.copy()
#    depths.sort_values('z', inplace=True)
##    depths.drop('z', axis=1, inplace=True)
#    depths['CVindices'] = (list(range(n_fold))*depths.shape[0])[:depths.shape[0]]
#    print(depths.head(), len(depths))
#    train_imgs_CVindices = pd.merge(train_imgs, depths, on=['img','id','z'], how='left')
#    # depths.to_csv(os.path.join(DATA_ROOT, 'folds.csv'), index=False)
#    return train_imgs_CVindices
#
#if __name__ == '__main__':
#    train_imgs_CVindices = main()
#
#train_imgs_CVindices.groupby(['CVindices']).count()
#
#
#train_imgs_CVindices.to_csv(inDir+'/input/Prav_10fold_CVindices.csv',index=False)

img_size_ori = 101
img_size_target = 128

train_df = pd.read_csv(inDir+"/input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv(inDir+"/input/depths.csv", index_col="id")


train_df["images"] = [np.array(load_img(inDir+"/input/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img(inDir+"/input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
        
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

train_df['id'] = train_df.index

#########################################################################################################################
random_state = 20180512
np.random.RandomState(random_state)

from sklearn.model_selection import StratifiedKFold

trainfoldSource = train_df[['id','coverage_class']]
trainfoldSource.reset_index(drop=True, inplace=True)

folds = 10
skf = StratifiedKFold(n_splits=folds, random_state = random_state, shuffle = True)

skf.get_n_splits(trainfoldSource, trainfoldSource['coverage_class'])

print(skf) 

Prav_CVindices = pd.DataFrame(columns=['index','CVindices_class'])

count = 1
for train_index, test_index in skf.split(trainfoldSource, trainfoldSource['coverage_class']):           
       df_index = pd.DataFrame(test_index,columns=['index']) 
       df_index['CVindices_class'] = count
       Prav_CVindices = Prav_CVindices.append(df_index)       
       count+=1
       
Prav_CVindices.set_index('index', inplace=True)
#Prav_CVindices.sort_index(inplace=True)

trainfoldSource = pd.merge(trainfoldSource, Prav_CVindices, left_index=True, right_index=True)

trainfoldSource.groupby(['CVindices_class','coverage_class'])[['coverage_class']].size()

del trainfoldSource['coverage_class']

train_imgs_CVindices = pd.read_csv(inDir+'/input/Prav_10fold_CVindices.csv')

train_imgs_CVindices_CoverageClasses = pd.merge(train_imgs_CVindices, trainfoldSource, on="id", how="left")

train_imgs_CVindices_CoverageClasses.to_csv(inDir+"/input/Prav_10folds_CVindices_CoverageClassStratified.csv", index=False)
############################################################################################################################







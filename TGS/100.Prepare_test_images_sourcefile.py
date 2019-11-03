# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:02:03 2017

@author: SriPrav
"""
import numpy as np
np.random.seed(201803)

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

inDir = 'C:/Users/SriPrav/Documents/R/51TGS'

path = os.path.join(inDir, 'input', 'test', 'images', '*.png')
files = glob.glob(path)
test_images = pd.DataFrame(files, columns=['img'])
test_images['img'] = test_images['img'].str.replace('\\','/')
sub_file = inDir + '/input/test_images.csv'
test_images.to_csv(sub_file, index=False)


path = os.path.join(inDir, 'input', 'train', 'images', '*.png')
files = glob.glob(path)
train_images = pd.DataFrame(files, columns=['img'])
train_images['img'] = train_images['img'].str.replace('\\','/')

train_images['mask'] = train_images['img'].str.replace('images','masks')

sub_file = inDir + '/input/train_images.csv'
train_images.to_csv(sub_file, index=False)

#path = os.path.join(inDir, 'input', 'train', 'masks', '*.png')
#files = glob.glob(path)
#test_images = pd.DataFrame(files, columns=['img'])
#sub_file = inDir + '/input/train_masks.csv'
#test_images.to_csv(sub_file, index=False)

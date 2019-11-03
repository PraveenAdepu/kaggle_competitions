# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 20:32:37 2018

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
from keras.layers import Input
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# Set some parameters
inDir = 'C:/Users/SriPrav/Documents/R/51TGS'

im_width = 128
im_height = 128
border = 5
im_chan = 3 # Number of channels: first is original and second cumsum(axis=0)
n_features = 1 # Number of extra features, like depth
path_train = inDir+'/input/train/'
path_test = inDir+'/input/test/'



train_ids = next(os.walk(path_train+"images"))[2]
test_ids = next(os.walk(path_test+"images"))[2]

import pickle

#with open('preds_test_unet041.pkl', 'rb') as f:
#    preds_test1 = pickle.load(f)
#with open('y_val_unet041.pkl', 'rb') as f:
#    y_val1 = pickle.load(f)
#with open('preds_val_unet041.pkl', 'rb') as f:
#    preds_val1 = pickle.load(f)
#    
#with open('preds_test_unet042.pkl', 'rb') as f:
#    preds_test2 = pickle.load(f)
#with open('y_val_unet042.pkl', 'rb') as f:
#    y_val2 = pickle.load(f)
#with open('preds_val_unet042.pkl', 'rb') as f:
#    preds_val2 = pickle.load(f)
#
#with open('preds_test_unet043.pkl', 'rb') as f:
#    preds_test3 = pickle.load(f)
#with open('y_val_unet043.pkl', 'rb') as f:
#    y_val3 = pickle.load(f)
#with open('preds_val_unet043.pkl', 'rb') as f:
#    preds_val3 = pickle.load(f)
#
#
#with open('preds_test_unet044.pkl', 'rb') as f:
#    preds_test4 = pickle.load(f)
#with open('y_val_unet044.pkl', 'rb') as f:
#    y_val4 = pickle.load(f)
#with open('preds_val_unet044.pkl', 'rb') as f:
#    preds_val4 = pickle.load(f)
#
#with open('preds_test_unet045.pkl', 'rb') as f:
#    preds_test5 = pickle.load(f)
#with open('y_val_unet045.pkl', 'rb') as f:
#    y_val5 = pickle.load(f)
#with open('preds_val_unet045.pkl', 'rb') as f:
#    preds_val5 = pickle.load(f)
#
#with open('preds_test_unet046.pkl', 'rb') as f:
#    preds_test6 = pickle.load(f)
#with open('y_val_unet046.pkl', 'rb') as f:
#    y_val6 = pickle.load(f)
#with open('preds_val_unet046.pkl', 'rb') as f:
#    preds_val6 = pickle.load(f)
#
#with open('preds_test_unet047.pkl', 'rb') as f:
#    preds_test7 = pickle.load(f)
#with open('y_val_unet047.pkl', 'rb') as f:
#    y_val7 = pickle.load(f)
#with open('preds_val_unet047.pkl', 'rb') as f:
#    preds_val7 = pickle.load(f)
#
#with open('preds_test_unet048.pkl', 'rb') as f:
#    preds_test8 = pickle.load(f)
#with open('y_val_unet048.pkl', 'rb') as f:
#    y_val8 = pickle.load(f)
#with open('preds_val_unet048.pkl', 'rb') as f:
#    preds_val8 = pickle.load(f)    
#
#with open('preds_test_unet049.pkl', 'rb') as f:
#    preds_test9 = pickle.load(f)
#with open('y_val_unet049.pkl', 'rb') as f:
#    y_val9 = pickle.load(f)
#with open('preds_val_unet049.pkl', 'rb') as f:
#    preds_val9 = pickle.load(f)
#
#with open('preds_test_unet0410.pkl', 'rb') as f:
#    preds_test10 = pickle.load(f)
#with open('y_val_unet0410.pkl', 'rb') as f:
#    y_val10 = pickle.load(f)
#with open('preds_val_unet0410.pkl', 'rb') as f:
#    preds_val10 = pickle.load(f)


#with open('preds_test_unet051.pkl', 'rb') as f:
#    preds_test1 = pickle.load(f)
#with open('y_val_unet051.pkl', 'rb') as f:
#    y_val1 = pickle.load(f)
#with open('preds_val_unet051.pkl', 'rb') as f:
#    preds_val1 = pickle.load(f)
#    
#with open('preds_test_unet052.pkl', 'rb') as f:
#    preds_test2 = pickle.load(f)
#with open('y_val_unet052.pkl', 'rb') as f:
#    y_val2 = pickle.load(f)
#with open('preds_val_unet052.pkl', 'rb') as f:
#    preds_val2 = pickle.load(f)
#
#with open('preds_test_unet053.pkl', 'rb') as f:
#    preds_test3 = pickle.load(f)
#with open('y_val_unet053.pkl', 'rb') as f:
#    y_val3 = pickle.load(f)
#with open('preds_val_unet053.pkl', 'rb') as f:
#    preds_val3 = pickle.load(f)
#
#
#with open('preds_test_unet054.pkl', 'rb') as f:
#    preds_test4 = pickle.load(f)
#with open('y_val_unet054.pkl', 'rb') as f:
#    y_val4 = pickle.load(f)
#with open('preds_val_unet054.pkl', 'rb') as f:
#    preds_val4 = pickle.load(f)
#
#with open('preds_test_unet055.pkl', 'rb') as f:
#    preds_test5 = pickle.load(f)
#with open('y_val_unet055.pkl', 'rb') as f:
#    y_val5 = pickle.load(f)
#with open('preds_val_unet055.pkl', 'rb') as f:
#    preds_val5 = pickle.load(f)
#
#with open('preds_test_unet056.pkl', 'rb') as f:
#    preds_test6 = pickle.load(f)
#with open('y_val_unet056.pkl', 'rb') as f:
#    y_val6 = pickle.load(f)
#with open('preds_val_unet056.pkl', 'rb') as f:
#    preds_val6 = pickle.load(f)
#
#with open('preds_test_unet057.pkl', 'rb') as f:
#    preds_test7 = pickle.load(f)
#with open('y_val_unet057.pkl', 'rb') as f:
#    y_val7 = pickle.load(f)
#with open('preds_val_unet057.pkl', 'rb') as f:
#    preds_val7 = pickle.load(f)
#
#with open('preds_test_unet058.pkl', 'rb') as f:
#    preds_test8 = pickle.load(f)
#with open('y_val_unet058.pkl', 'rb') as f:
#    y_val8 = pickle.load(f)
#with open('preds_val_unet058.pkl', 'rb') as f:
#    preds_val8 = pickle.load(f)    
#
#with open('preds_test_unet059.pkl', 'rb') as f:
#    preds_test9 = pickle.load(f)
#with open('y_val_unet059.pkl', 'rb') as f:
#    y_val9 = pickle.load(f)
#with open('preds_val_unet059.pkl', 'rb') as f:
#    preds_val9 = pickle.load(f)
#
#with open('preds_test_unet0510.pkl', 'rb') as f:
#    preds_test10 = pickle.load(f)
#with open('y_val_unet0510.pkl', 'rb') as f:
#    y_val10 = pickle.load(f)
#with open('preds_val_unet0510.pkl', 'rb') as f:
#    preds_val10 = pickle.load(f)
    

with open('preds_test_unet061.pkl', 'rb') as f:
    preds_test1 = pickle.load(f)
with open('y_val_unet061.pkl', 'rb') as f:
    y_val1 = pickle.load(f)
with open('preds_val_unet061.pkl', 'rb') as f:
    preds_val1 = pickle.load(f)
    
with open('preds_test_unet062.pkl', 'rb') as f:
    preds_test2 = pickle.load(f)
with open('y_val_unet062.pkl', 'rb') as f:
    y_val2 = pickle.load(f)
with open('preds_val_unet062.pkl', 'rb') as f:
    preds_val2 = pickle.load(f)

with open('preds_test_unet063.pkl', 'rb') as f:
    preds_test3 = pickle.load(f)
with open('y_val_unet063.pkl', 'rb') as f:
    y_val3 = pickle.load(f)
with open('preds_val_unet063.pkl', 'rb') as f:
    preds_val3 = pickle.load(f)


with open('preds_test_unet064.pkl', 'rb') as f:
    preds_test4 = pickle.load(f)
with open('y_val_unet064.pkl', 'rb') as f:
    y_val4 = pickle.load(f)
with open('preds_val_unet064.pkl', 'rb') as f:
    preds_val4 = pickle.load(f)

with open('preds_test_unet065.pkl', 'rb') as f:
    preds_test5 = pickle.load(f)
with open('y_val_unet065.pkl', 'rb') as f:
    y_val5 = pickle.load(f)
with open('preds_val_unet065.pkl', 'rb') as f:
    preds_val5 = pickle.load(f)

with open('preds_test_unet066.pkl', 'rb') as f:
    preds_test6 = pickle.load(f)
with open('y_val_unet066.pkl', 'rb') as f:
    y_val6 = pickle.load(f)
with open('preds_val_unet066.pkl', 'rb') as f:
    preds_val6 = pickle.load(f)

with open('preds_test_unet067.pkl', 'rb') as f:
    preds_test7 = pickle.load(f)
with open('y_val_unet067.pkl', 'rb') as f:
    y_val7 = pickle.load(f)
with open('preds_val_unet067.pkl', 'rb') as f:
    preds_val7 = pickle.load(f)

with open('preds_test_unet068.pkl', 'rb') as f:
    preds_test8 = pickle.load(f)
with open('y_val_unet068.pkl', 'rb') as f:
    y_val8 = pickle.load(f)
with open('preds_val_unet068.pkl', 'rb') as f:
    preds_val8 = pickle.load(f)    

with open('preds_test_unet069.pkl', 'rb') as f:
    preds_test9 = pickle.load(f)
with open('y_val_unet069.pkl', 'rb') as f:
    y_val9 = pickle.load(f)
with open('preds_val_unet069.pkl', 'rb') as f:
    preds_val9 = pickle.load(f)

with open('preds_test_unet0610.pkl', 'rb') as f:
    preds_test10 = pickle.load(f)
with open('y_val_unet0610.pkl', 'rb') as f:
    y_val10 = pickle.load(f)
with open('preds_val_unet0610.pkl', 'rb') as f:
    preds_val10 = pickle.load(f)
    
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
##################################################################################################################################

with open('preds_test_unet0910.pkl', 'rb') as f:
    preds_test10 = pickle.load(f)
with open('y_val_unet0910.pkl', 'rb') as f:
    y_val10 = pickle.load(f)
with open('preds_val_unet1010.pkl', 'rb') as f:
    preds_val10 = pickle.load(f)

thres = np.linspace(0.25, 0.75, 20)
thres_ioc10 = [iou_metric_batch(y_val10, np.int32(preds_val10 > t)) for t in tqdm_notebook(thres)]
best_thres10 = thres[np.argmax(thres_ioc10)]
best_thres10, max(thres_ioc10)      
    
pred_dict = {id_[:-4]:RLenc(np.round(preds_test10[i] > best_thres10)) for i,id_ in tqdm_notebook(enumerate(test_ids))}
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub_file = inDir + "/submissions/Prav_Unet10_fold10.csv"
sub.to_csv(sub_file)

#################################################################################################################################
with open('preds_test_unet0910.pkl', 'rb') as f:
    preds_unet09 = pickle.load(f)
with open('preds_test_unet1010.pkl', 'rb') as f:
    preds_unet10 = pickle.load(f)


0.48684210526315785, 0.78553299492385786

best_thres = (0.6578947368421052+0.61052631578947358)/2
best_thres = 0.5
    
preds_test_upsampled_Ensemble = [0.5*a + 0.5*b    for a, b in zip(preds_unet09, preds_unet10)] 

preds_unet09[0]
preds_unet10[0]
preds_test_upsampled_Ensemble[0]

#unet04 = 0.6763157894736842
#unet05 = 0.6578947368421052
#unet06 = 0.61052631578947358

pred_dict = {id_[:-4]:RLenc(np.round(preds_test_upsampled_Ensemble[i] > best_thres)) for i,id_ in tqdm_notebook(enumerate(test_ids))}
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub_file = inDir + "/submissions/Prav_unet0910_folds10.csv"
sub.to_csv(sub_file)
    

################################################################################################################################## 
import numpy as np
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
%matplotlib inline

def rle_decode(rle_mask):
    '''
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101*101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101,101)
"""
reading and decoding the submission 

"""
df = pd.read_csv(inDir + "/submissions/Prav_Unet06_folds10.csv")
i = 0
j = 0
plt.figure(figsize=(30,15))
plt.subplots_adjust(bottom=0.2, top=0.8, hspace=0.2)  #adjust this to change vertical and horiz. spacings..
# Visualizing the predicted outputs
while True:
    if str(df.loc[i,'rle_mask'])!=str(np.nan):        
        decoded_mask = rle_decode(df.loc[i,'rle_mask'])
        plt.subplot(1,6,j+1)
        plt.imshow(decoded_mask)
        plt.title('ID: '+df.loc[i,'id'])
        j = j + 1
        if j>5:
            break
    i = i + 1
    
"""
Function which returns the labelled image after applying CRF

"""
#Original_image = Image which has to labelled
#Mask image = Which has been labelled by some technique..
def crf(original_image, mask_img):
    
    # Converting annotated image to RGB if it is Gray scale
    if(len(mask_img.shape)<3):
        mask_img = gray2rgb(mask_img)

#     #Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)
    
#     # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2
    
    #Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 10 steps 
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0],original_image.shape[1]))
test_path = inDir+ '/input/test/images/' #../input/tgs-salt-identification-challenge/test/images/'

"""
visualizing the effect of applying CRF

"""
nImgs = 3
i = np.random.randint(1000)
j = 1
plt.figure(figsize=(15,15))
plt.subplots_adjust(wspace=0.2,hspace=0.1)  #adjust this to change vertical and horiz. spacings..
while True:
    if str(df.loc[i,'rle_mask'])!=str(np.nan):        
        decoded_mask = rle_decode(df.loc[i,'rle_mask'])        
        orig_img = imread(test_path+df.loc[i,'id']+'.png')
        #Applying CRF on FCN-16 annotated image
        crf_output = crf(orig_img,decoded_mask)
        plt.subplot(nImgs,4,4*j-3)
        plt.imshow(orig_img)
        plt.title('Original image')
        plt.subplot(nImgs,4,4*j-2)
        plt.imshow(decoded_mask) 
        plt.title('Original Mask')
        plt.subplot(nImgs,4,4*j-1)
        plt.imshow(crf_output) 
        plt.title('Mask after CRF')
        if j == nImgs:
            break
        else:
            j = j + 1
    i = i + 1
    
"""
used for converting the decoded image to rle mask

"""
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
"""
Applying CRF on the predicted mask 

"""
#from PIL import Image
#import io
#
## This portion is part of my test code
#byteImgIO = io.BytesIO()
#byteImg = Image.open("some/location/to/a/file/in/my/directories.png")
#byteImg.save(byteImgIO, "PNG")
#byteImgIO.seek(0)
#byteImg = byteImgIO.read()
## Non test code
#dataBytesIO = io.BytesIO(byteImg)
#Image.open(dataBytesIO)

for i in tqdm(range(df.shape[0])):
    if str(df.loc[i,'rle_mask'])!=str(np.nan):        
        decoded_mask = rle_decode(df.loc[i,'rle_mask'])        
        orig_img = imread(test_path+df.loc[i,'id']+'.png')
#        byteImgIO = io.BytesIO()
#        byteImg = Image.open(test_path+df.loc[i,'id']+'.png')
#        byteImg.save(byteImgIO, "PNG")
#        byteImgIO.seek(0)
#        byteImg = byteImgIO.read()
#        # Non test code
#        dataBytesIO = io.BytesIO(byteImg)
#        orig_img = Image.open(dataBytesIO)        
        crf_output = crf(orig_img,decoded_mask)
        df.loc[i,'rle_mask'] = rle_encode(crf_output)
#df.to_csv('crf_correction.csv',index=False)
sub_file = inDir + "/submissions/Prav_Unet06_folds10_crf_corrected.csv"
df.to_csv(sub_file,index=False)

##################################################################################################################################    
thres = np.linspace(0.25, 0.75, 20)
thres_ioc01 = [iou_metric_batch(y_val1, np.int32(preds_val1 > t)) for t in tqdm_notebook(thres)]
best_thres01 = thres[np.argmax(thres_ioc01)]
best_thres01, max(thres_ioc01)

thres_ioc02 = [iou_metric_batch(y_val2, np.int32(preds_val2 > t)) for t in tqdm_notebook(thres)]
best_thres02 = thres[np.argmax(thres_ioc02)]
best_thres02, max(thres_ioc02)

thres_ioc03 = [iou_metric_batch(y_val3, np.int32(preds_val3 > t)) for t in tqdm_notebook(thres)]
best_thres03 = thres[np.argmax(thres_ioc03)]
best_thres03, max(thres_ioc03)

thres_ioc04 = [iou_metric_batch(y_val4, np.int32(preds_val4 > t)) for t in tqdm_notebook(thres)]
best_thres04 = thres[np.argmax(thres_ioc04)]
best_thres04, max(thres_ioc04)

thres_ioc05 = [iou_metric_batch(y_val5, np.int32(preds_val5 > t)) for t in tqdm_notebook(thres)]
best_thres05 = thres[np.argmax(thres_ioc05)]
best_thres05, max(thres_ioc05)

thres_ioc06 = [iou_metric_batch(y_val6, np.int32(preds_val6 > t)) for t in tqdm_notebook(thres)]
best_thres06 = thres[np.argmax(thres_ioc06)]
best_thres06, max(thres_ioc06)

thres_ioc07 = [iou_metric_batch(y_val7, np.int32(preds_val7 > t)) for t in tqdm_notebook(thres)]
best_thres07 = thres[np.argmax(thres_ioc07)]
best_thres07, max(thres_ioc07)

thres_ioc08 = [iou_metric_batch(y_val8, np.int32(preds_val8 > t)) for t in tqdm_notebook(thres)]
best_thres08 = thres[np.argmax(thres_ioc08)]
best_thres08, max(thres_ioc08)

thres_ioc09 = [iou_metric_batch(y_val9, np.int32(preds_val9 > t)) for t in tqdm_notebook(thres)]
best_thres09 = thres[np.argmax(thres_ioc09)]
best_thres09, max(thres_ioc09)

thres_ioc10 = [iou_metric_batch(y_val10, np.int32(preds_val10 > t)) for t in tqdm_notebook(thres)]
best_thres10 = thres[np.argmax(thres_ioc10)]
best_thres10, max(thres_ioc10)



best_thres = (best_thres01 + best_thres02+ best_thres03+ best_thres04+ best_thres05+ best_thres06+ best_thres07+ best_thres08+ best_thres09+ best_thres10)/10
best_thres
    
preds_test_upsampled_Ensemble = [0.1*a + 0.1*b + 0.1*c + 0.1*d + 0.1*e + 0.1*f + 0.1*g + 0.1*h + 0.1*i + 0.1*j   for a, b,c,d,e,f,g,h,i,j in zip(preds_test1, preds_test2,preds_test3, preds_test4,preds_test5, preds_test6,preds_test7, preds_test8,preds_test9,preds_test10)] 

preds_test1[0]
preds_test2[0]
preds_test_upsampled_Ensemble[0]

#unet04 = 0.6763157894736842
#unet05 = 0.6578947368421052
#unet06 = 0.61052631578947358

pred_val_pkl = 'Prav_unet06_folds10.pkl'    
with open(pred_val_pkl, 'wb') as f:
    pickle.dump(preds_test_upsampled_Ensemble, f)

pred_dict = {id_[:-4]:RLenc(np.round(preds_test_upsampled_Ensemble[i] > best_thres)) for i,id_ in tqdm_notebook(enumerate(test_ids))}
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub_file = inDir + "/submissions/Prav_unet06_folds10.csv"
sub.to_csv(sub_file)


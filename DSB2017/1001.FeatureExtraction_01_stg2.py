# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 17:53:22 2017

@author: SriPrav
"""

import numpy as np
import dicom
import glob
import os
import cv2
import mxnet as mx
import pandas as pd
from sklearn import cross_validation
import xgboost as xgb
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc


inDir = 'C:/Users/SriPrav/Documents/R/19DSB2017'

resnet50Model = inDir + "/input/preModels/resnet-50"
Stage1SourceFolder = inDir + "/input/sources/stage2/stage2/*"
FeaturesExtraction_numpyFiles = inDir + "/input/sources/stage2/stage2/"
FeatureExtraction_Folder = inDir + "/input/FeatureExtraction_01_stg2"


def get_extractor():
    model = mx.model.FeedForward.load(resnet50Model, 0, ctx=mx.gpu(), numpy_batch_size=1)
    #model = mx.mod.Module.load('C:/Users/SriPrav/Documents/R/19DSB2017/input/model/resnet-50', 0)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), 
                                      symbol=fea_symbol ,
                                             numpy_batch_size=64,
                                             arg_params=model.arg_params, 
                                             aux_params=model.aux_params,
                                             allow_extra_params=True
                                             )

    return feature_extractor


def get_segmented_lungs(im, plot=False):
    
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < 604
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone) 
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone) 
        
    return im

def segment_lung_from_ct_scan(slices):
      segmented_slice = np.asarray([get_segmented_lungs(slice) for slice in slices])
      segmented_slice[segmented_slice < 604] = 0
      return segmented_slice
	
def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
	# Get the pixel values for all the slices
    slices = np.stack([s.pixel_array for s in slices])
    slices[slices == -2000] = 0
    return slices

def one_if_zero(x):
   if x == 0:
       return 1
   return x

def get_data_id(path):
    sample_image = get_3d_data(path)
    sample_image = segment_lung_from_ct_scan(sample_image)

    batch = []
    cnt = 0
    dx = 40
    ds = 512
    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = sample_image[i + j]
            imgmax = one_if_zero(np.amax(img))
            img = 255.0 / imgmax * img
            #img = 255.0 / np.amax(img) * img
            img = cv2.equalizeHist(img.astype(np.uint8))
            img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(np.array(tmp))

    batch = np.array(batch)
    return batch


def calc_features():
    net = get_extractor()
    for folder in glob.glob(Stage1SourceFolder):
        batch = get_data_id(folder)
        feats = net.predict(batch)
        print(feats.shape)
        np.save(folder, feats)

if __name__ == '__main__':
    calc_features()

##########################################################################################################
# Move the feature extraction numpy files to features folder 
##########################################################################################################    
import os
import shutil


files = os.listdir(FeaturesExtraction_numpyFiles)

for f in files:
    if f.endswith('.npy'):
        shutil.move(os.path.join(FeaturesExtraction_numpyFiles,f), os.path.join(FeatureExtraction_Folder,f))   
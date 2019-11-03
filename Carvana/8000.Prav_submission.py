# -*- coding: utf-8 -*-
"""
Created on Tue Aug 01 19:19:05 2017

@author: SriPrav
"""

import time

import numpy as np
import pandas as pd
from scipy import ndimage
import cv2
from skimage.transform import resize

from matplotlib import pyplot as plt

np.random.seed(2017)

inDir = 'C:/Users/SriPrav/Documents/R/29Carvana'

test_id_224_3                 = np.load(inDir +"/input/test_id_128_1.npy")
imgs_mask_test10_Unet_CNN01   = np.load(inDir +"/imgs_mask_test10_Unet_CNN01.npy")

image_rows = 1280 
image_cols = 1918

#path = "C:/Users/SriPrav/Documents/R/29Carvana/input/test/0004d4463b50_01.jpg"
#img = cv2.imread(path)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#plt.imshow(img) # 1280, 1918
#test_id_224_3[3]
#img = imgs_mask_test10_Unet_CNN01[3]
#img = (img[0,:, :] * 255.).astype(np.uint8)
#img = img[0,:,:]
#img = prep(img)
#plt.imshow(img)
#imgs.shape


def prep(img):
    img = img.astype('float32')
    img = (img[0,:, :] * 255.).astype(np.uint8)
    img = resize(img, ( image_rows,image_cols))
    img = (img > 0.5).astype(np.uint8)  # threshold    
    return img
    
#rle_encode1 = rle_encode(img)    
#def rle_encode(mask_image):
#    pixels = mask_image.flatten()
#    # We avoid issues with '1' at the start or end (at the corners of 
#    # the original image) by setting those pixels to '0' explicitly.
#    # We do not expect these to be non-zero for an accurate mask, 
#    # so this should not harm the score.
#    pixels[0] = 0
#    pixels[-1] = 0
#    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
#    runs[1::2] = runs[1::2] - runs[:-1:2]
#    return runs
#
#
#
#def rle_to_string(runs):
#    return ' '.join(str(x) for x in runs)
    
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)
    
    
#def test_rle_encode():
#    test_mask = np.asarray([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
#    assert rle_to_string(rle_encode(test_mask)) == '7 2 11 2'
#    num_masks = len(train_masks['img'])
#    print('Verfiying RLE encoding on', num_masks, 'masks ...')
#    time_read = 0.0 # seconds
#    time_rle = 0.0 # seconds
#    time_stringify = 0.0 # seconds
#    for mask_idx in range(num_masks):
#        img_file_name = train_masks.loc[mask_idx, 'img']
#        car_code, angle_code = img_file_name.split('.')[0].split('_')
#        t0 = time.clock()
#        mask_image = read_mask_image(car_code, angle_code)
#        time_read += time.clock() - t0
#        t0 = time.clock()
#        rle_truth_str = train_masks.loc[mask_idx, 'rle_mask']
#        rle = rle_encode(mask_image)
#        time_rle += time.clock() - t0
#        t0 = time.clock()
#        rle_str = rle_to_string(rle)
#        time_stringify += time.clock() - t0
#        assert rle_str == rle_truth_str
#        if mask_idx and (mask_idx % 500) == 0:
#            print('  ..', mask_idx, 'tested ..')
#    print('Time spent reading mask images:', time_read, 's, =>', \
#            1000*(time_read/num_masks), 'ms per mask.')
#    print('Time spent RLE encoding masks:', time_rle, 's, =>', \
#            1000*(time_rle/num_masks), 'ms per mask.')
#    print('Time spent stringifying RLEs:', time_stringify, 's, =>', \
#            1000*(time_stringify/num_masks), 'ms per mask.')
#
#
#test_rle_encode()    

#def submission():
#    
#    imgs_id_test = test_id_224_3
#    imgs_test = imgs_mask_test10_Unet_CNN01
#
#    total = imgs_test.shape[0]
#   
#    img = []
#    rle_mask = []
#    for i in range(total):
#        imgs = imgs_test[i]
#        imgs = imgs[0,:,:]
#        imgs = prep(imgs)
#        rle = rle_encode(imgs)
#        rle_str = rle_to_string(rle)
#
#        rle_mask.append(rle_str)
#        img.append(imgs_id_test[i])
#
#        if i % 100 == 0:
#            print('{}/{}'.format(i, total))
#
#    first_row = 'img,rle_mask'
#    file_name = 'submission.csv'
#
#    with open(file_name, 'w+') as f:
#        f.write(first_row + '\n')
#        for i in range(total):
#            s = str(img[i]) + ',' + rle_mask[i]
#            f.write(s + '\n')
#
#
#if __name__ == '__main__':
#    submission()

    
def submission():
    
    imgs_id_test = test_id_224_3
    imgs_test = imgs_mask_test

    total =  imgs_id_test.shape[0]
   
    img = []
    rle_mask = []
    for i in range(total):
        imgs = imgs_test[i]
#        imgs = imgs[0,:,:]
        imgs = prep(imgs)
#        rle = rle_encode(imgs)
#        rle_str = rle_to_string(rle)
        rle_str = rle_encode(imgs)
        rle_mask.append(rle_str)
        img.append(imgs_id_test[i])
        
        if i % 100 == 0:
            print('{}/{}'.format(i, total))
        
    img_pd = pd.DataFrame(img)        
    img_pd.columns = ['img']

    rle_mask_pd = pd.DataFrame(rle_mask)
    rle_mask_pd.columns = ['rle_mask']

    submission_pd = pd.concat([img_pd, rle_mask_pd], axis = 1)
    sub_file = inDir + "/submissions/Prav_Unet02_CNN01_hq.csv"
    submission_pd.to_csv(sub_file, index=False)
    
if __name__ == '__main__':
    submission()

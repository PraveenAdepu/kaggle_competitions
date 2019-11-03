# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:23:01 2018

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
import urllib.request


inDir = 'C:/Users/SriPrav/Documents/R/43IEEE'
images_train = pd.read_csv(inDir+'/input/AllData/val_images/htc_m7/urls_dpreview.csv', header=None)
images_train.columns = ["ImagePath"]

for i in range(0, len(images_train)):
    try:
        image_path = images_train["ImagePath"].ix[i]
        image_name = inDir+"/input/AllData/val_images/htc_m7/" +image_path.split("/")[-1]
        urllib.request.urlretrieve(image_path, image_name)
    except:
        continue

images_train = pd.read_csv(inDir+'/input/AllData/val_images/iphone_4s/urls_anandtech.csv', header=None)
images_train.columns = ["ImagePath"]

for i in range(0, len(images_train)):
    try:
        image_path = images_train["ImagePath"].ix[i]
        image_name = inDir+"/input/AllData/val_images/iphone_4s/" +image_path.split("/")[-1]
        urllib.request.urlretrieve(image_path, image_name)
    except:
        continue
    
images_train = pd.read_csv(inDir+'/input/AllData/val_images/iphone_6/urls_dpreview.csv', header=None)
images_train.columns = ["ImagePath"]

for i in range(0, len(images_train)):
    try:
        image_path = images_train["ImagePath"].ix[i]
        image_name = inDir+"/input/AllData/val_images/iphone_6/" +image_path.split("/")[-1]
        urllib.request.urlretrieve(image_path, image_name)
    except:
        continue  

images_train = pd.read_csv(inDir+'/input/AllData/val_images/moto_maxx/urls_phonearena.csv', header=None)
images_train.columns = ["ImagePath"]

for i in range(0, len(images_train)):
    try:
        image_path = images_train["ImagePath"].ix[i]
        image_name = inDir+"/input/AllData/val_images/moto_maxx/" +image_path.split("/")[-1]
        urllib.request.urlretrieve(image_path, image_name)
    except:
        continue   

images_train = pd.read_csv(inDir+'/input/AllData/val_images/moto_x/urls_dpreview.csv', header=None)
images_train.columns = ["ImagePath"]

for i in range(0, len(images_train)):
    try:
        image_path = images_train["ImagePath"].ix[i]
        image_name = inDir+"/input/AllData/val_images/moto_x/" +image_path.split("/")[-1]
        urllib.request.urlretrieve(image_path, image_name)
    except:
        continue  
    

images_train = pd.read_csv(inDir+'/input/AllData/val_images/nexus_5x/urls_engadget.csv', header=None)
images_train.columns = ["ImagePath"]

for i in range(0, len(images_train)):
    try:
        image_path = images_train["ImagePath"].ix[i]
        image_name = inDir+"/input/AllData/val_images/nexus_5x/" +image_path.split("/")[-1]
        urllib.request.urlretrieve(image_path, image_name)
    except:
        continue
    
images_train = pd.read_csv(inDir+'/input/AllData/val_images/nexus_6/urls_dpreview.csv', header=None)
images_train.columns = ["ImagePath"]

for i in range(0, len(images_train)):
    try:
        image_path = images_train["ImagePath"].ix[i]
        image_name = inDir+"/input/AllData/val_images/nexus_6/" +image_path.split("/")[-1]
        urllib.request.urlretrieve(image_path, image_name)
    except:
        continue
    
images_train = pd.read_csv(inDir+'/input/AllData/val_images/samsung_note3/urls_dpreview.csv', header=None)
images_train.columns = ["ImagePath"]

for i in range(0, len(images_train)):
    try:
        image_path = images_train["ImagePath"].ix[i]
        image_name = inDir+"/input/AllData/val_images/samsung_note3/" +image_path.split("/")[-1]
        urllib.request.urlretrieve(image_path, image_name)
    except:
        continue
    
images_train = pd.read_csv(inDir+'/input/AllData/val_images/samsung_s4/urls_dpreview.csv', header=None)
images_train.columns = ["ImagePath"]

for i in range(0, len(images_train)):
    try:
        image_path = images_train["ImagePath"].ix[i]
        image_name = inDir+"/input/AllData/val_images/samsung_s4/" +image_path.split("/")[-1]
        urllib.request.urlretrieve(image_path, image_name)
    except:
        continue
    
images_train = pd.read_csv(inDir+'/input/AllData/val_images/sony_nex7/urls_dpreview.csv', header=None)
images_train.columns = ["ImagePath"]

for i in range(0, len(images_train)):
    try:
        image_path = images_train["ImagePath"].ix[i]
        image_name = inDir+"/input/AllData/val_images/sony_nex7/" +image_path.split("/")[-1]
        urllib.request.urlretrieve(image_path, image_name)
    except:
        continue
    

    


    
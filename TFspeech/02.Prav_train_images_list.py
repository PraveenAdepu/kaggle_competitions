# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 18:34:52 2017

@author: SriPrav
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import math
from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
import glob
import os


inDir = 'C:\\Users\\SriPrav\\Documents\R\\37TFSpeech\\input\\picts\\train'


def write_train_csv():
    out = open('images_train.csv', "w")
    out.write("image_path,image_name,image_category\n")
    for path, subdirs, files in os.walk(inDir):
        for name in files:
            image_path =  os.path.join(path, name)           
            image_name = name
            image_category = path.replace(inDir,'').replace('\\','')
            
            out.write(str(image_path))
            out.write(',' + str(image_name))
            out.write(',' + str(image_category))            
            out.write('\n')
    
    out.close()

if __name__ == '__main__':
    write_train_csv()


#3.095,080
inDir = 'C:\\Users\\SriPrav\\Documents\\R\\37TFSpeech\\input'

test_images = 'C:\\Users\\SriPrav\\Documents\\R\\37TFSpeech\\input\\picts\\test'

def write_test_csv():
    out = open('images_test.csv', "w")
    out.write("image_path,image_name\n")
    for files in os.listdir(test_images):
#        files = "clip_01a55f8fe.png"          
        image_path =  os.path.join(test_images,files) 
#        firstsplit = files.split(".",1)[0]
#        secondsplit = firstsplit.split("-",1)
#        _id = secondsplit[0]
        image_name = files

        out.write(str(image_path))
#        out.write(',' + str(_id))
        out.write(',' + str(image_name))            
        out.write('\n')
    
    out.close()

if __name__ == '__main__':
    write_test_csv()



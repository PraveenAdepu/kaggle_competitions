import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string

from PIL import Image, ImageDraw, ImageFilter, ImageStat, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
%matplotlib inline
import nltk
import glob

inDir = 'C:/Users/SriPrav/Documents/R/21Rental'

images = sorted(glob.glob(inDir + '/input/images/images/**/**.jpg')) # 696,137 , 6811957

columns = ["folder","imageNo","width", " height", "sizebytes"," extrema00", " extrema01", " extrema10", " extrema11", " extrema20", " extrema21", "count00", "count01", "count02", "sum00", "sum01", "sum02", "mean00", "mean01", "mean02" , "median00", "median01", "median02", "rms00", "rms01", "rms02", "var00", "var01", "var02", "stddev00", "stddev01", "stddev02"]

image_properties = []

#im = 'C:/Users/SriPrav/Documents/R/21Rental/input/images/images/6851857/6851857_1f751afc6826e7660cba4ce46c6fddfc.jpg'

for im in images:
    #print(im)
    current_image = []
    folder = (im.split("\\")[-2:])[0]
    imageNo = (im.split("\\")[-1:])[0]
    img = Image.open(im)
    width, height = img.size
    sizebytes = os.path.getsize(im)
    stats = ImageStat.Stat(img, mask=None)
    extrema00 = stats.extrema[0][0]
    extrema01 = stats.extrema[0][1]
    try:
        extrema10 = stats.extrema[1][0]
    except IndexError:
        extrema10 = 0
    try:
        extrema11 = stats.extrema[1][1]
    except IndexError:
        extrema11 = 0
    try:
        extrema20 = stats.extrema[2][0]
    except IndexError:
        extrema20 = 0
    try:
        extrema21 = stats.extrema[2][1]
    except IndexError:
        extrema21 = 0
    
    count00 = stats.count[0]
    try:
        count01 = stats.count[1]
    except IndexError:
        count01 = 0
    try:
        count02 = stats.count[2]
    except IndexError:
        count02 = 0
    
    sum00 = stats.sum[0]
    
    try:
        sum01 = stats.sum[1]
    except IndexError:
        sum01 = 0
    try:
        sum02 = stats.sum[2]
    except IndexError:
        sum02 = 0
    
    sum200 = stats.sum2[0]
    try:
        sum201 = stats.sum2[1]
    except IndexError:
        sum201 = 0
    try:
        sum202 = stats.sum2[2]
    except IndexError:
        sum202 = 0
    
    mean00 = stats.mean[0]
    try:
        mean01 = stats.mean[1]
    except IndexError:
        mean01 = 0
    try:
        mean02 = stats.mean[2]
    except IndexError:
        mean02 = 0
    
    median00 = stats.median[0]
    try:
        median01 = stats.median[1]
    except IndexError:
        median01 = 0
    try:
        median02 = stats.median[2]
    except IndexError:
        median02 = 0
    
    rms00 = stats.rms[0]
    try:
        rms01 = stats.rms[1]
    except IndexError:
        rms01 = 0
    
    try:
        rms02 = stats.rms[2]
    except IndexError:
        rms02 = 0
    
    var00 = stats.var[0]
    try:
        var01 = stats.var[1]
    except IndexError:
        var01 = 0
    try:
        var02 = stats.var[2]
    except IndexError:
        var02 = 0
    
    stddev00 = stats.stddev[0]
    try:
        stddev01 = stats.stddev[1]
    except IndexError:
        stddev01 = 0
    try:
        stddev02 = stats.stddev[2]
    except IndexError:
        stddev02 = 0
    
  
    current_image.append(folder)
    current_image.append(imageNo)
    current_image.append(width)
    current_image.append(height)
    current_image.append(sizebytes)
    current_image.append(extrema00)
    current_image.append(extrema01)
    current_image.append(extrema10)
    current_image.append(extrema11)
    current_image.append(extrema20)
    current_image.append(extrema21)
    current_image.append(count00)
    current_image.append(count01)
    current_image.append(count02)
    
    current_image.append(sum00)
    current_image.append(sum01)
    current_image.append(sum02)
    
    current_image.append(mean00)
    current_image.append(mean01)
    current_image.append(mean02)
    
    current_image.append(median00)
    current_image.append(median01)
    current_image.append(median02)
    
    current_image.append(rms00)
    current_image.append(rms01)
    current_image.append(rms02)
    
    current_image.append(var00)
    current_image.append(var01)
    current_image.append(var02)
    
    current_image.append(stddev00)
    current_image.append(stddev01)
    current_image.append(stddev02)
#    break 
    image_properties.append(current_image)

image_properties_df = pd.DataFrame(image_properties, columns=columns) 

image_properties_file = 'C:/Users/SriPrav/Documents/R/21Rental/input/Image_Properties_02.csv'
image_properties_df.to_csv(image_properties_file, index=False)

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
import io
from PIL import Image, ImageDraw, ImageFilter, ImageStat, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
%matplotlib inline
import nltk
import glob
import imagehash, hashlib

inDir = 'C:/Users/SriPrav/Documents/R/21Rental'

images = sorted(glob.glob(inDir + '/input/images/images/**/**.jpg')) # 696,137 , 6811957

columns = ["folder","imageNo","hash"]

image_properties = []

def dhash(image,hash_size = 16):
    image = image.convert('LA').resize((hash_size+1,hash_size),Image.ANTIALIAS)
    pixels = list(image.getdata())
    difference = []
    for row in xrange(hash_size):
        for col in xrange(hash_size):
            pixel_left = image.getpixel((col,row))
            pixel_right = image.getpixel((col+1,row))
            difference.append(pixel_left>pixel_right)
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(difference):
        if value:
            decimal_value += 2**(index%8)
        if (index%8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2,'0'))
            decimal_value = 0
    
    return ''.join(hex_string)
    
for im in images:
    #print(im)
    current_image = []
    folder = (im.split("\\")[-2:])[0]
    imageNo = (im.split("\\")[-1:])[0]
    img = Image.open(im)
    img_hash = dhash(img)
    current_image.append(folder)
    current_image.append(imageNo)
    current_image.append(img_hash)  
    
    image_properties.append(current_image)

   

image_properties_df = pd.DataFrame(image_properties, columns=columns) 

image_properties_file = 'C:/Users/SriPrav/Documents/R/21Rental/input/Image_Hash_01.csv'
image_properties_df.to_csv(image_properties_file, index=False)


#          
#for im1 in images:
#    #print(im)
#    current_image = []
#    folder1 = (im1.split("\\")[-2:])[0]
#    image1No = (im1.split("\\")[-1:])[0]
#    hash00 = imagehash.dhash(Image.open(im1))
#    for im2 in images02:
#        current_image02 = []
#        hash02 = imagehash.dhash(Image.open(im2))
#        feature = hash00 - hash02
#        if feature <7 and im1 != im2:
#            folder2 = (im2.split("\\")[-2:])[0]
#            image2No = (im2.split("\\")[-1:])[0]
#    
#            current_image02.append(folder1)
#            current_image02.append(image1No)
#            current_image02.append(folder2)
#            current_image02.append(image2No)
#        current_image.append(current_image02)
#    image_properties.append(current_image)
#    
##    break 
# 

######################################################################################

#import os
#import sys
#import operator
#import numpy as np
#import pandas as pd
#from scipy import sparse
#import xgboost as xgb
#from sklearn import model_selection, preprocessing, ensemble
#from sklearn.metrics import log_loss
#from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#import string
#import io
#from PIL import Image, ImageDraw, ImageFilter, ImageStat, ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
#import matplotlib.pyplot as plt
#%matplotlib inline
#import nltk
#import glob
#import imagehash, hashlib
#
#inDir = 'C:/Users/SriPrav/Documents/R/21Rental'
#
#image_hashs = inDir + "/input/Image_Hash_01.csv"
#image_hashs = pd.read_csv(image_hashs)
#
#imagelisting_df = pd.DataFrame( image_hashs['imageNo'].apply(lambda x: pd.Series(str(x).split('_'))))
#
#imagelisting_df.columns = ['listing_id', 'image','None']
#
#images_hashings = pd.concat([image_hashs, imagelisting_df['listing_id']], axis=1)
#
#image_properties_file = 'C:/Users/SriPrav/Documents/R/21Rental/input/Image_Hash_02.csv'
#images_hashings.to_csv(image_properties_file , index=False)

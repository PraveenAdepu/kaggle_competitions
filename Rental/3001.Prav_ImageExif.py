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

from PIL import Image
from PIL.ExifTags import TAGS

# import exifread


inDir = 'C:/Users/SriPrav/Documents/R/21Rental'

images = sorted(glob.glob(inDir + '/input/images/images/**/**.jpg')) # 696,137 , 6811957

image_properties = []

#im = "C:/Users/SriPrav/Documents/R/21Rental/input/images/images/6811957/6811957_33d08c8dc440c89bccc8d9889c5485a6.jpg"
#for im in images:
#    current_image = []
#    f = open(im, 'rb')
#    tags = exifread.process_file(f)
#    for tag in tags.keys():
#        #if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
#        print "Key: %s, value %s" % (tag, tags[tag])
    
    
def get_exif(fn):
    ret = {}
    i = Image.open(fn)
    info = i._getexif()
    if info is not None:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            ret[decoded] = value
        return ret
        
columns = ["folder","imageNo"#, "Artist"
           , "ColorSpace", "Contrast"
        ,"CustomRendered","DateTimeOriginal"#,"DigitalZoomRatio"
        ,"ExifImageHeight","ExifImageWidth","ExifOffset"
        ,"ExposureMode","Flash","LightSource","Make"
        ,"Sharpness"
        ,"SubjectDistanceRange","WhiteBalance","YCbCrPositioning","ISOSpeedRatings"
        ,"Orientation","Saturation","SceneCaptureType"#,"SceneType"
        ,"SensingMethod"]

for im in images:
    current_image = []
    current_imageInfo = get_exif(im)
    if current_imageInfo is not None:
        folder = (im.split("\\")[-2:])[0]
        imageNo = (im.split("\\")[-1:])[0]
#        print(current_imageinfo)
#        break
#        try:
#            Artist = current_imageInfo["Artist"]
#        except KeyError:
#            Artist = ""
        try:
            ColorSpace = current_imageInfo["ColorSpace"]
        except KeyError:
            ColorSpace = 0
        try:
            Contrast = current_imageInfo["Contrast"]
        except KeyError:
            Contrast = 0
        try:
            CustomRendered = current_imageInfo["CustomRendered"]
        except KeyError:
            CustomRendered = 0
        try:
            DateTimeOriginal = current_imageInfo["DateTimeOriginal"]
        except KeyError:
            DateTimeOriginal = u""
#        try:
#            DigitalZoomRatio = current_imageInfo["DigitalZoomRatio"]
#        except IndexError:
#            DigitalZoomRatio = ""
        try:
            ExifImageHeight = current_imageInfo["ExifImageHeight"]
        except KeyError:
            ExifImageHeight = 0
        try:
            ExifImageWidth = current_imageInfo["ExifImageWidth"]
        except KeyError:
            ExifImageWidth = 0
        try:
            ExifOffset = current_imageInfo["ExifOffset"]
        except KeyError:
            ExifOffset = 0
        try:
            ExposureMode = current_imageInfo["ExposureMode"]
        except KeyError:
            ExposureMode = 0
        try:
            Flash = current_imageInfo["Flash"]
        except KeyError:
            Flash = 0
        try:
            LightSource = current_imageInfo["LightSource"]
        except KeyError:
            LightSource = 0
        try:
            Make = current_imageInfo["Make"]
        except KeyError:
            Make = u""
        try:
            Sharpness = current_imageInfo["Sharpness"]
        except KeyError:
            Sharpness = 0
        try:
            SubjectDistanceRange = current_imageInfo["SubjectDistanceRange"]
        except KeyError:
            SubjectDistanceRange = 0
        try:
            WhiteBalance = current_imageInfo["WhiteBalance"]
        except KeyError:
            WhiteBalance = 0
        try:
            YCbCrPositioning = current_imageInfo["YCbCrPositioning"]
        except KeyError:
            YCbCrPositioning = 0
        try:
            ISOSpeedRatings = current_imageInfo["ISOSpeedRatings"]
        except KeyError:
            ISOSpeedRatings = 0
        try:
            Orientation = current_imageInfo["Orientation"]
        except KeyError:
            Orientation = 0
        try:
            Saturation = current_imageInfo["Saturation"]
        except KeyError:
            Saturation = 0
        try:
            SceneCaptureType = current_imageInfo["SceneCaptureType"]
        except KeyError:
            SceneCaptureType = 0
        try:
            SensingMethod = current_imageInfo["SensingMethod"]
        except KeyError:
            SensingMethod = 0

        
        current_image.append(folder)
        current_image.append(imageNo)
#        current_image.append(Artist)
        current_image.append(ColorSpace)
        current_image.append(Contrast)
        current_image.append(CustomRendered)
        current_image.append(DateTimeOriginal)
        current_image.append(ExifImageHeight)
        current_image.append(ExifImageWidth)
        current_image.append(ExifOffset)
        current_image.append(ExposureMode)
        current_image.append(Flash)
        current_image.append(LightSource)
        current_image.append(Make)
        current_image.append(Sharpness)
        current_image.append(SubjectDistanceRange)
        current_image.append(WhiteBalance)
        current_image.append(YCbCrPositioning)
        current_image.append(ISOSpeedRatings)
        current_image.append(Orientation)
        current_image.append(Saturation)
        current_image.append(SceneCaptureType)
        current_image.append(SensingMethod)
        
        image_properties.append(current_image)

image_properties_df = pd.DataFrame(image_properties, columns=columns) 

image_properties_file = 'C:/Users/SriPrav/Documents/R/21Rental/input/Image_Exif_02.csv'
image_properties_df.to_csv(image_properties_file, index=False)

            
#            
#for im in images:
#    #print(im)
#    current_image = []
#    folder = (im.split("\\")[-2:])[0]
#    imageNo = (im.split("\\")[-1:])[0]
#    img = Image.open(im)
#    width, height = img.size
#    sizebytes = os.path.getsize(im)
#    stats = ImageStat.Stat(img, mask=None)
#    extrema00 = stats.extrema[0][0]
#    extrema01 = stats.extrema[0][1]
#    try:
#        extrema10 = stats.extrema[1][0]
#    except IndexError:
#        extrema10 = 0
#    try:
#        extrema11 = stats.extrema[1][1]
#    except IndexError:
#        extrema11 = 0
#    try:
#        extrema20 = stats.extrema[2][0]
#    except IndexError:
#        extrema20 = 0
#    try:
#        extrema21 = stats.extrema[2][1]
#    except IndexError:
#        extrema21 = 0
#    
#    count00 = stats.count[0]
#    try:
#        count01 = stats.count[1]
#    except IndexError:
#        count01 = 0
#    try:
#        count02 = stats.count[2]
#    except IndexError:
#        count02 = 0
#    
#    sum00 = stats.sum[0]
#    
#    try:
#        sum01 = stats.sum[1]
#    except IndexError:
#        sum01 = 0
#    try:
#        sum02 = stats.sum[2]
#    except IndexError:
#        sum02 = 0
#    
#    sum200 = stats.sum2[0]
#    try:
#        sum201 = stats.sum2[1]
#    except IndexError:
#        sum201 = 0
#    try:
#        sum202 = stats.sum2[2]
#    except IndexError:
#        sum202 = 0
#    
#    mean00 = stats.mean[0]
#    try:
#        mean01 = stats.mean[1]
#    except IndexError:
#        mean01 = 0
#    try:
#        mean02 = stats.mean[2]
#    except IndexError:
#        mean02 = 0
#    
#    median00 = stats.median[0]
#    try:
#        median01 = stats.median[1]
#    except IndexError:
#        median01 = 0
#    try:
#        median02 = stats.median[2]
#    except IndexError:
#        median02 = 0
#    
#    rms00 = stats.rms[0]
#    try:
#        rms01 = stats.rms[1]
#    except IndexError:
#        rms01 = 0
#    
#    try:
#        rms02 = stats.rms[2]
#    except IndexError:
#        rms02 = 0
#    
#    var00 = stats.var[0]
#    try:
#        var01 = stats.var[1]
#    except IndexError:
#        var01 = 0
#    try:
#        var02 = stats.var[2]
#    except IndexError:
#        var02 = 0
#    
#    stddev00 = stats.stddev[0]
#    try:
#        stddev01 = stats.stddev[1]
#    except IndexError:
#        stddev01 = 0
#    try:
#        stddev02 = stats.stddev[2]
#    except IndexError:
#        stddev02 = 0
#    
#  
#    current_image.append(folder)
#    current_image.append(imageNo)
#    current_image.append(width)
#    current_image.append(height)
#    current_image.append(sizebytes)
#    current_image.append(extrema00)
#    current_image.append(extrema01)
#    current_image.append(extrema10)
#    current_image.append(extrema11)
#    current_image.append(extrema20)
#    current_image.append(extrema21)
#    current_image.append(count00)
#    current_image.append(count01)
#    current_image.append(count02)
#    
#    current_image.append(sum00)
#    current_image.append(sum01)
#    current_image.append(sum02)
#    
#    current_image.append(mean00)
#    current_image.append(mean01)
#    current_image.append(mean02)
#    
#    current_image.append(median00)
#    current_image.append(median01)
#    current_image.append(median02)
#    
#    current_image.append(rms00)
#    current_image.append(rms01)
#    current_image.append(rms02)
#    
#    current_image.append(var00)
#    current_image.append(var01)
#    current_image.append(var02)
#    
#    current_image.append(stddev00)
#    current_image.append(stddev01)
#    current_image.append(stddev02)
##    break 
#    image_properties.append(current_image)
#
#image_properties_df = pd.DataFrame(image_properties, columns=columns) 
#
#image_properties_file = 'C:/Users/SriPrav/Documents/R/21Rental/input/Image_Exif_01.csv'
#image_properties_df.to_csv(image_properties_file, index=False)

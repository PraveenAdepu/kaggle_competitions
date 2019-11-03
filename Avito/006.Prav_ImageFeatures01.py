# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 18:27:34 2018

@author: SriPrav
"""


from collections import defaultdict
from scipy.stats import itemfreq
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image as IMG
import numpy as np
import pandas as pd 
import operator
import cv2
import os 

from IPython.core.display import HTML 
from IPython.display import Image


inDir = 'C:/Users/SriPrav/Documents/R/48Avito'

images_path = inDir +'/input/test_jpg.zip'
imgs = os.listdir(images_path)



import zipfile

NUM_IMAGES_TO_EXTRACT = 10

with zipfile.ZipFile(inDir +'/input/test_jpg.zip', 'r') as train_zip:
    files_in_zip = sorted(train_zip.namelist())
    for idx, file in enumerate(files_in_zip[:NUM_IMAGES_TO_EXTRACT]):
        if file.endswith('.jpg'):
            train_zip.extract(file, path=file.split('/')[3])
            
features = pd.DataFrame()
features['image'] = files_in_zip[1:]

pool = 'avg' # one of max of avg
batch_size = 64
im_dim = 96
n_channels = 3
limit = None # Limit number of images processed (useful for debug)
resize_mode = 'fit' # One of fit or crop
bar_iterval = 10 # in seconds
empty_im = np.zeros((im_dim, im_dim, n_channels), dtype=np.uint8) # Used when no image is present

def resize_fit(im, inter=cv2.INTER_AREA):
    height, width, _ = im.shape
    
    if height > width:
        new_dim = (width*im_dim//height, im_dim)
    else:
        new_dim = (im_dim, height*im_dim//width)
        
    imr = cv2.resize(im, new_dim, interpolation=inter)
    
    h, w = imr.shape[:2]

    off_x = (im_dim-w)//2
    off_y = (im_dim-h)//2
    
    im_out = np.zeros((im_dim, im_dim, n_channels), dtype=imr.dtype)

    im_out[off_y:off_y+h, off_x:off_x+w] = imr
    
    del imr
    
    return im_out


def resize_crop(im, inter=cv2.INTER_AREA):
    height, width, _ = im.shape
    
    if height > width:
        offy = (height-width) // 2
        imc = im[offy:offy+width]
    else:
        offx = (width-height) // 2
        imc = im[:, offx:offx+height]
        
    return cv2.resize(imc, (im_dim, im_dim), interpolation=inter)


def resize(im, inter=cv2.INTER_AREA):
    if resize_mode == 'fit':
        return resize_fit(im, inter)
    else:
        return resize_crop(im, inter)
    
def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent
zfile = 'data/competition_files/test_jpg/00011bcb5536797f5f4f33783201700bc436c30b1116842e8b6107a4d9c853b6.jpg'
train_zip = zipfile.ZipFile(images_path)
zinfo = train_zip.getinfo(zfile)
zbuf = np.frombuffer(train_zip.read(zinfo), dtype='uint8')
im = IMG.frombuffer(train_zip.read(zinfo))
im = resize( cv2.imdecode(zbuf, cv2.IMREAD_COLOR) )
im = np.array(im)           
def perform_color_analysis(img, flag):
    path = images_path + img 
    im = IMG.open(path) #.convert("RGB")
    
    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 
    
    if flag == 'black':
        return dark_percent
    elif flag == 'white':
        return light_percent
    else:
        return None
    
features['whiteness'] = features['image'].apply(lambda x : perform_color_analysis(x, 'white'))
topdull = features.sort_values('whiteness', ascending = False)
topdull.head(5)

def average_pixel_width(img):
    path = images_path + img 
    im = IMG.open(path)    
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100

def get_dominant_color(img):
    path = images_path + img 
    img = cv2.imread(path)
    arr = np.float32(img)
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(img.shape)

    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
    return dominant_color

features['dominant_color'] = features['image'].apply(get_dominant_color)
features.head(10)

features['dominant_red'] = features['dominant_color'].apply(lambda x: x[0]) / 255
features['dominant_green'] = features['dominant_color'].apply(lambda x: x[1]) / 255
features['dominant_blue'] = features['dominant_color'].apply(lambda x: x[2]) / 255
features[['dominant_red', 'dominant_green', 'dominant_blue']].head(5)

def get_average_color(img):
    path = images_path + img 
    img = cv2.imread(path)
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    return average_color

features['average_color'] = features['image'].apply(get_average_color)
features.head(10)

features['average_red'] = features['average_color'].apply(lambda x: x[0]) / 255
features['average_green'] = features['average_color'].apply(lambda x: x[1]) / 255
features['average_blue'] = features['average_color'].apply(lambda x: x[2]) / 255
features[['average_red', 'average_green', 'average_blue']].head(5)

def getSize(filename):
    filename = images_path + filename
    st = os.stat(filename)
    return st.st_size

def getDimensions(filename):
    filename = images_path + filename
    img_size = IMG.open(filename).size
    return img_size 

features['image_size'] = features['image'].apply(getSize)
features['temp_size'] = features['image'].apply(getDimensions)
features['width'] = features['temp_size'].apply(lambda x : x[0])
features['height'] = features['temp_size'].apply(lambda x : x[1])
features = features.drop(['temp_size', 'average_color', 'dominant_color'], axis=1)
features.head()

def get_blurrness_score(image):
    path =  images_path + image 
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm

features['blurrness'] = features['image'].apply(get_blurrness_score)
features[['image','blurrness']].head(5)

tempdf = features.sort_values('blurrness')
for y,x in tempdf.head(5).iterrows():
    path = images_path + x['image']
    html = "<h4>Image : "+x['image']+" &nbsp;&nbsp;&nbsp; (Blurrness : " + str(x['blurrness']) +")</h4>"
    display(HTML(html))
    display(IMG.open(path).resize((300,300), IMG.ANTIALIAS))

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
from tqdm import tqdm_notebook
import concurrent.futures
from multiprocessing import cpu_count
from sklearn.cross_validation import train_test_split
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.models import Model
from ipywidgets import IntProgress

from keras.utils import np_utils

import time
import glob
import math

from tqdm import tqdm
from PIL import Image
from io import BytesIO
import jpeg4py as jpeg
import random

import warnings
warnings.filterwarnings("ignore")
from sklearn.utils import shuffle


from keras.applications.resnet50 import ResNet50
inDir = 'C:/Users/SriPrav/Documents/R/43IEEE'


#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg19 import VGG19
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
               


images_train1 = pd.read_csv(inDir + '/input/100Prav_5folds_CVindices.csv')
images_train1.groupby("image_category").size()
images_train2 = pd.read_csv(inDir + '/input/100Prav_external_good_jpgs.csv')

images_train2 = images_train2[images_train2['image_rank']<=500]

images_train_externaldata = pd.read_csv(inDir + '/input/100images_train_externaldata.csv')
images_train_externaldata['image_rank'] = images_train_externaldata.groupby(['image_category']).cumcount()+1
images_train_externaldata.groupby("image_category").size()


images_train3_Moto = images_train_externaldata[(images_train_externaldata['image_category']=="Motorola-X") & (images_train_externaldata['image_rank']<=500)]
images_train3_LG = images_train_externaldata[(images_train_externaldata['image_category']=="LG-Nexus-5x") & (images_train_externaldata['image_rank']<=95)]
images_train3_SAM = images_train_externaldata[(images_train_externaldata['image_category']=="Samsung-Galaxy-Note3") & (images_train_externaldata['image_rank']<=226)]

images_train = pd.concat([images_train1, images_train2, images_train3_Moto,images_train3_LG,images_train3_SAM])
images_train = images_train.reset_index(drop=True)

# Every category got 525 images, balanced classes
images_train.groupby("image_category").size()


images_val = pd.read_csv(inDir + '/input/100val_images_extradata.csv')
images_val.groupby("image_category").nunique()



images_train['image_rank'] = images_train.groupby(['image_category']).cumcount()+1
images_train_balanced = images_train[images_train['image_rank']<=750]

images_train_balanced_valid = images_train[images_train['image_rank'] > 500]

images_train_balanced_2 = images_train_balanced_valid[ (images_train_balanced_valid['image_category']=="LG-Nexus-5x") & (images_train_balanced_valid['image_rank'] <= 503)]
images_train_balanced_3 = images_train_balanced_valid[ (images_train_balanced_valid['image_category']=="Motorola-Droid-Maxx") & (images_train_balanced_valid['image_rank'] <= 526)]
images_train_balanced_4 = images_train_balanced_valid[ (images_train_balanced_valid['image_category']=="Motorola-Nexus-6") & (images_train_balanced_valid['image_rank'] <= 512)]
images_train_balanced_5 = images_train_balanced_valid[ (images_train_balanced_valid['image_category']=="Motorola-X") & (images_train_balanced_valid['image_rank'] <= 510)]
images_train_balanced_6 = images_train_balanced_valid[ (images_train_balanced_valid['image_category']=="Samsung-Galaxy-Note3") & (images_train_balanced_valid['image_rank'] <= 522)]
images_train_balanced_9 = images_train_balanced_valid[ (images_train_balanced_valid['image_category']=="iPhone-4s") & (images_train_balanced_valid['image_rank'] <= 509)]
images_train_balanced_10 = images_train_balanced_valid[ (images_train_balanced_valid['image_category']=="iPhone-6") & (images_train_balanced_valid['image_rank'] <= 510)]


images_val_balanced = pd.concat([images_val, images_train_balanced_2,images_train_balanced_3,images_train_balanced_4,images_train_balanced_5,images_train_balanced_6,images_train_balanced_9,images_train_balanced_10])
images_val_balanced = images_val_balanced.reset_index(drop=True)

images_val.groupby("image_category").size()
images_val_balanced.groupby("image_category").size()

images_val_balanced = shuffle(images_val_balanced)
images_val_balanced['image_rank'] = images_val_balanced.groupby(['image_category']).cumcount()+1
images_val_balanced = images_val_balanced[images_val_balanced['image_rank']<=40]


images_train_balanced
images_val_balanced

images_train_balanced.groupby("image_category").size()
images_val_balanced.groupby("image_category").size()

images_train_balanced.to_csv(inDir+"/images_train_balanced.csv", index=False)
images_val_balanced.to_csv(inDir+"/images_val_balanced.csv", index=False)

images_val = pd.read_csv(inDir + '/input/images_val_balanced.csv')
images_val.groupby(['CVindices','image_category'])[['image_category']].size()
images_val.head()

images_val['image_category_lower'] = images_val['image_category'].str.lower()
images_val = images_val.sort_values('image_category_lower')

del images_val['image_category_lower']
y, rev_labels = pd.factorize(images_val['image_category'])
images_val['y'] = y

images_train = pd.read_csv(inDir + '/input/images_train_balanced.csv')
images_train.groupby(['CVindices','image_category'])[['image_category']].size()
images_train.head()

images_train['image_category_lower'] = images_train['image_category'].str.lower()
images_train = images_train.sort_values('image_category_lower')

del images_train['image_category_lower']
y, rev_labels = pd.factorize(images_train['image_category'])
images_train['y'] = y



test = pd.read_csv(inDir+'/input/images_test_v2.csv')
test.head()


num_classes = 10  # This will reduce the max accuracy to about 0.75
CHANNELS = 3
ROWS = 224
COLUMNS = 224
nb_epoch = 20
VERBOSEFLAG = 2
batch_size  = 16
patience = 10
optim_type = 'Adam'
learning_rate = 1e-3
crop_size = 224
#path = 'C:\\Users\\SriPrav\\Documents\\R\\43IEEE\\input\\train_2\\HTC-1-M7\\(HTC-1-M7)1_0.jpg'
# 'jpg70', 'jpg90',
MANIPULATIONS = ['gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0']

def get_crop(img, crop_size, random_crop=False):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    half_crop = crop_size // 2
    pad_x = max(0, crop_size - img.shape[1])
    pad_y = max(0, crop_size - img.shape[0])
    if (pad_x > 0) or (pad_y > 0):
        img = np.pad(img, ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2), (0,0)), mode='wrap')
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    if random_crop:
        freedom_x, freedom_y = img.shape[1] - crop_size, img.shape[0] - crop_size
        if freedom_x > 0:
            center_x += np.random.randint(math.ceil(-freedom_x/2), freedom_x - math.floor(freedom_x/2) )
        if freedom_y > 0:
            center_y += np.random.randint(math.ceil(-freedom_y/2), freedom_y - math.floor(freedom_y/2) )

    return img[center_y - half_crop : center_y + crop_size - half_crop, center_x - half_crop : center_x + crop_size - half_crop]

def random_manipulation(img, manipulation=None):

    if manipulation == None:
        manipulation = random.choice(MANIPULATIONS)

    if manipulation.startswith('jpg'):
        quality = int(manipulation[3:])
        out = BytesIO()
        im = Image.fromarray(img)
        im.save(out, format='jpeg', quality=quality)
        im_decoded = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()
        del out
        del im
    elif manipulation.startswith('gamma'):
        gamma = float(manipulation[5:])
        # alternatively use skimage.exposure.adjust_gamma
        # img = skimage.exposure.adjust_gamma(img, gamma)
        im_decoded = np.uint8(cv2.pow(img / 255., gamma)*255.)
    elif manipulation.startswith('bicubic'):
        scale = float(manipulation[7:])
        im_decoded = cv2.resize(img,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    else:
        assert False
    return im_decoded

def get_im_cv2(path,use_manipulation=True,crop_size=224,random_crop=True):
    img = cv2.imread(path)
    if use_manipulation and np.random.rand() < 0.5:
        img = random_manipulation(img, manipulation=None)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = get_crop(img,crop_size, random_crop=True)
    return img

def get_im_cv2_test(path):
    img = cv2.imread(path)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

#def normalize_image(img):
#    img = np.array(img, dtype=np.uint8)
#    img= img.astype('float32')
#    kernel_filter = 1/12. * np.array([\
#            [-1,  2,  -2,  2, -1],  \
#            [ 2, -6,   8, -6,  2],  \
#            [-2,  8, -12,  8, -2],  \
#            [ 2, -6,   8, -6,  2],  \
#            [-1,  2,  -2,  2, -1]]) 
#
#    return cv2.filter2D(img,-1,kernel_filter)
    
def normalize_image(x):
    x = np.array(x, dtype=np.uint8)
    #x=x.transpose((0,1,2,3))
    x= x.astype('float32')
    # Subtract ImageNet mean pixel 
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68
#    x = x / 255
#    x -= 0.5
#    x *= 2.
    return x
    
def load_train_frombatch(images_batch):
    X_train = []

    for fl in images_batch.image_path.values:
#        print(fl)
        #img = get_crop(fl, crop_size=224, random_crop=True)        
        img = get_im_cv2(fl,use_manipulation=True,crop_size=224,random_crop=True)        
        X_train.append(img)

    return X_train


def load_test_frombatch(images_batch):
    X_test = []
    for fl in images_batch.image_path.values:        
        img = get_im_cv2_test(fl)        
        X_test.append(img)
    return X_test




def batch_generator_X_build(images_build,X_build, y_build ,batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    
    number_of_batches = np.ceil(len(images_build)/batch_size)
    counter = 0
    sample_index = images_build.index.get_values()
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_build[list(batch_index)]
        y_batch = y_build[list(batch_index)]       
        
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0 
            
def batch_generator_X_valid(images_valid,X_valid, y_valid ,batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(len(images_valid)/batch_size)
    counter = 0
    sample_index = images_valid.index.get_values()
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_valid[list(batch_index)]
        y_batch = y_valid[list(batch_index)]       
        
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

            

def model_ResNet50(num_classes):
    base_model = ResNet50(weights='imagenet')

    x = base_model.output
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=x)    
    return model


 
ModelName= 'Resnet_11'
i=1

MODEL_WEIGHTS_FILE = inDir + '/Prav_01_Resnet11_'+str(i)+'.h5'
print('Fold ', i , ' Processing')


images_build = images_train
images_valid = images_val

images_build = shuffle(images_build)
images_valid = shuffle(images_valid)

########################################################################################################
#from multiprocessing import pool
#from multiprocessing.dummy import Pool as ThreadPool
#
#from functools import partial
#from itertools import  islice
#from conditional import conditional
#
## Create thread pool
#pool = ThreadPool(2)
#
#
##import gc
##gc.collect()
#
#if __name__ == '__main__':
#    get_im_cv2_func  = partial(get_im_cv2)       
#    X_build = pool.map( get_im_cv2_func, images_build["image_path"])
########################################################################################################

model = model_ResNet50(num_classes=num_classes)

save_checkpoint = ModelCheckpoint(
            MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True, verbose=VERBOSEFLAG, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.25, patience=10, min_lr=1e-9, epsilon = 0.00001, verbose=1, mode='max')
learning_rate = 1e-3
if optim_type == 'SGD':
    optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
else:
    optim = Adam(lr=learning_rate)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'])


    
for i in range(1,100):
    print("Processing Iteration {}".format(i))
    
    if i !=1:
        print("build and valid deleted")
        del X_build, y_build, X_valid, y_valid
    
    print("Preparing X_build")
    X_build = load_train_frombatch(images_build)
    X_build = normalize_image(X_build)
    y_build = np_utils.to_categorical(images_build['y'],10)
    
    print("Preparing X_valid")
    X_valid = load_train_frombatch(images_valid)
    X_valid = normalize_image(X_valid)
    y_valid = np_utils.to_categorical(images_valid['y'],10)
    
    nb_epoch = 20 * i
    initial_epoch = (20 * i) - 20  
       
    print("Model training")
    model.fit_generator( generator=batch_generator_X_build(images_build,X_build, y_build, batch_size, shuffle=True),
                             #samples_per_epoch = len(build_index), 
                             steps_per_epoch = math.ceil(len(images_build) / batch_size), #int(len(build_index)/float(batch_size)),
                             initial_epoch = initial_epoch,
                             nb_epoch = nb_epoch, 
                             callbacks = [save_checkpoint, reduce_lr],
                             validation_data=batch_generator_X_valid(images_valid,X_valid,y_valid, batch_size, shuffle=True), 
                             #nb_val_samples=len(valid_index), 
                             validation_steps = math.ceil(len(images_valid) / batch_size), #int(len(valid_index)/float(batch_size)),
                             max_q_size=10,
                             verbose = VERBOSEFLAG 
                  )
    
    
    

model.load_weights(MODEL_WEIGHTS_FILE)

del X_build, y_build, X_valid, y_valid

X_valid_data = load_test_frombatch(images_valid)
X_valid_data = normalize_image(X_valid_data)

X_test = load_test_frombatch(test)
X_test = normalize_image(X_test)

pred_cv = np.zeros([images_valid.shape[0],10])    
pred_test = np.zeros([X_test.shape[0],10])


#bag_cv  += model.predict(X_valid_data, batch_size=batch_size, verbose=VERBOSEFLAG)        
pred_cv += model.predict(X_valid_data, batch_size=batch_size, verbose=VERBOSEFLAG)        
pred_test += model.predict(X_test, batch_size=batch_size, verbose=VERBOSEFLAG)

pred_cv = pd.DataFrame(pred_cv)
pred_cv.columns = rev_labels
pred_cv["fname"] = images_valid.image_name.values
pred_cv["fname_patch"] = images_valid.image_name_patch.values
pred_cv = pred_cv[["fname","fname_patch",'HTC-1-M7', 'iPhone-4s', 'iPhone-6', 'LG-Nexus-5x',
   'Motorola-Droid-Maxx', 'Motorola-Nexus-6', 'Motorola-X',
   'Samsung-Galaxy-Note3', 'Samsung-Galaxy-S4', 'Sony-NEX-7']]
sub_valfile = inDir+'/submissions/Prav.'+ModelName+'.fold' + str(i) + '.csv'    
pred_cv.to_csv(sub_valfile, index=False)

pred_test = pd.DataFrame(pred_test)
pred_test.columns = rev_labels
pred_test["fname"] = test._id.values
pred_test["fname_patch"] = test._id1.values
pred_test = pred_test[["fname","fname_patch",'HTC-1-M7', 'iPhone-4s', 'iPhone-6', 'LG-Nexus-5x',
   'Motorola-Droid-Maxx', 'Motorola-Nexus-6', 'Motorola-X',
   'Samsung-Galaxy-Note3', 'Samsung-Galaxy-S4', 'Sony-NEX-7']]
sub_file = inDir+'/submissions/Prav.'+ModelName+'.fold' + str(i) + '-test' + '.csv'
pred_test.to_csv(sub_file, index=False)
del pred_cv
del pred_test
del model

    
    

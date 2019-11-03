# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import io
import bson
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import concurrent.futures
from multiprocessing import cpu_count
from sklearn.cross_validation import train_test_split
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.utils import np_utils
import math
import keras.backend as K

import time
import glob

from keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg19 import VGG19
#from keras.applications.inception_resnet_v2 import InceptionResNetV2

random_state = 2017

np.random.seed(random_state)
               
inDir = 'C:/Users/SriPrav/Documents/R/37TFSpeech'

MODEL_WEIGHTS_FILE = inDir + '/Prav_01_resnet50.h5'
#top_model_weights_path = inDir + "/input/vgg19_weights_th_dim_ordering_th_kernels_notop.h5"

images_test = pd.read_csv(inDir + '/input/images_test.csv')
images_test['image_id'] = images_test['_id']

images_test.image_id.replace('.png', '.wav', regex=True, inplace= True)

images_train = pd.read_csv(inDir + '/input/Prav_10folds_CVindices.csv')

images_train['image_category'].unique()
images_train.head()
#
#'''Transform data into something more useful.'''
#train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
#words = images_train['image_category'].unique().tolist()
#silence = ['_background_noise_']
#unknown = [w for w in words if w not in silence + train_words]
#
## there are only 6 silence files. Mark them as unknown too.
#images_train.loc[images_train.image_category.isin(silence), 'image_category'] = 'unknown'
#images_train.loc[images_train.image_category.isin(unknown), 'image_category'] = 'unknown'

images_train['image_category'].unique()
    
images_train.image_category[images_train.image_category == '_background_noise_'] = 'unknown'

train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','unknown']


images_train = images_train[images_train['image_category'].isin(train_words)]


y, rev_labels = pd.factorize(images_train['image_category'])

images_train['y'] = y

images_train_index = images_train.index.get_values()


build_index, valid_index,y_build, y_valid = train_test_split(images_train_index, y, test_size=0.1, random_state=random_state, stratify=y)

print(len(set(y_build))) # 5270
print(len(set(y_valid))) # 5270

images_build = images_train.ix[list(build_index)] # 9897034
images_valid = images_train.ix[list(valid_index)] # 2474259

#i = 10
#train = images_train[images_train['CVindices'] != i]
#valid = images_train[images_train['CVindices'] == i]

   

#import gc
#gc.collect()

num_classes = 11  # This will reduce the max accuracy to about 0.75
CHANNELS = 3
ROWS = 224
COLUMNS = 224
nb_epoch = 10
VERBOSEFLAG = 1
batch_size  = 32
patience = 2
optim_type = 'Adam'
learning_rate = 1e-4
#path = r'C:\Users\SriPrav\Documents\R\37TFSpeech\input\picts\train\left\b3327675_nohash_0.png'
def get_im_cv2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    resized = cv2.resize(img, (ROWS, COLUMNS), cv2.INTER_LINEAR)
    if (img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1] * 256 / img.shape[0]), 256)
    else:
        tile_size = (256, int(img.shape[0] * 256 / img.shape[1]))

    img = cv2.resize(img, dsize=(ROWS, COLUMNS),interpolation=cv2.INTER_AREA)
    return img
#img.shape
#plt.imshow(img)
#img = normalize_image(img)
#img = np.array(img, dtype=np.uint8)
#img=img.transpose((0,1,2,3))
#img= img.astype('float32')

#def normalize_image_resnet50(x):
#    x = np.array(x, dtype=np.uint8)
#    x=x.transpose((0,3,1,2))
#    x= x.astype('float32')
#    x[:, 0, :, :] -= 103.939
#    x[:, 1, :, :] -= 116.779
#    x[:, 2, :, :] -= 123.68
#    return x
#    
#def InceptionV3_preprocess_input(x):
#    x = np.array(x, dtype=np.uint8)
#    x=x.transpose((0,3,1,2))
#    x= x.astype('float32')
#    x /= 255.
#    x -= 0.5
#    x *= 2.
#    return x
    
def normalize_image(x):
    x = np.array(x, dtype=np.uint8)
    #x=x.transpose((0,1,2,3))
    x= x.astype('float32')
    # Subtract ImageNet mean pixel 
    #x[:, :, :, 0] -= 103.939
    #x[:, :, :, 1] -= 116.779
    #x[:, :, :, 2] -= 123.68
    x = x / 255
#    x -= 0.5
#    x *= 2.
    return x
    
def load_train_frombatch(images_batch):
    X_train = []
    y = []
    y_batch_target = []
    y= images_batch['y']
    y_batch_target = np_utils.to_categorical(y, num_classes)
#    start_time = time.time()

    for fl in images_batch.image_path.values:
#        print(fl)        
        img = get_im_cv2(fl)        
        X_train.append(img)
       
#    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_batch_target

    
def batch_generator_train(images_build ,batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(len(images_build)/batch_size)
    counter = 0
    sample_index = images_build.index.get_values()
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        images_build_batch = images_build.ix[list(batch_index)]
        X_batch, y_batch = load_train_frombatch(images_build_batch)
        X_batch = normalize_image(X_batch)
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0 
            
def batch_generator_valid(images_valid ,batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(len(images_valid)/batch_size)
    counter = 0
    sample_index = images_valid.index.get_values()
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        images_valid_batch = images_valid.ix[list(batch_index)]
        X_batch, y_batch = load_train_frombatch(images_valid_batch)
        X_batch = normalize_image(X_batch)
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0 

def load_test_frombatch(images_batch):
    X_test = []  
#    start_time = time.time()
    for fl in images_batch.image_path.values:
#        print(fl)        
        img = get_im_cv2(fl)        
        X_test.append(img)       
#    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test

def batch_generator_test(images_test ,batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(len(images_test)/batch_size)
    counter = 0
    sample_index = images_test.index.get_values()
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        images_test_batch = images_test.ix[list(batch_index)]
        X_batch = load_test_frombatch(images_test_batch)
        X_batch = normalize_image(X_batch)
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0 


#def model_vgg16(num_classes):
#    base_model = VGG16(include_top=False, weights='imagenet',input_shape=(ROWS, COLUMNS,CHANNELS))
#    x = base_model.output
#    x = Flatten(name='flatten')(x)
#    #x = Dense(num_classes, activation='relu', name='fc1')(x)
#    x = Dense(num_classes, activation='softmax', name='predictions')(x)
#    model = Model(input=base_model.input, output=x)
#    return model

#model = model_vgg16(num_classes=num_classes)
  
#def model_ResNet50(num_classes):
#    base_model = ResNet50(include_top=False, weights='imagenet',input_shape=( ROWS, COLUMNS,CHANNELS))
#    
#    for layer in base_model.layers:
#        layer.trainable = False
#    x = base_model.output
#    x = GlobalAveragePooling2D()(x)    
#    x = Dense(num_classes, activation='softmax', name='predictions')(x)
#    model = Model(input=base_model.input, output=x)
#    return model

def RESNET_50(num_classes):

    base_model = ResNet50(include_top=True, weights=None)
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=x)
    
    # print(model.summary())
    return model

model = RESNET_50(num_classes=num_classes) 

 
callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=VERBOSEFLAG),
        ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=VERBOSEFLAG),
                ]
#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
if optim_type == 'SGD':
    optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
else:
    optim = Adam(lr=learning_rate)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
   
model.fit_generator( generator=batch_generator_train(images_build, batch_size, shuffle=False),
                             #samples_per_epoch = len(build_index), 
                             steps_per_epoch = int(len(build_index)/float(batch_size)),
                             nb_epoch = nb_epoch, 
                             callbacks = callbacks,
                             validation_data=batch_generator_valid(images_valid, batch_size, shuffle=False), 
                             #nb_val_samples=len(valid_index), 
                             validation_steps = int(len(valid_index)/float(batch_size)),
                             max_q_size=10,
                             verbose = VERBOSEFLAG 
                  )

model.load_weights(MODEL_WEIGHTS_FILE)


pred_test = model.predict_generator(generator=batch_generator_test(images_test, batch_size, shuffle=False)
                                     , val_samples=int(images_test.shape[0]/float(batch_size))
                                     , verbose = VERBOSEFLAG)
pred_test = pd.DataFrame(pred_test)
pred_test.columns = rev_labels #["slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]

pred_test['label'] = pred_test.idxmax(axis=1)

pred_test["fname"] = images_test['image_id']


sub_file = inDir + '/submissions/Prav.resnet50_01-test'+'.csv'
pred_test[["fname","label"]].to_csv(sub_file, index=False) 


## Keep training for as long as you like.
#for i in range(1):
#    # gradually decrease the learning rate
#    K.set_value(model.optimizer.lr, 0.95 * K.get_value(model.optimizer.lr))
#    start_epoch = (i * 2)
#    epochs = ((i + 1) * 2)    
#    model.fit_generator(generator=batch_generator_train(images_build, batch_size, shuffle=False),
#                        steps_per_epoch=math.ceil(2000000 / batch_size),
#                        verbose=1,
#                        callbacks=callbacks,
#                        validation_data=batch_generator_valid(images_valid, batch_size, shuffle=False), 
#                        initial_epoch=start_epoch ,
#                        epochs=epochs ,
#                        use_multiprocessing=True,
#                        max_queue_size=10,
#                        workers = 8,
#                        validation_steps=math.ceil(10000 / batch_size))
    
num_cpus = 24
#def normalize_image_resnet50_test(x):
#    
#    x=x.transpose((0,3,1,2))
#    x= x.astype('float32')
#    x[:, 0, :, :] -= 103.939
#    x[:, 1, :, :] -= 116.779
#    x[:, 2, :, :] -= 123.68
#    return x
    
def imread(buf):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_ANYCOLOR)

def img2feat(im):
    x = cv2.resize(im, (ROWS, COLUMNS), interpolation=cv2.INTER_AREA)
    x = normalize_image(x)
    return x

def load_image(pic, target, bar):
    picture = imread(pic)
    x = img2feat(picture)
    bar.update()
    
    return x, target
    
   
submission = pd.read_csv(inDir + '/input/sample_submission.csv', index_col='_id')

most_frequent_guess = 1000018296
submission['category_id'] = most_frequent_guess # Most frequent guess


num_images_test = 1768182  # We only have time for a few test images..

bar = tqdm_notebook(total=num_images_test * 2)
with open(inDir+'/input/test.bson', 'rb') as f, \
         concurrent.futures.ThreadPoolExecutor(num_cpus) as executor:

    data = bson.decode_file_iter(f)

    future_load = []
    
    for i,d in enumerate(data):
        if i >= num_images_test:
            break
        future_load.append(executor.submit(load_image, d['imgs'][0]['picture'], d['_id'], bar))

    print("Starting future processing")
    for future in concurrent.futures.as_completed(future_load):
        x, _id = future.result()
        
        y_cat = rev_labels[np.argmax(model.predict(x[None])[0])]
        if y_cat == -1:
            y_cat = most_frequent_guess

        bar.update()
        submission.loc[_id, 'category_id'] = y_cat
print('Finished')

submission.to_csv(inDir+'/submissions/Prav_01_resnet50.csv')



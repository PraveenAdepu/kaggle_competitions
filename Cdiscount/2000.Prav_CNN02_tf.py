# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
                 
inDir = 'C:/Users/SriPrav/Documents/R/32Cdiscount'
MODEL_WEIGHTS_FILE = inDir + '/Prav_CNN02_tf.h5'

num_images = 7069896
im_size = 32
num_cpus = 24

def imread(buf):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_ANYCOLOR)

def img2feat(im):
    x = cv2.resize(im, (im_size, im_size), interpolation=cv2.INTER_AREA)
    return x
    #return np.float32(x) / 255

X = []
y = []

def load_image(pic, target, bar):
    picture = imread(pic)
    x = img2feat(picture)
    bar.update()
    
    return x, target

bar = tqdm_notebook(total=num_images)
with open(inDir + '/input/train.bson', 'rb') as f, \
        concurrent.futures.ThreadPoolExecutor(num_cpus) as executor:
    
    data = bson.decode_file_iter(f)
    delayed_load = []

    i = 0
    try:
        for c, d in enumerate(data):
            target = d['category_id']
            for e, pic in enumerate(d['imgs']):
                delayed_load.append(executor.submit(load_image, pic['picture'], target, bar))
                
                i = i + 1

                if i >= num_images:
                    raise IndexError()

    except IndexError:
        pass;
    
    for i, future in enumerate(concurrent.futures.as_completed(delayed_load)):
        x, target = future.result()
        
        X.append(x)
        y.append(target)


X = np.array(X, dtype=np.uint8)
       
X.shape, len(y)
#X = X.transpose((0, 3, 1, 2))
y = pd.Series(y)
print(len(set(y)))
num_classes = 5270  # This will reduce the max accuracy to about 0.75

#import gc
#gc.collect()
# Now we must find the most `num_classes-1` frequent classes
# (there will be an aditional 'other' class)
valid_targets = set(y.value_counts().index[:num_classes-1].tolist())
valid_y = y.isin(valid_targets)

# Set other classes to -1
y[~valid_y] = -1

max_acc = valid_y.mean()
print(max_acc)

# Now we categorize the dataframe
y, rev_labels = pd.factorize(y)

# Train a simple NN


CHANNELS = 3
ROWS = 32
COLUMNS = 32
random_state = 2017
nb_epoch = 5
VERBOSEFLAG = 1
batch_size  = 1024
patience = 2


train_datagen = ImageDataGenerator(        
                                  
                                  )

valid_datagen = ImageDataGenerator(
                                  
                                   )

test_datagen = ImageDataGenerator(        
                                
                                 )


#model = Sequential()
#
#model.add(Conv2D(8, 3,3, activation='relu', input_shape=(CHANNELS,ROWS, COLUMNS)))
#model.add(Conv2D(8, 3,3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
#
#model.add(Conv2D(16, 3,3, activation='relu'))
#model.add(Conv2D(16, 3,3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
#
#model.add(Conv2D(32, 3,3, activation='relu'))
#model.add(Conv2D(32, 3,3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
#
##model.add(Conv2D(64, 3,3, activation='relu'))
##model.add(Conv2D(64, 3,3, activation='relu'))
##model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
#
#model.add(Flatten())
#model.add(Dense(num_classes, activation='relu'))
#model.add(Dense(num_classes, activation='softmax'))
#
#
#opt = Adam(lr=0.01)
#
#model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
#
#model.summary()


model = Sequential()
model.add(Conv2D(16, 3, activation='relu', input_shape=(ROWS, COLUMNS,CHANNELS)))
model.add(Conv2D(16, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, 3, activation='relu'))
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(num_classes, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


opt = Adam(lr=0.01)

model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


X_build, X_valid, y_build, y_valid = train_test_split( X, y, test_size=0.2, random_state=random_state, stratify=y)


print(len(set(y_build)))
print(len(set(y_valid)))

del X
del y

X_build = X_build.astype('float32')
X_build = X_build / 255

X_valid = X_valid.astype('float32')
X_valid = X_valid / 255

callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=VERBOSEFLAG),
        ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=VERBOSEFLAG),
                ]
   
model.fit_generator( train_datagen.flow( X_build, y_build, batch_size = batch_size,shuffle=True),
                             samples_per_epoch = len(X_build), nb_epoch = nb_epoch, callbacks = callbacks,
                             validation_data=valid_datagen.flow(X_valid, y_valid, batch_size=batch_size), 
                             nb_val_samples=X_valid.shape[0], verbose = VERBOSEFLAG )
#    min_loss = min(model.model['val_loss'])
#    print('Minimum loss for given fold: ', min_loss)
model.load_weights(MODEL_WEIGHTS_FILE)
    
        
#model.fit(X, y, validation_split=0.1, nb_epoch=2)
#
#model.save_weights(inDir+'/model.h5')

def img2feat(im):
    x = cv2.resize(im, (im_size, im_size), interpolation=cv2.INTER_AREA)
#    return x
    return np.float32(x) / 255
    
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
        
        y_cat = rev_labels[np.argmax(model.predict(x[None].transpose((0, 3, 1, 2)))[0])]
        if y_cat == -1:
            y_cat = most_frequent_guess

        bar.update()
        submission.loc[_id, 'category_id'] = y_cat
print('Finished')

submission.to_csv(inDir+'/submissions/Prav_CNN02.csv')

#
#test = np.empty((num_images_test, im_size, im_size, 3), dtype=np.float32)
#test_id = []
#
#bar = tqdm_notebook(total=num_images_test*2)
#with open(inDir + '/input/test.bson', 'rb') as f, \
#        concurrent.futures.ThreadPoolExecutor(num_cpus) as executor:
#    
#    testdata = bson.decode_file_iter(f)
#    delayed_load = []
#
#    i = 0
#    try:
#        for c, d in enumerate(testdata):
#            target = d['_id']
#            for e, pic in enumerate(d['imgs']):
#                delayed_load.append(executor.submit(load_image, pic['picture'], target, bar))
#                
#                i = i + 1
#
#                if i >= num_images_test:
#                    raise IndexError()
#
#    except IndexError:
#        pass;
#    
#    for i, future in enumerate(concurrent.futures.as_completed(delayed_load)):
#        x, target = future.result()
#        
#        test[i] = x
#        test_id.append(target)
#
#test = test.transpose((0, 3, 1, 2))
#y_cat=[]
#
#for i in range(0, len(test)):
#    test_present = test[i]
#    y_pred = rev_labels[np.argmax(model.predict(test_present[None])[0])]
#    y_cat.append(y_pred)
#
#final_sub = pd.DataFrame(list(zip(test_id, y_cat)),
#              columns=['_id','category_id'])
#
#final_sub_order = final_sub.sort_values(['_id'])
#
#final_sub_order.to_csv(inDir+'/submissions/Prav_CNN01.csv', index=False)


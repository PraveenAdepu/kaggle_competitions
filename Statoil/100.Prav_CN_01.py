# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 19:13:34 2017

@author: SriPrav
"""

import numpy as np # linear algebra
random_state = 201804
np.random.seed(random_state)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import train_test_split
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D, Input
import os
from sklearn.model_selection import StratifiedKFold

inDir = 'C:/Users/SriPrav/Documents/R/35Statoil'

train_df = pd.read_json(inDir + "/input/train.json")
test_df = pd.read_json(inDir + "/input/test.json")

#trainfoldSource = train_df[['id','is_iceberg']]
#
#random_state = 2017
#np.random.seed(random_state)
#
#folds = 4
#skf = StratifiedKFold(n_splits=folds, random_state = random_state, shuffle = True)
#
#skf.get_n_splits(trainfoldSource, trainfoldSource['is_iceberg'])
#
#print(skf) 
#
#Prav_CVindices = pd.DataFrame(columns=['index','CVindices'])
#
#count = 1
#for train_index, test_index in skf.split(trainfoldSource, trainfoldSource['is_iceberg']):           
#       df_index = pd.DataFrame(test_index,columns=['index']) 
#       df_index['CVindices'] = count
#       Prav_CVindices = Prav_CVindices.append(df_index)       
#       count+=1
#       
#Prav_CVindices.set_index('index', inplace=True)
#
#trainfoldSource = pd.merge(trainfoldSource, Prav_CVindices, left_index=True, right_index=True)
#
#trainfoldSource.groupby(['CVindices'])[['is_iceberg']].sum()
#
#del trainfoldSource['is_iceberg']
#train_df[['id','CVindices']].to_csv(inDir+"/input/Prav_4folds_CVindices.csv", index=False)
#
#train_df = pd.merge(train_df, trainfoldSource, how='left',on="id")

trainfoldSource = pd.read_csv(inDir+'/input/Prav_4folds_CVindices.csv')
train_df = pd.merge(train_df, trainfoldSource, how='left',on="id")

train_df.groupby(['CVindices'])[['is_iceberg']].sum()


train_df.inc_angle = train_df.inc_angle.replace('na', 0)
train_df.inc_angle = train_df.inc_angle.astype(float).fillna(0.0)

test_df.inc_angle = test_df.inc_angle.replace('na', 0)
test_df.inc_angle = test_df.inc_angle.astype(float).fillna(0.0)

print("done!")

# Train data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])

X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis],((x_band1+x_band2)/2)[:, :, :, np.newaxis]], axis=-1)
X_angle_train = np.array(train_df.inc_angle)
y_train = np.array(train_df["is_iceberg"])
print("Xtrain:", X_train.shape)

# Test data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]])
X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis],((x_band1+x_band2)/2)[:, :, :, np.newaxis]], axis=-1)
X_angle_test = np.array(test_df.inc_angle)
print("Xtest:", X_test.shape)

def get_scaled_imgs(df):
    """
    basic function for reshaping and rescaling data as images
    """
    imgs = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)    

X_train = get_scaled_imgs(train_df)
y_train = np.array(train_df['is_iceberg'])
X_angle_train = np.array(train_df.inc_angle)

X_test = get_scaled_imgs(test_df)
X_angle_test = np.array(test_df.inc_angle)



MODEL_WEIGHTS_FILE = inDir + '/Prav_CN_01.h5'
ROWS     = 75
COLUMNS  = 75
CHANNELS = 3
nb_epoch = 100
VERBOSEFLAG = 1
batch_size  = 128
patience = 10
optim_type = 'Adam'
learning_rate = 1e-4

from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, concatenate
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet

# Define the image transformations here
gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         )

# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=55)
    genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=55)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield X1i[0], X1i[1]

# Finally create generator
def get_callbacks(filepath, patience=2):
   #es = EarlyStopping('val_loss', patience=10, mode="min")
   es = EarlyStopping('val_loss', patience=20, mode="min")
   msave = ModelCheckpoint(filepath, save_best_only=True)
   return [es, msave]


def get_model():
    
    """
    Keras Sequential model

    """
    
    model=Sequential()
    
    # Conv block 1
    model.add(Conv2D(64, 3,activation='relu', input_shape=(75, 75, 3)))
    model.add(Conv2D(64, 3, activation='relu' ))
    model.add(Conv2D(64, 3, activation='relu' ))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
   
    # Conv block 2
    model.add(Conv2D(128, 3, activation='relu' ))
    model.add(Conv2D(128, 3, activation='relu' ))
    model.add(Conv2D(128, 3, activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    # Conv block 3
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    #Conv block 4
    model.add(Conv2D(256, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    # Flatten before dense
    model.add(Flatten())

    #Dense 1
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))

    #Dense 2
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

#    optimizer = Adam(lr=0.0001, decay=0.0)
#    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def train_nn(i):
    
    trainindex = train_df[train_df['CVindices'] != i].index.tolist()
    valindex   = train_df[train_df['CVindices'] == i].index.tolist()
    
    X_val_df = train_df.iloc[valindex,:]
    
    X_build , X_valid = X_train[trainindex,:], X_train[valindex,:]
    X_angle_build , X_angle_valid = X_angle_train[trainindex], X_angle_train[valindex]
    y_build , y_valid = y_train[trainindex], y_train[valindex]   
    
    #model = get_model()
    model= get_model()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=VERBOSEFLAG),
        ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=VERBOSEFLAG),
                ]
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, momentum=0.9)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
#    early_stop = EarlyStopping(monitor='val_loss', patience=75, verbose=0)
#    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)
    gen_flow = gen_flow_for_two_inputs(X_build, X_angle_build, y_build)
    model.fit_generator(
                        gen_flow,
                        steps_per_epoch = int(len(trainindex)/float(batch_size)),
                        nb_epoch = nb_epoch, 
                        validation_data=(X_valid, y_valid),
                        callbacks=callbacks,
                        verbose = VERBOSEFLAG
                        )

    model.load_weights(MODEL_WEIGHTS_FILE)
    print("Train evaluate:")
    print(model.evaluate(X_build, y_build, verbose=1, batch_size=200))
    print("####################")
    print("watch list evaluate:")
    print(model.evaluate(X_valid, y_valid, verbose=1, batch_size=200))
    pred_cv = model.predict(X_valid, verbose=1, batch_size=200)

      
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.head()
    pred_cv.columns = ["is_iceberg"]
    pred_cv["id"] = X_val_df.id.values
    
    sub_valfile = inDir+'/submissions/Prav.CN01.fold' + str(i) + '.csv'
    pred_cv = pred_cv[["id","is_iceberg"]]
    pred_cv.to_csv(sub_valfile, index=False)
    pred_test = model.predict(X_test, verbose=1, batch_size=200)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["is_iceberg"]
    pred_test["id"] = test_df.id.values
    pred_test = pred_test[["id","is_iceberg"]]
    sub_file = inDir+'/submissions/Prav.CN01.fold' + str(i) + '-test' + '.csv'
    pred_test.to_csv(sub_file, index=False)
    if i != 4:
        os.remove(MODEL_WEIGHTS_FILE)
   
    del pred_cv
    del pred_test
    del model

folds = 4
if __name__ == '__main__':
    for i in range(1, folds+1):
        train_nn(i)
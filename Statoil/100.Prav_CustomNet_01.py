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
## Train data
#x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
#x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
#X_train = np.concatenate([x_band1[:, :, :, np.newaxis]
#                          , x_band2[:, :, :, np.newaxis]
#                         , ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)
#X_angle_train = np.array(train_df.inc_angle)
#y_train = np.array(train_df["is_iceberg"])
#
## Test data
#x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
#x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
#X_test = np.concatenate([x_band1[:, :, :, np.newaxis]
#                          , x_band2[:, :, :, np.newaxis]
#                         , ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)
#X_angle_test = np.array(test.inc_angle)
#
#
#X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train
#                    , X_angle_train, y_train, random_state=123, train_size=0.75)
#

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



#X_build, X_valid, y_build, y_valid = train_test_split( X_train, y_train, test_size=0.25, random_state=random_state, stratify=y_train), stratify=y_train
#X_build, X_valid, X_angle_build, X_angle_valid, y_build, y_valid = train_test_split(X_train, X_angle_train, y_train, test_size=0.25, random_state=random_state)


MODEL_WEIGHTS_FILE = inDir + '/Prav_Customnet_01.h5'
ROWS     = 75
COLUMNS  = 75
CHANNELS = 3
nb_epoch = 5
VERBOSEFLAG = 1
batch_size  = 128
patience = 5
optim_type = 'Adam'
learning_rate = 1e-3

from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

#def get_callbacks(filepath, patience=2):
#    es = EarlyStopping('val_loss', patience=patience, mode="min")
#    msave = ModelCheckpoint(filepath, save_best_only=True)
#    return [es, msave]
    
def get_model():
    bn_model = 0
    p_activation = "elu"
    input_1 = Input(shape=(75, 75, 3), name="X_1")
    input_2 = Input(shape=[1], name="angle")
    
    img_1 = Conv2D(16, kernel_size = (3,3), activation=p_activation) ((BatchNormalization(momentum=bn_model))(input_1))
    img_1 = Conv2D(16, kernel_size = (3,3), activation=p_activation) (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(32, kernel_size = (3,3), activation=p_activation) (img_1)
    img_1 = Conv2D(32, kernel_size = (3,3), activation=p_activation) (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(64, kernel_size = (3,3), activation=p_activation) (img_1)
    img_1 = Conv2D(64, kernel_size = (3,3), activation=p_activation) (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(128, kernel_size = (3,3), activation=p_activation) (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = GlobalMaxPooling2D() (img_1)
    
    
    img_2 = Conv2D(128, kernel_size = (3,3), activation=p_activation) ((BatchNormalization(momentum=bn_model))(input_1))
    img_2 = MaxPooling2D((2,2)) (img_2)
    img_2 = Dropout(0.2)(img_2)
    img_2 = GlobalMaxPooling2D() (img_2)
    
    img_concat =  (Concatenate()([img_1, img_2, BatchNormalization(momentum=bn_model)(input_2)]))
    
    dense_ayer = Dropout(0.5) (BatchNormalization(momentum=bn_model) ( Dense(256, activation=p_activation)(img_concat) ))
    dense_ayer = Dropout(0.5) (BatchNormalization(momentum=bn_model) ( Dense(64, activation=p_activation)(dense_ayer) ))
    output = Dense(1, activation="sigmoid")(dense_ayer)
    
    model = Model([input_1,input_2],  output)
    
    return model
#model = get_model()
#model.summary()

#def model_Xception():
#    base_model = Xception(include_top=False, weights='imagenet',input_tensor=None,input_shape=( ROWS, COLUMNS,CHANNELS))
#    
#    for layer in base_model.layers[1:]:
#        layer.trainable = False
#    x = base_model.output
#    x = GlobalAveragePooling2D()(x) 
#    x = Dropout(0.5)(x)
#    x = Dense(1, activation='sigmoid', name='predictions')(x)
#    model = Model(input=base_model.input, output=x)
#    return model
#
#
#model = model_Xception() 


#callbacks = [
#        EarlyStopping(monitor='val_loss', patience=patience, verbose=VERBOSEFLAG),
#        ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=VERBOSEFLAG),
#                ]
##sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#if optim_type == 'SGD':
#    optim = SGD(lr=learning_rate, momentum=0.9)
#else:
#    optim = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
#model.summary()
#
#model.fit([X_build, X_angle_build], y_build, epochs=25
#          , validation_data=([X_valid, X_angle_valid], y_valid)
#         #, nb_epoch = nb_epoch
#         , batch_size=32
#         , callbacks=callbacks
#         ,verbose = VERBOSEFLAG )

def train_nn(i):
    
    trainindex = train_df[train_df['CVindices'] != i].index.tolist()
    valindex   = train_df[train_df['CVindices'] == i].index.tolist()
    
    X_val_df = train_df.iloc[valindex,:]
    
    X_build , X_valid = X_train[trainindex,:], X_train[valindex,:]
    X_angle_build , X_angle_valid = X_angle_train[trainindex], X_angle_train[valindex]
    y_build , y_valid = y_train[trainindex], y_train[valindex]
    
    
    
    model = get_model()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=VERBOSEFLAG),
        ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=VERBOSEFLAG),
                ]
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, momentum=0.9)
    else:
        optim = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
#    early_stop = EarlyStopping(monitor='val_loss', patience=75, verbose=0)
#    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)
    model.fit([X_build, X_angle_build], y_build, epochs=25
          , validation_data=([X_valid, X_angle_valid], y_valid)
         #, nb_epoch = nb_epoch
         , batch_size=32
         , callbacks=callbacks
         ,verbose = VERBOSEFLAG )
    model.load_weights(MODEL_WEIGHTS_FILE)
    print("Train evaluate:")
    print(model.evaluate([X_build, X_angle_build], y_build, verbose=1, batch_size=200))
    print("####################")
    print("watch list evaluate:")
    print(model.evaluate([X_valid, X_angle_valid], y_valid, verbose=1, batch_size=200))
    pred_cv = model.predict([X_valid, X_angle_valid], verbose=1, batch_size=200)

      
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.head()
    pred_cv.columns = ["is_iceberg"]
    pred_cv["id"] = X_val_df.id.values
    
    sub_valfile = inDir+'/submissions/Prav.nn01.fold' + str(i) + '.csv'
    pred_cv = pred_cv[["id","is_iceberg"]]
    pred_cv.to_csv(sub_valfile, index=False)
    pred_test = model.predict([X_test, X_angle_test], verbose=1, batch_size=200)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["is_iceberg"]
    pred_test["id"] = test_df.id.values
    pred_test = pred_test[["id","is_iceberg"]]
    sub_file = inDir+'/submissions/Prav.nn01.fold' + str(i) + '-test' + '.csv'
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
#        
#def batch_generator_train(X_build, y_build ,batch_size):
#    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
#    number_of_batches = np.ceil(len(X_build)/batch_size)
#    counter = 0
#    
#    while True:
#        X_batch = X_build[batch_size*counter:batch_size*(counter+1)]        
#        y_batch = y_build[batch_size*counter:batch_size*(counter+1)]
#        counter += 1
#        yield X_batch, y_batch
#        if (counter == number_of_batches):
#            counter = 0 
#            
#def batch_generator_valid(X_valid, y_valid ,batch_size):
#    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
#    number_of_batches = np.ceil(len(X_valid)/batch_size)
#    counter = 0
#    
#    while True:
#        X_batch = X_valid[batch_size*counter:batch_size*(counter+1)]
#        y_batch = y_valid[batch_size*counter:batch_size*(counter+1)]
##        X_batch = X_valid.ix[list(batch_index)]
##        y_batch = y_valid.ix[list(batch_index)]
#        #X_batch = normalize_image(X_batch)
#        counter += 1
#        yield X_batch, y_batch
#        if (counter == number_of_batches):
#            counter = 0 
#
#model.fit_generator( generator=batch_generator_train(X_build, y_build ,batch_size),                            
#                     steps_per_epoch = np.ceil(len(X_build) / batch_size), 
#                     nb_epoch = nb_epoch, 
#                     callbacks = callbacks,
#                     validation_data=batch_generator_valid(X_valid, y_valid ,batch_size),
#                     validation_steps = np.ceil(len(X_valid) / batch_size), 
#                     max_q_size=10,
#                     workers = 4,
#                     verbose = VERBOSEFLAG 
#                  )
#
#################################################
#print("fine tune the model")
#################################################
#
#for layer in model.layers[1:]:
#   layer.trainable = True
#  
#nb_epoch = 15
#initial_epoch = 5
#learning_rate = 1e-4
#optim = Adam(lr=learning_rate)
#model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
#model.summary()
#
#model.fit_generator( generator=batch_generator_train(X_build, y_build ,batch_size),                            
#                     steps_per_epoch = np.ceil(len(X_build) / batch_size),
#                     initial_epoch = initial_epoch,
#                     nb_epoch = nb_epoch, 
#                     callbacks = callbacks,
#                     validation_data=batch_generator_valid(X_valid, y_valid ,batch_size),
#                     validation_steps = np.ceil(len(X_valid) / batch_size), 
#                     max_q_size=10,
#                     workers = 4,
#                     verbose = VERBOSEFLAG 
#                  )
#
#
#model.load_weights(MODEL_WEIGHTS_FILE)
#
#prediction = model.predict(X_test, verbose=1)
#
#submit_df = pd.DataFrame({'id': test_df["id"], 'is_iceberg': prediction.flatten()})
#submit_df.to_csv(inDir +"/submissions/Prav_Xception_01.csv", index=False)
#
#


            
#
#from keras.models import Sequential
#from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense, Dropout
#
#model = Sequential()
#model.add(Convolution2D(32, 3, activation="relu", input_shape=(75, 75, 2)))
#model.add(Convolution2D(64, 3, activation="relu", input_shape=(75, 75, 2)))
#model.add(GlobalAveragePooling2D())
#model.add(Dropout(0.3))
#model.add(Dense(1, activation="sigmoid"))
#model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
#model.summary()
#
#model.fit(X_train, y_train, validation_split=0.2)
#
## Make predictions
#prediction = model.predict(X_test, verbose=1)
#
#submit_df = pd.DataFrame({'id': test_df["id"], 'is_iceberg': prediction.flatten()})
#submit_df.to_csv(inDir +"/submissions/naive_submission.csv", index=False)
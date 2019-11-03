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

trainfoldSource = pd.read_csv(inDir+'/input/Prav_4folds_CVindices.csv')
train_df = pd.merge(train_df, trainfoldSource, how='left',on="id")

train_df.groupby(['CVindices'])[['is_iceberg']].sum()


train_df.inc_angle = train_df.inc_angle.replace('na', 0)
train_df.inc_angle = train_df.inc_angle.astype(float).fillna(0.0)

test_df.inc_angle = test_df.inc_angle.replace('na', 0)
test_df.inc_angle = test_df.inc_angle.astype(float).fillna(0.0)

print("done!")

## Train data
#x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])
#x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])
#X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis],((x_band1+x_band2)/2)[:, :, :, np.newaxis]], axis=-1)
#X_angle_train = np.array(train_df.inc_angle)
#y_train = np.array(train_df["is_iceberg"])
#print("Xtrain:", X_train.shape)
#
## Test data
#x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]])
#x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]])
#X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis],((x_band1+x_band2)/2)[:, :, :, np.newaxis]], axis=-1)
#X_angle_test = np.array(test_df.inc_angle)
#print("Xtest:", X_test.shape)

#Generate the training data
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])
#X_band_3=(X_band_1+X_band_2)/2
X_band_3=np.fabs(np.subtract(X_band_1,X_band_2))
X_band_4=np.maximum(X_band_1,X_band_2)
X_band_5=np.minimum(X_band_1,X_band_2)
#X_band_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in train["inc_angle"]])
X_train = np.concatenate([
                          
                          X_band_3[:, :, :, np.newaxis],X_band_4[:, :, :, np.newaxis],X_band_5[:, :, :, np.newaxis]], axis=-1)

X_angle_train = np.array(train_df.inc_angle)
y_train = np.array(train_df["is_iceberg"])
print("Xtrain:", X_train.shape)

X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]])
#X_band_test_3=(X_band_test_1+X_band_test_2)/2
X_band_test_3=np.fabs(np.subtract(X_band_test_1,X_band_test_2))
X_band_test_4=np.maximum(X_band_test_1,X_band_test_2)
X_band_test_5=np.minimum(X_band_test_1,X_band_test_2)
#X_band_test_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in test["inc_angle"]])
X_test = np.concatenate([
                          X_band_test_3[:, :, :, np.newaxis], X_band_test_4[:, :, :, np.newaxis],X_band_test_5[:, :, :, np.newaxis]],axis=-1)

X_angle_test = np.array(test_df.inc_angle)
print("Xtest:", X_test.shape)
#X_build, X_valid, y_build, y_valid = train_test_split( X_train, y_train, test_size=0.25, random_state=random_state, stratify=y_train), stratify=y_train
#X_build, X_valid, X_angle_build, X_angle_valid, y_build, y_valid = train_test_split(X_train, X_angle_train, y_train, test_size=0.25, random_state=random_state)


MODEL_WEIGHTS_FILE = inDir + '/Prav_vgg_04.h5'
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
gen = ImageDataGenerator(#horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.,
                         height_shift_range = 0.,
                         channel_shift_range=0,
                         zoom_range = 0.5,
                         rotation_range = 10)

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
            yield [X1i[0], X2i[1]], X1i[1]

# Finally create generator
def get_callbacks(filepath, patience=2):
   #es = EarlyStopping('val_loss', patience=10, mode="min")
   es = EarlyStopping('val_loss', patience=20, mode="min")
   msave = ModelCheckpoint(filepath, save_best_only=True)
   return [es, msave]


def getVggAngleModel():
    input_2 = Input(shape=[1], name="angle")
    angle_layer = Dense(1, )(input_2)
    base_model = VGG16(weights='imagenet', include_top=False, 
                 input_shape=X_train.shape[1:], classes=1)
    x = base_model.get_layer('block5_pool').output
        #x = GlobalMaxPooling2D()(x)
    x = Flatten()(x)
    base_model2 = MobileNet(weights=None, alpha=0.9,input_tensor = base_model.input,include_top=False, input_shape=X_train.shape[1:])

    x2 = base_model2.output
    x2 = GlobalAveragePooling2D()(x2)

    merge_one = concatenate([x, x2, angle_layer])

    merge_one = Dropout(0.6)(merge_one)
    predictions = Dense(1, activation='sigmoid',kernel_initializer='he_normal')(merge_one)
    
    model = Model(input=[base_model.input, input_2], output=predictions)
    
    return model


model.summary()

def train_nn(i):
    
    trainindex = train_df[train_df['CVindices'] != i].index.tolist()
    valindex   = train_df[train_df['CVindices'] == i].index.tolist()
    
    X_val_df = train_df.iloc[valindex,:]
    
    X_build , X_valid = X_train[trainindex,:], X_train[valindex,:]
    X_angle_build , X_angle_valid = X_angle_train[trainindex], X_angle_train[valindex]
    y_build , y_valid = y_train[trainindex], y_train[valindex]   
    
    #model = get_model()
    model= getVggAngleModel()
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
                        validation_data=([X_valid,X_angle_valid], y_valid),
                        callbacks=callbacks,
                        verbose = VERBOSEFLAG
                        )

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
    
    sub_valfile = inDir+'/submissions/Prav.vgg04.fold' + str(i) + '.csv'
    pred_cv = pred_cv[["id","is_iceberg"]]
    pred_cv.to_csv(sub_valfile, index=False)
    pred_test = model.predict([X_test, X_angle_test], verbose=1, batch_size=200)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["is_iceberg"]
    pred_test["id"] = test_df.id.values
    pred_test = pred_test[["id","is_iceberg"]]
    sub_file = inDir+'/submissions/Prav.vgg04.fold' + str(i) + '-test' + '.csv'
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

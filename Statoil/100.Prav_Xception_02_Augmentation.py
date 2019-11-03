# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 19:13:34 2017

@author: SriPrav
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import train_test_split
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D, Input
from sklearn.metrics import log_loss
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(        
                            zoom_range=0.2,
                            rotation_range=10.,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True
                             )

valid_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator(        
                            zoom_range=0.2,
                            rotation_range=10.,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True
                             )

inDir = 'C:/Users/SriPrav/Documents/R/35Statoil'

train_df = pd.read_json(inDir + "/input/train.json")
test_df = pd.read_json(inDir + "/input/test.json")


# Train data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis],x_band1[:, :, :, np.newaxis]], axis=-1)
y_train = np.array(train_df["is_iceberg"])
print("Xtrain:", X_train.shape)

# Test data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]])
X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis],x_band1[:, :, :, np.newaxis]], axis=-1)
print("Xtest:", X_test.shape)

random_state = 2017
np.random.seed(random_state)

X_build, X_valid, y_build, y_valid = train_test_split( X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train)

MODEL_WEIGHTS_FILE = inDir + '/Prav_xception_02.h5'
ROWS     = 75
COLUMNS  = 75
CHANNELS = 3
nb_epoch = 10
VERBOSEFLAG = 1
batch_size  = 64
patience = 2
optim_type = 'Adam'
learning_rate = 1e-3

#del model
def model_Xception():
    base_model = Xception(include_top=False, weights='imagenet',input_tensor=None,input_shape=( ROWS, COLUMNS,CHANNELS))
    
    for layer in base_model.layers[1:]:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x) 
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)
    model = Model(input=base_model.input, output=x)
    return model


model = model_Xception() 


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
model.summary()

def batch_generator_train(X_build, y_build ,batch_size):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(len(X_build)/batch_size)
    counter = 0
    
    while True:
        X_batch = X_build[batch_size*counter:batch_size*(counter+1)]        
        y_batch = y_build[batch_size*counter:batch_size*(counter+1)]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            counter = 0 
            
def batch_generator_valid(X_valid, y_valid ,batch_size):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(len(X_valid)/batch_size)
    counter = 0
    
    while True:
        X_batch = X_valid[batch_size*counter:batch_size*(counter+1)]
        y_batch = y_valid[batch_size*counter:batch_size*(counter+1)]
#        X_batch = X_valid.ix[list(batch_index)]
#        y_batch = y_valid.ix[list(batch_index)]
        #X_batch = normalize_image(X_batch)
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            counter = 0 

model.fit_generator(train_datagen.flow(X_build, y_build, batch_size=batch_size),
                    steps_per_epoch = np.ceil(len(X_build) / batch_size), 
                    nb_epoch=nb_epoch,
                    validation_data=valid_datagen.flow(X_valid, y_valid, batch_size=batch_size),
                    validation_steps = np.ceil(len(X_valid) / batch_size), 
                    verbose=VERBOSEFLAG,
                    callbacks=callbacks)

model.fit_generator( generator=batch_generator_train(X_build, y_build ,batch_size),                            
                     steps_per_epoch = np.ceil(len(X_build) / batch_size), 
                     nb_epoch = nb_epoch, 
                     callbacks = callbacks,
                     validation_data=batch_generator_valid(X_valid, y_valid ,batch_size),
                     validation_steps = np.ceil(len(X_valid) / batch_size), 
                     max_q_size=10,
                     workers = 4,
                     verbose = VERBOSEFLAG 
                  )

################################################
print("fine tune the model")
################################################

for layer in model.layers[1:]:
   layer.trainable = True
  

initial_epoch = nb_epoch
nb_epoch = 20
learning_rate = 1e-3
optim = Adam(lr=learning_rate)
model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit_generator(train_datagen.flow(X_build, y_build, batch_size=batch_size),
                    steps_per_epoch = np.ceil(len(X_build) / batch_size),
                    initial_epoch = initial_epoch,
                    nb_epoch=nb_epoch,
                    validation_data=valid_datagen.flow(X_valid, y_valid, batch_size=batch_size),
                    validation_steps = np.ceil(len(X_valid) / batch_size), 
                    verbose=VERBOSEFLAG,
                    callbacks=callbacks)

model.fit_generator( generator=batch_generator_train(X_build, y_build ,batch_size),                            
                     steps_per_epoch = np.ceil(len(X_build) / batch_size),
                     initial_epoch = initial_epoch,
                     nb_epoch = nb_epoch, 
                     callbacks = callbacks,
                     validation_data=batch_generator_valid(X_valid, y_valid ,batch_size),
                     validation_steps = np.ceil(len(X_valid) / batch_size), 
                     max_q_size=10,
                     workers = 4,
                     verbose = VERBOSEFLAG 
                  )


model.load_weights(MODEL_WEIGHTS_FILE)

predictions_valid = model.predict(X_valid, verbose=VERBOSEFLAG)
score = log_loss(y_valid, predictions_valid)
print('Score log_loss: ', score)
        
prediction = model.predict(X_test, verbose=1)

submit_df = pd.DataFrame({'id': test_df["id"], 'is_iceberg': prediction.flatten()})
submit_df.to_csv(inDir +"/submissions/Prav_Xception_02.csv", index=False)


from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, GlobalMaxPooling2D, Dense
simple_cnn = Sequential()
simple_cnn.add(BatchNormalization(input_shape = (75, 75, 3)))
for i in range(4):
    simple_cnn.add(Conv2D(8*2**i, kernel_size = (3,3)))
    simple_cnn.add(MaxPooling2D((2,2)))
simple_cnn.add(GlobalMaxPooling2D())
simple_cnn.add(Dropout(0.5))
simple_cnn.add(Dense(8))
simple_cnn.add(Dense(1, activation = 'sigmoid'))
simple_cnn.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
simple_cnn.summary()

simple_cnn.fit(X_build, y_build, validation_data = (X_valid, y_valid), epochs = 10, shuffle = True)
del simple_cnn
simple_cnn.fit_generator( generator=batch_generator_train(X_build, y_build ,batch_size),                            
                     steps_per_epoch = np.ceil(len(X_build) / batch_size), 
                     nb_epoch = nb_epoch, 
                     callbacks = callbacks,
                     validation_data=batch_generator_valid(X_valid, y_valid ,batch_size),
                     validation_steps = np.ceil(len(X_valid) / batch_size), 
                     max_q_size=10,
                     workers = 4,
                     verbose = VERBOSEFLAG 
                  )

simple_cnn.fit_generator(train_datagen.flow(X_build, y_build, batch_size=batch_size),
                    steps_per_epoch = np.ceil(len(X_build) / batch_size), 
                    nb_epoch=nb_epoch,
                    validation_data=valid_datagen.flow(X_valid, y_valid, batch_size=batch_size),
                    validation_steps = np.ceil(len(X_valid) / batch_size), 
                    verbose=VERBOSEFLAG,
                    callbacks=callbacks)

            
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
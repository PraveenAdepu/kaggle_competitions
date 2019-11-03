# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:51:19 2017

@author: PA23309
"""

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense , Activation, Dropout
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array

import pandas as pd
import numpy as np
import datetime

from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint

import math
import os

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	# transform data to be stationary
	diff_series = difference(raw_values, 1)
	diff_values = diff_series.values
	diff_values = diff_values.reshape(len(diff_values), 1)
	# rescale values to -1, 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(diff_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return scaler, train, test


inDir =r'C:\Users\SriPrav\Documents\R\40Recruit'

source = pd.read_csv(inDir+"/input/air_visit_data.csv")

source_pivot = pd.pivot_table(source, values = 'visitors', index=['air_store_id'], columns = 'visit_date').reset_index()
source_pivot.fillna(0, inplace = True)

horizon_periods = 39
features = 90

n_batch = 1
Total_features = features + horizon_periods

Prav_5fold_CVindices = pd.read_csv(inDir+"/input/Prav_5folds_CVindices.csv")

source_pivot = pd.merge(source_pivot, Prav_5fold_CVindices, how='left',on="air_store_id")

source_train_id = pd.DataFrame(source_pivot[["air_store_id","CVindices"]])
del source_pivot["air_store_id"]
del source_pivot["CVindices"]

X1 = np.array(source_pivot)

# y = (x - min) / (max - min)
#mini = X1.min()
#maxi =X1.max()

#maximini = maxi - mini

#X2 = X1-mini
#X2 = X2/maximini

#X2 = np.log1p(X1)
#X3 = np.expm1(X2)
#X2.min()
#X2.max()
#
#sd = np.std(X1)
#
#scaler = MinMaxScaler(feature_range=(-1, 1))
#scaler = scaler.fit(X1)
##print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
## normalize the dataset and print
#X2 = scaler.transform(X1)
#
## inverse transform and print
#X3 = scaler.inverse_transform(X2)

    
#X2 = X1-mean
#X2 = X2/sd

# Remove last 3 days for cross validation
X2 = X1

mean = np.mean(X2)
#sd = np.std(X1)

X3 = X2-mean
#X2 = X2/sd

testingSet = X3[:,-features:]
X4 = X3[:,:-3]
validationSet = X4[:,-Total_features:-horizon_periods]
trainingSet = X4

##########################################################################################################################################
##########################################################################################################################################

features_size = Total_features

nb_epoch = 5
VERBOSEFLAG = 1
batch_size  = 512
patience = 2
optim_type = 'Adam'
learning_rate = 1e-3
ModelName = 'LSTM04'
MODEL_WEIGHTS_FILE = inDir + '/Prav_LSTM_04.h5'


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
            
#def LSTM_Model(train, y, n_batch, nb_epoch, n_neurons):
#    #train = train.reshape(train.shape[0], 1, train.shape[1])
#	# design network
#	model = Sequential()
#	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train.shape[1], train.shape[2]), stateful=True))
#	model.add(Dense(y.shape[1]))
#	#model.compile(loss='mean_squared_error', optimizer='adam')
#	return model

def LSTM_Model(train, y, n_batch, nb_epoch, n_neurons):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, stateful=True, batch_input_shape=(n_batch, train.shape[1], train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True, stateful=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, stateful=True))
    model.add(Dense(y.shape[1]))
    return model


folds = 5
for i in range(1, folds+1):
    print(i)
    trainindex = source_train_id[source_train_id['CVindices'] != i].index.tolist()
    valindex   = source_train_id[source_train_id['CVindices'] == i].index.tolist()
    
    X_val_df = source_train_id.iloc[valindex,:]
    X_val_df = X_val_df.reset_index()
    X_val_predSet = validationSet[valindex,:]
    
    X_build_features, X_valid_features = trainingSet[trainindex], trainingSet[valindex]
    
    
    X_train = []
    
    nb_sample = 0
    for nb_sample in range(0,X_build_features.shape[0]):
        #print(nb_sample)
        current_sample = []
        current_sample = X_build_features[nb_sample]
        current_sample_format = []
        current_sample_format = np.atleast_2d(np.array([current_sample[start:start + features_size] for start in range(0, current_sample.shape[0] - features_size)]))
        if nb_sample ==0:
            X_train = current_sample_format
        else:
            X_train = np.append(X_train, current_sample_format, axis=0)
    
    X_build = X_train[:,:-horizon_periods]
    y_build = X_train[:,-horizon_periods:]
    
    X_train = []
    
    nb_sample = 0
    for nb_sample in range(0,X_valid_features.shape[0]):
        #print(nb_sample)
        current_sample = []
        current_sample = X_valid_features[nb_sample]
        current_sample_format = []
        current_sample_format = np.atleast_2d(np.array([current_sample[start:start + features_size] for start in range(0, current_sample.shape[0] - features_size)]))
        if nb_sample ==0:
            X_train = current_sample_format
        else:
            X_train = np.append(X_train, current_sample_format, axis=0)
    
    X_valid = X_train[:,:-horizon_periods]
    y_valid = X_train[:,-horizon_periods:]
    
    
    X_build =  X_build.reshape(X_build.shape[0], X_build.shape[1],1)
    X_valid =  X_valid.reshape(X_valid.shape[0], X_valid.shape[1],1)
    
    model = LSTM_Model(X_build, y_build,n_batch = batch_size, nb_epoch = 2, n_neurons = 100 )
    callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, verbose=VERBOSEFLAG),
            ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=VERBOSEFLAG),
                    ]
    
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, momentum=0.9)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim,loss='mean_squared_error',metrics=['accuracy'])
#    model.summary()    
    
    X_build = X_build[0:math.floor(np.ceil((np.ceil(len(X_build) / batch_size)-1)*batch_size)),]
    X_valid = X_valid[0:math.floor(np.ceil((np.ceil(len(X_valid) / batch_size)-1)*batch_size)),]
    
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
    
    model.load_weights(MODEL_WEIGHTS_FILE)
       
    # expand to what shape 
    validationSet_target = np.zeros((math.floor(np.ceil((np.ceil(len(X_val_predSet) / batch_size))*batch_size)), X_val_predSet.shape[1]))
    
    # do expand
    validationSet_target[:X_val_predSet.shape[0], :X_val_predSet.shape[1]] = X_val_predSet    
    
    validationSet_target = validationSet_target.reshape(validationSet_target.shape[0], validationSet_target.shape[1],1)    
    
    pred_valid = model.predict(validationSet_target,batch_size=batch_size,verbose = 1)
    
    pred_valid = pred_valid[:X_val_predSet.shape[0],]
    
#    pred_valid = scaler.inverse_transform(pred_valid)
    
#    pred_valid = pred_valid * maximini
#    pred_valid = pred_valid + mini
#    pred_valid = np.expm1(pred_valid)
    
    pred_valid = pred_valid + mean
    pred_valid = pd.DataFrame(pred_valid)
    
    pred_valid_names = pd.date_range("2017-03-12", "2017-04-19", freq="1D")
    
    pred_valid.columns = pred_valid_names
    source_train_id.shape[0]
    X_validationSet = pd.concat([X_val_df["air_store_id"], pred_valid], axis = 1)    
    
    X_validationSet = pd.melt(X_validationSet, id_vars='air_store_id', value_vars=pred_valid_names)
    
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
    X_validationSet.to_csv(sub_file, index=False)
    
    # expand to what shape 
    testingSet_target = np.zeros((math.floor(np.ceil((np.ceil(len(testingSet) / batch_size))*batch_size)), testingSet.shape[1]))
    
    # do expand
    testingSet_target[:testingSet.shape[0], :testingSet.shape[1]] = testingSet    
    
    testingSet_target = testingSet_target.reshape(testingSet_target.shape[0], testingSet_target.shape[1],1)    
    
    pred_test = model.predict(testingSet_target,batch_size=batch_size,verbose = 1)
    
    pred_test = pred_test[:testingSet.shape[0],]
    
    #pred_test = scaler.inverse_transform(pred_test)
    #pred_test = pred_test * maximini
    #pred_test = pred_test + mini
    #pred_test = np.expm1(pred_test)
    pred_test = pred_test + mean
    pred_test = pd.DataFrame(pred_test)
    
    pred_test_names = pd.date_range("2017-04-23", "2017-05-31", freq="1D")
    
    pred_test.columns = pred_test_names
    source_train_id.shape[0]
    X_testingSet = pd.concat([source_train_id["air_store_id"], pred_test], axis = 1)    
    
    X_testingSet = pd.melt(X_testingSet, id_vars='air_store_id', value_vars=pred_test_names)
    
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold'+ str(i) +'-test' + '.csv'
    X_testingSet.to_csv(sub_file, index=False)

    os.remove(MODEL_WEIGHTS_FILE)

    del model


#X2 = X1
#mean = np.mean(X2)
##sd = np.std(X1)
#
#X3 = X2-mean
##X2 = X2/sd
#
#testingSet = X3[:,-features:]
##validationSet = X3[:,-Total_features:-horizon_periods]
#trainingSet = X3[:,:-3]

X_train = []
    
nb_sample = 0
for nb_sample in range(0,trainingSet.shape[0]):
    #print(nb_sample)
    current_sample = []
    current_sample = trainingSet[nb_sample]
    current_sample_format = []
    current_sample_format = np.atleast_2d(np.array([current_sample[start:start + features_size] for start in range(0, current_sample.shape[0] - features_size)]))
    if nb_sample ==0:
        X_train = current_sample_format
    else:
        X_train = np.append(X_train, current_sample_format, axis=0)
    
X_full_train = X_train[:,:-horizon_periods]
y_full_train = X_train[:,-horizon_periods:]

X_full_train =  X_full_train.reshape(X_full_train.shape[0], X_full_train.shape[1],1)

model = LSTM_Model(X_full_train, y_full_train,n_batch = batch_size, nb_epoch = 2, n_neurons = 100 )
callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=VERBOSEFLAG),
        ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=VERBOSEFLAG),
                ]

#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
if optim_type == 'SGD':
    optim = SGD(lr=learning_rate, momentum=0.9)
else:
    optim = Adam(lr=learning_rate)
model.compile(optimizer=optim,loss='mean_squared_error')
model.summary()    

X_full_train = X_full_train[0:math.floor(np.ceil((np.ceil(len(X_full_train) / batch_size)-1)*batch_size)),]

nb_epoch = 6
model.fit_generator( generator=batch_generator_train(X_full_train, y_full_train ,batch_size),                            
                     steps_per_epoch = np.ceil(len(X_full_train) / batch_size), 
                     nb_epoch = nb_epoch, 
                     #callbacks = callbacks,
#                     validation_data=batch_generator_valid(X_valid, y_valid ,batch_size),
#                     validation_steps = np.ceil(len(X_valid) / batch_size), 
                     max_q_size=10,
                     workers = 4,
                     verbose = VERBOSEFLAG 
                  )

#model.load_weights(MODEL_WEIGHTS_FILE)  

# expand to what shape 
testingSet_target = np.zeros((math.floor(np.ceil((np.ceil(len(testingSet) / batch_size))*batch_size)), testingSet.shape[1]))

# do expand
testingSet_target[:testingSet.shape[0], :testingSet.shape[1]] = testingSet    

testingSet_target = testingSet_target.reshape(testingSet_target.shape[0], testingSet_target.shape[1],1)    

pred_test = model.predict(testingSet_target,batch_size=batch_size,verbose = 1)

pred_test = pred_test[:testingSet.shape[0],]

#pred_test = scaler.inverse_transform(pred_test)
#pred_test = pred_test * maximini
#pred_test = pred_test + mini
#pred_test = np.expm1(pred_test)
pred_test = pred_test + mean
pred_test = pd.DataFrame(pred_test)

pred_test_names = pd.date_range("2017-04-23", "2017-05-31", freq="1D")

pred_test.columns = pred_test_names
source_train_id.shape[0]
X_testingSet = pd.concat([source_train_id["air_store_id"], pred_test], axis = 1)    

X_testingSet = pd.melt(X_testingSet, id_vars='air_store_id', value_vars=pred_test_names)

sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'full-test' + '.csv'
X_testingSet.to_csv(sub_file, index=False)

#os.remove(MODEL_WEIGHTS_FILE)

del model








#def build_model(layers):
#    model = Sequential()
#
#    model.add(LSTM(
#        input_shape=(layers[1], layers[0]),
#        output_dim=layers[1],
#        return_sequences=True))
#    model.add(Dropout(0.2))
#
#    model.add(LSTM(
#        layers[2],
#        return_sequences=False))
#    model.add(Dropout(0.2))
#
#    model.add(Dense(
#        output_dim=layers[3]))
#    model.add(Activation("linear"))
#
#    model.compile(loss="mse", optimizer="rmsprop")
#    return model
#model = build_model([1, 50, 100, 1])    
#
#
#
#
#testingSet =  testingSet.reshape(testingSet.shape[0], testingSet.shape[1],1)
#pred_test = model.predict(testingSet,verbose = 1)
#
#pred_test = pred_test * sd
#pred_test = pred_test + mean
#
#from sklearn.metrics import mean_squared_error
#from math import sqrt
#
#pred_cv[pred_cv<0] = 0
#pred_cv[pred_cv<0] = 0
#rms = sqrt(mean_squared_error(np.log1p(pred_cv), np.log1p(y_valid)))
#rms
#prediction = model.predict(X_test, verbose=1)
#
#
#
#
#
#
#
#
######################################################################################################################
#source_validation_set = np.array(source_validation_features)
#
#source_validation_set = source_validation_set.reshape(source_validation_set.shape[0], source_validation_set.shape[1], 1)
#
#pred_cv = model.predict(source_validation_set, batch_size = n_batch, verbose = 1)
#
#pred_cv = pd.DataFrame(pred_cv)
#
#pred_cv.columns = source_validation.columns
#
#
#X_validationSet = pd.concat([source_train_id, pred_cv], axis = 1)
#
#
#X_validationSet = pd.melt(X_validationSet, id_vars='device_mmc', value_vars=source_validation.columns)
#
#X_validationSet.to_csv(inDir+"/X_LSTM_validationSet.csv", index=False)
#
#
#
#del model
## configure
#n_lag = 180
#n_seq = 60
#n_test = 20
#n_epochs = 100
#n_batch = 1
#n_neurons = 1
## prepare data
#X10 = pd.DataFrame(X1[0])
#
#scaler, train, test = prepare_data(X10, n_test, n_lag, n_seq)
#
#
#
##X10 = X1[0]
##
##window_size = 90
##
##X101 = np.atleast_2d(np.array([X10[start:start + window_size] for start in range(0, X10.shape[0] - window_size)]))
##
##
##
##X1 = np.array(X)
##
##X20 = X1[1]
##
##window_size = 90
##
##X201 = np.atleast_2d(np.array([X20[start:start + window_size] for start in range(0, X20.shape[0] - window_size)]))
##
##X_train = []
##
##X_train.append(X101)
##X_train.append(X201)
##
##X_train = np.atleast_3d(X_train)
##
##X_train[1]
#
#X_train = []
#
#features_size = 90
#for nb_sample in range(0,X.shape[0]):
#    print(nb_sample)
#    current_sample = []
#    current_sample = X1[nb_sample]
#    current_sample_format = []
#    current_sample_format = np.atleast_2d(np.array([current_sample[start:start + features_size] for start in range(0, current_sample.shape[0] - features_size)]))
#    X_train.append(current_sample_format)
#X_train = np.array(X_train)
#    
#Y_train = np.array(Y)
#test_size = int(0.2 * X.shape[0])           # In real life you'd want to use 0.2 - 0.5
#X_build, X_valid, y_build, y_valid = X_train[:-test_size], X_train[-test_size:], Y_train[:-test_size], Y_train[-test_size:]
#
#X_build = np.atleast_3d(X_build)
## fit an LSTM network to training data
#def LSTM_Model(train, y, n_batch, nb_epoch, n_neurons):
#
#	# design network
#	model = Sequential()
#	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train.shape[1], train.shape[2]), stateful=True))
#	model.add(Dense(y.shape[1]))
#	model.compile(loss='mean_squared_error', optimizer='adam')
#	return model
#
#model = LSTM_Model(X_build, y_build,n_batch = 1, nb_epoch = 25, n_neurons = 10 )
#	# fit network
#	for i in range(nb_epoch):
#		model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
#		model.reset_states()
#
#from __future__ import print_function, division
#
#import numpy as np
#from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
#from keras.models import Sequential
#    
#def make_timeseries_regressor(window_size, filter_length, nb_input_series=1, nb_outputs=1, nb_filter=4):
#    """:Return: a Keras Model for predicting the next value in a timeseries given a fixed-size lookback window of previous values.
#    The model can handle multiple input timeseries (`nb_input_series`) and multiple prediction targets (`nb_outputs`).
#    :param int window_size: The number of previous timeseries values to use as input features.  Also called lag or lookback.
#    :param int nb_input_series: The number of input timeseries; 1 for a single timeseries.
#      The `X` input to ``fit()`` should be an array of shape ``(n_instances, window_size, nb_input_series)``; each instance is
#      a 2D array of shape ``(window_size, nb_input_series)``.  For example, for `window_size` = 3 and `nb_input_series` = 1 (a
#      single timeseries), one instance could be ``[[0], [1], [2]]``. See ``make_timeseries_instances()``.
#    :param int nb_outputs: The output dimension, often equal to the number of inputs.
#      For each input instance (array with shape ``(window_size, nb_input_series)``), the output is a vector of size `nb_outputs`,
#      usually the value(s) predicted to come after the last value in that input instance, i.e., the next value
#      in the sequence. The `y` input to ``fit()`` should be an array of shape ``(n_instances, nb_outputs)``.
#    :param int filter_length: the size (along the `window_size` dimension) of the sliding window that gets convolved with
#      each position along each instance. The difference between 1D and 2D convolution is that a 1D filter's "height" is fixed
#      to the number of input timeseries (its "width" being `filter_length`), and it can only slide along the window
#      dimension.  This is useful as generally the input timeseries have no spatial/ordinal relationship, so it's not
#      meaningful to look for patterns that are invariant with respect to subsets of the timeseries.
#    :param int nb_filter: The number of different filters to learn (roughly, input patterns to recognize).
#    """
#    model = Sequential((
#        # The first conv layer learns `nb_filter` filters (aka kernels), each of size ``(filter_length, nb_input_series)``.
#        # Its output will have shape (None, window_size - filter_length + 1, nb_filter), i.e., for each position in
#        # the input timeseries, the activation of each filter at that position.
#        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu', input_shape=(window_size, nb_input_series)),
#        MaxPooling1D(),     # Downsample the output of convolution by 2X.
#        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu'),
#        MaxPooling1D(),
#        Flatten(),
#        Dense(nb_outputs, activation='linear'),     # For binary classification, change the activation to 'sigmoid'
#    ))
#    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#    # To perform (binary) classification instead:
#    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
#    return model 
#   
#window_size = X_build.shape[1]
#input_nb_series = X_build.shape[2]
#filter_length = 5
#nb_filter = 4
#output_nb_series = y_build.shape[1]
#    
#model = make_timeseries_regressor(window_size=window_size, filter_length=filter_length, nb_input_series=input_nb_series, nb_outputs=output_nb_series, nb_filter=nb_filter)
#
#print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape, model.output_shape, nb_filter, filter_length))
#model.summary()
#
#
#model.fit(X_build, y_build, nb_epoch=25, batch_size=2, validation_data=(X_valid, y_valid))
#
#pred = model.predict(X_build)
#pred_cv = model.predict(X_valid)




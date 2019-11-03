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
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array

import pandas as pd
import numpy as np
import datetime


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

series = X10

inDir =r'C:\Users\PA23309\Documents\Prav-Development\Alex\Forecasting\HTC11'

source = pd.read_csv(inDir+"/HistoricalPresentDevicesSales_HTC.csv")

source["MMC Band"].fillna("NA", inplace=True)
source.head()

source["device_mmc"] = source["Model Name"]+"_"+source["MMC Band"]

del source["Model Name"]
del source["MMC Band"]
del source["Avg Customer Contribution"]

source["Device Count"].fillna(0, inplace=True)

source["date"] = pd.to_datetime(source["Day"])
source["date"] = source["date"].dt.date

del source["Day"]

source = source[source["date"] < datetime.date(2017,11,4)]

source_pivot = pd.pivot_table(source, values = 'Device Count', index=['device_mmc'], columns = 'date').reset_index()

horizon_periods = 60
features = 120

Total_features = features + horizon_periods

source_validation = source_pivot[source_pivot.columns[-horizon_periods:]].copy()

source_validation_features = source_pivot[source_pivot.columns[-Total_features:-horizon_periods]].copy()

source_train = source_pivot[source_pivot.columns[:-horizon_periods]].copy()

source_train_id = pd.DataFrame(source_train["device_mmc"])
del source_train["device_mmc"]
X1 = np.array(source_train)

X_train = []
features_size = Total_features

nb_sample = 0
for nb_sample in range(0,X1.shape[0]):
    print(nb_sample)
    current_sample = []
    current_sample = X1[nb_sample]
    current_sample_format = []
    current_sample_format = np.atleast_2d(np.array([current_sample[start:start + features_size] for start in range(0, current_sample.shape[0] - features_size)]))
    if nb_sample ==0:
        X_train = current_sample_format
    else:
        X_train = np.append(X_train, current_sample_format, axis=0)


y_train = X_train[:,-horizon_periods:]

X_train = X_train[:,:-horizon_periods]

test_size = int(0.2 * X_train.shape[0])

X_build, X_valid, y_build, y_valid = X_train[:-test_size], X_train[-test_size:], y_train[:-test_size], y_train[-test_size:]

#X_build =  X_build.reshape(X_build.shape[0], 1, X_build.shape[1])
#X_valid =  X_valid.reshape(X_valid.shape[0], 1, X_valid.shape[1])

X_build =  X_build.reshape(X_build.shape[0], X_build.shape[1],1)
X_valid =  X_valid.reshape(X_valid.shape[0], X_valid.shape[1],1)

def LSTM_Model(train, y, n_batch, nb_epoch, n_neurons):
    #train = train.reshape(train.shape[0], 1, train.shape[1])
	# design network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train.shape[1], train.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

model = LSTM_Model(X_build, y_build,n_batch = 1, nb_epoch = 250, n_neurons = 10 )

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(X_build, y_build, epochs=5, batch_size=n_batch, validation_data=(X_valid, y_valid),verbose=1, shuffle=False)

pred = model.predict(X_valid, batch_size = n_batch, verbose = 1)

source_validation_set = np.array(source_validation_features)

source_validation_set = source_validation_set.reshape(source_validation_set.shape[0], source_validation_set.shape[1], 1)

pred_cv = model.predict(source_validation_set, batch_size = n_batch, verbose = 1)

pred_cv = pd.DataFrame(pred_cv)

pred_cv.columns = source_validation.columns


X_validationSet = pd.concat([source_train_id, pred_cv], axis = 1)


X_validationSet = pd.melt(X_validationSet, id_vars='device_mmc', value_vars=source_validation.columns)

X_validationSet.to_csv(inDir+"/X_LSTM_validationSet.csv", index=False)



del model
# configure
n_lag = 180
n_seq = 60
n_test = 20
n_epochs = 100
n_batch = 1
n_neurons = 1
# prepare data
X10 = pd.DataFrame(X1[0])

scaler, train, test = prepare_data(X10, n_test, n_lag, n_seq)



#X10 = X1[0]
#
#window_size = 90
#
#X101 = np.atleast_2d(np.array([X10[start:start + window_size] for start in range(0, X10.shape[0] - window_size)]))
#
#
#
#X1 = np.array(X)
#
#X20 = X1[1]
#
#window_size = 90
#
#X201 = np.atleast_2d(np.array([X20[start:start + window_size] for start in range(0, X20.shape[0] - window_size)]))
#
#X_train = []
#
#X_train.append(X101)
#X_train.append(X201)
#
#X_train = np.atleast_3d(X_train)
#
#X_train[1]

X_train = []

features_size = 90
for nb_sample in range(0,X.shape[0]):
    print(nb_sample)
    current_sample = []
    current_sample = X1[nb_sample]
    current_sample_format = []
    current_sample_format = np.atleast_2d(np.array([current_sample[start:start + features_size] for start in range(0, current_sample.shape[0] - features_size)]))
    X_train.append(current_sample_format)
X_train = np.array(X_train)
    
Y_train = np.array(Y)
test_size = int(0.2 * X.shape[0])           # In real life you'd want to use 0.2 - 0.5
X_build, X_valid, y_build, y_valid = X_train[:-test_size], X_train[-test_size:], Y_train[:-test_size], Y_train[-test_size:]

X_build = np.atleast_3d(X_build)
# fit an LSTM network to training data
def LSTM_Model(train, y, n_batch, nb_epoch, n_neurons):

	# design network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train.shape[1], train.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

model = LSTM_Model(X_build, y_build,n_batch = 1, nb_epoch = 25, n_neurons = 10 )
	# fit network
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
		model.reset_states()

from __future__ import print_function, division

import numpy as np
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential
    
def make_timeseries_regressor(window_size, filter_length, nb_input_series=1, nb_outputs=1, nb_filter=4):
    """:Return: a Keras Model for predicting the next value in a timeseries given a fixed-size lookback window of previous values.
    The model can handle multiple input timeseries (`nb_input_series`) and multiple prediction targets (`nb_outputs`).
    :param int window_size: The number of previous timeseries values to use as input features.  Also called lag or lookback.
    :param int nb_input_series: The number of input timeseries; 1 for a single timeseries.
      The `X` input to ``fit()`` should be an array of shape ``(n_instances, window_size, nb_input_series)``; each instance is
      a 2D array of shape ``(window_size, nb_input_series)``.  For example, for `window_size` = 3 and `nb_input_series` = 1 (a
      single timeseries), one instance could be ``[[0], [1], [2]]``. See ``make_timeseries_instances()``.
    :param int nb_outputs: The output dimension, often equal to the number of inputs.
      For each input instance (array with shape ``(window_size, nb_input_series)``), the output is a vector of size `nb_outputs`,
      usually the value(s) predicted to come after the last value in that input instance, i.e., the next value
      in the sequence. The `y` input to ``fit()`` should be an array of shape ``(n_instances, nb_outputs)``.
    :param int filter_length: the size (along the `window_size` dimension) of the sliding window that gets convolved with
      each position along each instance. The difference between 1D and 2D convolution is that a 1D filter's "height" is fixed
      to the number of input timeseries (its "width" being `filter_length`), and it can only slide along the window
      dimension.  This is useful as generally the input timeseries have no spatial/ordinal relationship, so it's not
      meaningful to look for patterns that are invariant with respect to subsets of the timeseries.
    :param int nb_filter: The number of different filters to learn (roughly, input patterns to recognize).
    """
    model = Sequential((
        # The first conv layer learns `nb_filter` filters (aka kernels), each of size ``(filter_length, nb_input_series)``.
        # Its output will have shape (None, window_size - filter_length + 1, nb_filter), i.e., for each position in
        # the input timeseries, the activation of each filter at that position.
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu', input_shape=(window_size, nb_input_series)),
        MaxPooling1D(),     # Downsample the output of convolution by 2X.
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dense(nb_outputs, activation='linear'),     # For binary classification, change the activation to 'sigmoid'
    ))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # To perform (binary) classification instead:
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model 
   
window_size = X_build.shape[1]
input_nb_series = X_build.shape[2]
filter_length = 5
nb_filter = 4
output_nb_series = y_build.shape[1]
    
model = make_timeseries_regressor(window_size=window_size, filter_length=filter_length, nb_input_series=input_nb_series, nb_outputs=output_nb_series, nb_filter=nb_filter)

print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape, model.output_shape, nb_filter, filter_length))
model.summary()


model.fit(X_build, y_build, nb_epoch=25, batch_size=2, validation_data=(X_valid, y_valid))

pred = model.predict(X_build)
pred_cv = model.predict(X_valid)




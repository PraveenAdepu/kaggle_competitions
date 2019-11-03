# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:51:19 2017

@author: PA23309
"""

import pandas as pd
import numpy as np
import datetime

d = lambda x: pd.datetime.strptime(x, '%d %b %y')
dateparse = lambda x: d(x)

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
Y = source_pivot[source_pivot.columns[-horizon_periods:]].copy()
X = source_pivot[source_pivot.columns[:-horizon_periods]].copy()

X_id = pd.DataFrame(X["device_mmc"])

del X["device_mmc"]

X1 = np.array(X)

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
X_train = np.atleast_3d(X_train)
    
Y_train = np.array(Y)
test_size = int(0.2 * X.shape[0])           # In real life you'd want to use 0.2 - 0.5
X_build, X_valid, y_build, y_valid, X_build_id, X_valid_id = X_train[:-test_size], X_train[-test_size:], Y_train[:-test_size], Y_train[-test_size:], X_id[:-test_size], X_id[-test_size:]

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

pred    = model.predict(X_build)
pred_cv = model.predict(X_valid)

X_validation = pd.merge(source_pivot, X_valid_id, how='inner',on='device_mmc')

X_validation_columns = source_pivot[source_pivot.columns[-horizon_periods:]].columns

X_pred_cv = pd.DataFrame(pred_cv)

X_pred_cv.columns = source_pivot[source_pivot.columns[-horizon_periods:]].columns

X_valid_id = X_valid_id.reset_index()

del X_valid_id['index']

X_validationSet = pd.concat([X_valid_id, X_pred_cv], axis = 1)

X_pred_cv.columns
X_validationSet = pd.melt(X_validationSet, id_vars='device_mmc', value_vars=X_pred_cv.columns)

X_validationSet.to_csv(inDir+"/X_validationSet.csv", index=False)

import matplotlib.pylab as plt
import seaborn as sns
sns.despine()

plt.figure() 
plt.plot(model.history['loss']) 
plt.plot(model.history['val_loss']) 
plt.title('model loss') 
plt.ylabel('loss') 
plt.xlabel('epoch') 
plt.legend(['train', 'test'], loc='best') 
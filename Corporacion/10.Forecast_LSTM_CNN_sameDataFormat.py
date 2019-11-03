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

import pandas as pd
import matplotlib.pylab as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.callbacks import EarlyStopping
from keras.initializers import *

import seaborn as sns
sns.despine()

import numba as nb


inDir = 'C:\\Users\\SriPrav\\Documents\\R\\34Corporacion'

df_train = pd.read_csv(inDir+'/input/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # 2016-01-01
)

df_train = df_train[df_train['date']>="2017-05-01"]

#df_train = df_train[df_train["store_nbr"]==1]

df_train["store_item"] = df_train["store_nbr"].map(str)+"-"+df_train["item_nbr"].map(str)
df_train.head()

df_train.onpromotion = df_train.onpromotion.astype(int)


source_pivot_sales = pd.pivot_table(df_train, values = 'unit_sales', index=['store_item'], columns = 'date').reset_index()

cols = [col for col in source_pivot_sales.columns if col not in ['store_item']]
source_pivot_sales_long = pd.melt(source_pivot_sales, id_vars=['store_item'], value_vars=cols,var_name='date', value_name='unit_sales')

source_pivot_sales_long.fillna(0, inplace = True)

source_pivot_sales_long["unit_sales_rolling7_mean"]= source_pivot_sales_long.groupby('store_item')['unit_sales'].apply(lambda x:x.rolling(center=False,window=7).mean())
source_pivot_sales_long["unit_sales_rolling14_mean"]= source_pivot_sales_long.groupby('store_item')['unit_sales'].apply(lambda x:x.rolling(center=False,window=14).mean())
source_pivot_sales_long["unit_sales_rolling21_mean"]= source_pivot_sales_long.groupby('store_item')['unit_sales'].apply(lambda x:x.rolling(center=False,window=21).mean())
source_pivot_sales_long["unit_sales_rolling28_mean"]= source_pivot_sales_long.groupby('store_item')['unit_sales'].apply(lambda x:x.rolling(center=False,window=28).mean())

source_pivot_sales_long.fillna(0, inplace = True)

#test1 = source_pivot_sales_long[source_pivot_sales_long["store_item"]=="1-1000866"]

#del test1

source_pivot_sales_long["unit_sales_rolling7_mean"] = (source_pivot_sales_long["unit_sales_rolling7_mean"] - source_pivot_sales_long["unit_sales_rolling7_mean"].mean())/source_pivot_sales_long["unit_sales_rolling7_mean"].std()
source_pivot_sales_long["unit_sales_rolling14_mean"] = (source_pivot_sales_long["unit_sales_rolling14_mean"] - source_pivot_sales_long["unit_sales_rolling14_mean"].mean())/source_pivot_sales_long["unit_sales_rolling14_mean"].std()
source_pivot_sales_long["unit_sales_rolling21_mean"] = (source_pivot_sales_long["unit_sales_rolling21_mean"] - source_pivot_sales_long["unit_sales_rolling21_mean"].mean())/source_pivot_sales_long["unit_sales_rolling21_mean"].std()
source_pivot_sales_long["unit_sales_rolling28_mean"] = (source_pivot_sales_long["unit_sales_rolling28_mean"] - source_pivot_sales_long["unit_sales_rolling28_mean"].mean())/source_pivot_sales_long["unit_sales_rolling28_mean"].std()

mean = source_pivot_sales_long.unit_sales.mean()
std =  source_pivot_sales_long.unit_sales.std()

source_pivot_sales_long["unit_sales"] = (source_pivot_sales_long["unit_sales"] - mean) / std


source_pivot_promotion = pd.pivot_table(df_train, values = 'onpromotion', index=['store_item'], columns = 'date').reset_index()

source_pivot_promotion_long = pd.melt(source_pivot_promotion, id_vars=['store_item'], value_vars=cols,var_name='date', value_name='onpromotion')

source_pivot_promotion_long.fillna(0, inplace = True)

source_pivot_sales_long = pd.merge(source_pivot_sales_long, source_pivot_promotion_long ,on=("store_item","date"),how="left")

source_pivot_sales_long.head()


source_pivot_sales_long['dow'] = source_pivot_sales_long['date'].dt.weekday_name

source_pivot_sales_long = pd.get_dummies(source_pivot_sales_long, prefix=['dow'], columns=['dow'])

#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()
#le.fit(source_pivot_sales_long['dow'])
#source_pivot_sales_long['dow'] = le.transform(source_pivot_sales_long['dow']) 
#
#source_pivot_sales_long['dow'].unique()

#for storeitem in source_pivot_sales_long["store_item"].unique():
#    print(storeitem)

storeitem_sample = ["1-1000866","1-1001305","1-1003679","1-1004545","1-1004550","1-1004551","1-1009512","1-1009539","1-1009997","1-1009998"]

#storeitem = "1-1000866"

#store_count = 0
##for storeitem in storeitem_sample:
#for storeitem in source_pivot_sales_long["store_item"].unique():
#    print(storeitem)
#    current_df = []
#    current_df = source_pivot_sales_long[source_pivot_sales_long["store_item"]==storeitem].values
#    
#    nb_samples = len(current_df) - n_sequence - n_forecast
#    
#    current_df = current_df[:,2:]
#    
#    current_train = [np.expand_dims(np.atleast_2d(current_df[i:n_sequence+i,:]), axis=0) for i in range(nb_samples)]
#    current_train_mat = np.concatenate(current_train, axis=0)
#    
#    # target - the first column in merged dataframe
#    current_target = [np.atleast_2d(current_df[i+n_sequence:n_sequence+i+n_forecast,0]) for i in range(nb_samples)]
#    current_target_mat = np.concatenate(current_target, axis=0)
#    
#    if store_count == 0:
#        X_train = current_train_mat
#        y_train = current_target_mat
#    else:
#        X_train = np.concatenate([X_train,current_train_mat])
#        y_train = np.concatenate([y_train,current_target_mat])
#    
#    store_count = store_count + 1
#
#def CNN_data_preparation(source_pivot_sales_long,n_sequence,n_forecast):
#    X_train = np.array([])
#    y_train = np.array([])
#    store_count = 0
#    for storeitem in storeitem_sample:    
#    #for storeitem in source_pivot_sales_long["store_item"].unique():
#        print(storeitem)
#        current_df = []
#        current_df = source_pivot_sales_long[source_pivot_sales_long["store_item"]==storeitem].values        
#        nb_samples = len(current_df) - n_sequence - n_forecast        
#        current_df = current_df[:,2:]
#        
#        current_train = [np.expand_dims(np.atleast_2d(current_df[i:n_sequence+i,:]), axis=0) for i in range(nb_samples)]
#        current_train_mat = np.concatenate(current_train, axis=0)
#        
#        # target - the first column in merged dataframe
#        current_target = [np.atleast_2d(current_df[i+n_sequence:n_sequence+i+n_forecast,0]) for i in range(nb_samples)]
#        current_target_mat = np.concatenate(current_target, axis=0)
#        
#        if store_count == 0:
#            X_train = current_train_mat
#            y_train = current_target_mat
#        else:
#            X_train = np.concatenate([X_train,current_train_mat])
#            y_train = np.concatenate([y_train,current_target_mat])
#        
#        store_count = store_count + 1
#    return X_train, y_train

#%timeit X_train , y_train = CNN_data_preparation(source_pivot_sales_long,n_sequence,n_forecast)

@nb.jit
def CNN_data_preparation_currentSet_numba1(current_df,nb_samples):
    X_current = np.array([])
    y_current = np.array([])
    for i in range(nb_samples):
        current_slice = current_df[i:n_sequence+i,:]
        current_y = current_df[i+n_sequence:n_sequence+i+n_forecast,0]
        if i == 0:
            X_current = current_slice
            y_current = current_y
        else:
            X_current = np.concatenate([X_current,current_slice])
            y_current = np.concatenate([y_current,current_y])
    return  X_current, y_current   
                 
@nb.jit
def CNN_data_preparation_currentSet_numba2(current_df,nb_samples):           
    current_train = [np.expand_dims(np.atleast_2d(current_df[i:n_sequence+i,:]), axis=0) for i in range(nb_samples)]       
    # target - the first column in merged dataframe
    current_target = [np.atleast_2d(current_df[i+n_sequence:n_sequence+i+n_forecast,0]) for i in range(nb_samples)]    
    return current_train, current_target

def CNN_data_preparation_currentSet_numpy(current_df,nb_samples):           
    current_train = [np.expand_dims(np.atleast_2d(current_df[i:n_sequence+i,:]), axis=0) for i in range(nb_samples)]       
    # target - the first column in merged dataframe
    current_target = [np.atleast_2d(current_df[i+n_sequence:n_sequence+i+n_forecast,0]) for i in range(nb_samples)]    
    return current_train, current_target
        
def CNN_data_preparation(source_pivot_sales_long,n_sequence,n_forecast):
    X_train = np.array([])
    y_train = np.array([])
    store_count = 0    
    #for storeitem in storeitem_sample:
    for storeitem in source_pivot_sales_long["store_item"].unique():
        #print(storeitem)
        current_df = []
        
        current_df = source_pivot_sales_long[source_pivot_sales_long["store_item"]==storeitem].values
        nb_samples = len(current_df) - n_sequence - n_forecast
        current_df = current_df[:,2:]
        #%timeit current_train, current_target = CNN_data_preparation_currentSet_numpy(current_df,nb_samples)
        #%timeit current_train, current_target = CNN_data_preparation_currentSet_numba1(current_df,nb_samples)
        #%timeit current_train, current_target = CNN_data_preparation_currentSet_numba2(current_df,nb_samples)
        current_train, current_target = CNN_data_preparation_currentSet_numpy(current_df,nb_samples)
        current_train_mat = np.concatenate(current_train, axis=0) 
        current_target_mat = np.concatenate(current_target, axis=0)
        
        if store_count == 0:
            X_train = current_train_mat
            y_train = current_target_mat
        else:
            X_train = np.concatenate([X_train,current_train_mat])
            y_train = np.concatenate([y_train,current_target_mat])
        
        store_count = store_count + 1
        
        if (store_count % 100) == 0:
            print("processed {}".format(store_count))
    return X_train, y_train

n_sequence = 60
n_features = 13
n_forecast = 15

X_train = np.array([])
y_train = np.array([])

X_train , y_train = CNN_data_preparation(source_pivot_sales_long,n_sequence,n_forecast)

    
def create_Xt_Yt(X, y, percentage=0.9):
    p = int(len(X) * percentage)
    X_train = X[0:p]
    Y_train = y[0:p]
     
    #X_train, Y_train = shuffle_in_unison(X_train, Y_train)
 
    X_test = X[p:]
    Y_test = y[p:]

    return X_train, X_test, Y_train, Y_test

X_build, X_valid, y_build, y_valid = create_Xt_Yt(X_train, y_train)

def LSTM_Model(train, y, n_batch, nb_epoch, n_neurons):
    #train = train.reshape(train.shape[0], 1, train.shape[1])
	# design network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train.shape[1], train.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def LSTM_Model(train, y, n_batch, n_neurons):    
    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(train.shape[1], train.shape[2])))
    model.add(Dropout(.2))
    model.add(Dense(y.shape[1]))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def CNN_Model(train, y):
    model = Sequential()
    model.add(Convolution1D(input_shape = (train.shape[1], train.shape[2]),
                            nb_filter=16,
                            filter_length=4,
                            border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
       
    model.add(Convolution1D(nb_filter=8,
                            filter_length=4,
                            border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
      
    model.add(Dense(y.shape[1]))
    model.add(Activation('sigmoid'))
    #model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#model = LSTM_Model(X_build, y_build,n_batch = 1, nb_epoch = 250, n_neurons = 10 )
model = CNN_Model(X_build, y_build )

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.summary()

batch_size = 1000
n_epochs = 25
VERBOSEFLAG = 1
callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, verbose=VERBOSEFLAG),
            ]

model.fit(X_build, y_build, epochs=n_epochs, batch_size=batch_size
          , validation_data=(X_valid, y_valid),verbose=1, shuffle=False
          ,callbacks=callbacks)

pred_cv = model.predict(X_valid, batch_size = batch_size, verbose = 1)

pred_cv = pred_cv * std
pred_cv = pred_cv + mean

y_valid = y_valid * std
y_valid = y_valid + mean

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_valid, pred_cv))
rms

# Prav - convinced to use target normalisation transformation everytime

# rms : target normalization - 0.7754568909558804
# 0.6916685848078676

#plt.figure()
#plt.plot(model.history['loss'])
#plt.plot(model.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='best')
#plt.show()
#
#plt.figure()
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='best')
#plt.show()



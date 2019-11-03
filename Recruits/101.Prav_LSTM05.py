# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:31:21 2017

@author: SriPrav
"""



from sklearn import ensemble, neighbors, linear_model, metrics, preprocessing


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


inDir =r'C:\Users\SriPrav\Documents\R\40Recruit'

data = {
    'tra': pd.read_csv(inDir + '/input/air_visit_data.csv'),
    'as': pd.read_csv(inDir + '/input/air_store_info.csv'),
    'hs': pd.read_csv(inDir + '/input/hpg_store_info.csv'),
    'ar': pd.read_csv(inDir + '/input/air_reserve.csv'),
    'hr': pd.read_csv(inDir + '/input/hpg_reserve.csv'),
    'id': pd.read_csv(inDir + '/input/store_id_relation.csv'),
    'tes': pd.read_csv(inDir + '/input/sample_submission.csv'),
    'hol': pd.read_csv(inDir + '/input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    data[df] = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date'})
    print(data[df].head())

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date


data['build'] = data['tra'][(data['tra']['visit_date']<datetime.date(2017,3,12))] 
data['valid'] = data['tra'][(data['tra']['visit_date']>= datetime.date(2017,3,12)) & (data['tra']['visit_date']<= datetime.date(2017,4,19)) ]

unique_stores = data['valid']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

#data['build'].to_csv(inDir+"/input/build.csv", index=False)
#data['valid'].to_csv(inDir+"/input/valid.csv", index=False)
#stores.to_csv(inDir+"/input/build_stores.csv", index=False)

#sure it can be compressed...
tmp = data['build'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 
tmp = data['build'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['build'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['build'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['build'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 
#tmp = data['build'].groupby(['air_store_id','dow'], as_index=False)['visitors'].std().rename(columns={'visitors':'std_visitors'})
#stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 
lbl = preprocessing.LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])


data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
X_build = pd.merge(data['build'], data['hol'], how='left', on=['visit_date']) 
X_valid = pd.merge(data['valid'], data['hol'], how='left', on=['visit_date']) 

X_build = pd.merge(X_build, stores, how='left', on=['air_store_id','dow']) 
X_valid = pd.merge(X_valid , stores, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    X_build = pd.merge(X_build, data[df], how='left', on=['air_store_id','visit_date']) 
    X_valid = pd.merge(X_valid, data[df], how='left', on=['air_store_id','visit_date'])
    
print(X_build.describe())
print(X_build.head())

col = [c for c in X_build if c not in ['id', 'air_store_id','visit_date','visitors']]
X_build = X_build.fillna(-1)
X_valid = X_valid.fillna(-1)

feature_names = col


X1 = X_build[['air_store_id','visit_date','visitors','day_of_week','air_genre_name','air_area_name']]
X2 = X_valid[['air_store_id','visit_date','visitors','day_of_week','air_genre_name','air_area_name']]


X = pd.concat([X1,X2], axis = 0)

Prav_5fold_CVindices = pd.read_csv(inDir+"/input/Prav_5folds_CVindices.csv")

source_pivot = pd.merge(X, Prav_5fold_CVindices, how='left',on="air_store_id")

source_pivot = pd.get_dummies(source_pivot,  columns=['day_of_week', 'air_genre_name','air_area_name'])

#source_train_id = pd.DataFrame(source_pivot[["air_store_id","CVindices"]])
#del source_pivot["air_store_id"]
#del source_pivot["CVindices"]

col = [c for c in source_pivot if c not in ['CVindices', 'air_store_id','visit_date']]

sequence = 180
y_examples = 39 


X_train = []
y_train = []


store_count = 0

unique_stores_LSTM = source_pivot['air_store_id'].unique()

for test_stores in unique_stores_LSTM:
    
    data = source_pivot[source_pivot['air_store_id'] == test_stores] #train[air_store_id == "air_ba937bf13d40fb24"]
    data = data.sort_values('visit_date')
    current_data = data[col].values
    current_data_y = data[["visitors"]].values
    nb_samples = len(current_data) - sequence - y_examples
    print(test_stores, nb_samples,' \n')
    if nb_samples >= sequence:
        for nb_sample in range(nb_samples):
    #        print(nb_sample)
            current_sample_format = []
            current_sample_format = np.atleast_2d(np.array([current_data[nb_sample:nb_sample + sequence,:]]))
            if nb_sample ==0 and store_count == 0:
                X_train = current_sample_format
            else:
                X_train = np.append(X_train, current_sample_format, axis=0)
            
        target_list = [np.atleast_2d(current_data_y[i+sequence:i+sequence+y_examples,0]) for i in range(nb_samples)]
        if store_count == 0:
            y_train = np.concatenate(target_list, axis=0)
            y_train =  y_train.reshape(y_train.shape[0],1, y_train.shape[1])
        else:
            y_train = np.append(y_train,np.atleast_2d(np.array(target_list)), axis=0)
        store_count = store_count + 1
        print('success \n')

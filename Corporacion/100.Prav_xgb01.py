# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:16:48 2017

@author: SriPrav
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime


inDir = 'C:\\Users\\SriPrav\\Documents\\R\\34Corporacion'

dtypes = {'store_nbr': np.dtype('int64'),
          'item_nbr': np.dtype('int64'),
          'unit_sales': np.dtype('float64'),
          'onpromotion': np.dtype('O')}

train = pd.read_csv(inDir + '/input/train.csv', index_col='id', parse_dates=['date'], dtype=dtypes)

validation = train[train['date']>='2017-08-01']

train = train[train['date']< '2017-08-01']

# If done on all train data, results in 367m rows. So, we're taking a small sample:
date_mask = (train['date'] >= '2017-04-01') & (train['date'] < '2017-08-01')
print(train.shape)
train = train[date_mask]
print(train.shape)

orig_train_index = train.index # will need later

# Bracket the dates
max_date = train['date'].max()  
min_date = train['date'].min()  
days = (max_date - min_date).days + 1 

# Master list of dates
dates = [min_date + datetime.timedelta(days=x) for x in range(days)]
dates.sort()

# Master list of stores
unique_stores = list(set(train['store_nbr'].unique())) # | set(test['Store'].unique()))
unique_stores.sort()
num_unique_stores = len(unique_stores)

# Master list of Items
unique_items = list(set(train['item_nbr'].unique())) # | set(test['item_nbr'].unique()))
unique_items.sort()
num_unique_items = len(unique_items)

# Unique Date / Store index
date_index = np.repeat(dates, num_unique_stores * num_unique_items) # num dates * num stores * num items
store_index = np.concatenate([np.repeat(unique_stores, num_unique_items)] * days)
item_index = np.concatenate([unique_items] * days * num_unique_stores)

print(len(date_index))
print(len(store_index))
print(len(item_index))

start = train.index.tolist()[0]
new_train_index = list(range(len(item_index)))
new_train_index = new_train_index + start

train_new = pd.DataFrame(index=new_train_index, columns=train.columns)

train_new['date'] = date_index
train_new['store_nbr'] = store_index
train_new['item_nbr'] = item_index

train_new.index.name = 'id'

# Set the indexes (makes it easy to insert data into new)
train_new.set_index(['date', 'store_nbr', 'item_nbr'], drop=True, inplace=True)
train.set_index(['date', 'store_nbr', 'item_nbr'], drop=True, inplace=True)

# Update the master index with and train
train_new.update(train)
train_new.reset_index(inplace=True)

# Return the original train back to normal
train.reset_index(inplace=True)
train.set_index(orig_train_index, inplace=True)

# Fill the created unit sales with zero
train_new['unit_sales'].fillna(0, inplace=True)

print(train.shape)
print(train_new.shape)



del train
print(train_new.shape)
print(validation.shape)
train_new.head()
validation.head()

train_new = train_new.append(validation)

stores = pd.read_csv(inDir + '/input/stores.csv')
items = pd.read_csv(inDir + '/input/items.csv')
oil = pd.read_csv(inDir + '/input/oil.csv', parse_dates=['date'])
holidays_events = pd.read_csv(inDir + '/input/holidays_events.csv', parse_dates=['date'])

stores.head()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

stores.dtypes
# Iterating over all the common columns in train and test
for col in stores.columns.values:
   # Encoding only categorical variables
   if stores[col].dtypes=='object':
   # Using whole data to form an exhaustive list of levels
       data=stores[col]
       le.fit(data.values)
       stores[col]=le.transform(stores[col])

items.dtypes
items.head()
# Iterating over all the common columns in train and test
for col in items.columns.values:
   # Encoding only categorical variables
   if items[col].dtypes=='object':
   # Using whole data to form an exhaustive list of levels
       data=items[col]
       le.fit(data.values)
       items[col]=le.transform(items[col])

oil.dtypes
oil.head()

holidays_events.dtypes
holidays_events.head()

holidays_events['date'].max()
# Iterating over all the common columns in train and test
for col in holidays_events.columns.values:
   # Encoding only categorical variables
   if holidays_events[col].dtypes=='object':
   # Using whole data to form an exhaustive list of levels
       data=holidays_events[col]
       le.fit(data.values)
       holidays_events[col]=le.transform(holidays_events[col])       

for col in holidays_events.columns.values:
   # Encoding only categorical variables
   if holidays_events[col].dtypes=='bool':
   # Using whole data to form an exhaustive list of levels
       data=holidays_events[col]
       le.fit(data.values)
       holidays_events[col]=le.transform(holidays_events[col]) 
       
       
       
trainingSet = pd.merge(train_new, stores, how='left', on='store_nbr')
trainingSet = pd.merge(trainingSet, items, how='left', on='item_nbr')
trainingSet = pd.merge(trainingSet, oil, how='left', on='date')
print(holidays_events.shape)
holidays_events = holidays_events.drop_duplicates(subset='date', keep="last")
print(holidays_events.shape)
trainingSet = pd.merge(trainingSet, holidays_events, how='left', on='date')

trainingSet.dtypes
trainingSet.tail()

trainingSet['onpromotion'] = trainingSet['onpromotion'].fillna('')
trainingSet['dcoilwtico'] = trainingSet['dcoilwtico'].fillna(0)




#for col in trainingSet.columns.values:
#   # Encoding only categorical variables
#   if trainingSet[col].dtypes=='object':
#   # Using whole data to form an exhaustive list of levels
#       data=trainingSet[col]
#       le.fit(data.values)
#       trainingSet[col]=le.transform(trainingSet[col]) 

## Importing LabelEncoder and initializing it
#from sklearn.preprocessing import LabelEncoder
#le=LabelEncoder()
## Iterating over all the common columns in train and test
#for col in X_test.columns.values:
#   # Encoding only categorical variables
#   if X_test[col].dtypes=='object':
#   # Using whole data to form an exhaustive list of levels
#       data=X_train[col].append(X_test[col])
#       le.fit(data.values)
#       X_train[col]=le.transform(X_train[col])
#       X_test[col]=le.transform(X_test[col])



trainingSet['CVindices'] = np.where(trainingSet['date']>='2017-08-01', 1, 2)

testingSet = pd.read_csv(inDir + '/input/test.csv',  parse_dates=['date'], dtype=dtypes)
testingSet = pd.merge(testingSet, stores, how='left', on='store_nbr')
testingSet = pd.merge(testingSet, items, how='left', on='item_nbr')
testingSet = pd.merge(testingSet, oil, how='left', on='date')
testingSet = pd.merge(testingSet, holidays_events, how='left', on='date')

testingSet.dtypes
testingSet['onpromotion'] = testingSet['onpromotion'].fillna('')
testingSet['dcoilwtico']  = testingSet['dcoilwtico'].fillna(0)

trainingSet.head()
testingSet.head()

testingSet.columns
data=trainingSet['onpromotion'].append(testingSet['onpromotion'])
le.fit(data.values)
trainingSet['onpromotion']=le.transform(trainingSet['onpromotion']) 
testingSet['onpromotion']=le.transform(testingSet['onpromotion']) 

trainingSet['onpromotion'].unique()
testingSet['onpromotion'].unique()

trainingSet['Month'] = trainingSet['date'].dt.month
trainingSet['Day'] = trainingSet['date'].dt.day
trainingSet['WeekDay'] = trainingSet['date'].dt.weekday

testingSet['Month'] = testingSet['date'].dt.month
testingSet['Day'] = testingSet['date'].dt.day
testingSet['WeekDay'] = testingSet['date'].dt.weekday

trainingSet['onpromotion'].unique()
testingSet['onpromotion'].unique()

trainingSet.loc[(trainingSet.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
trainingSet['unit_sales'] =  trainingSet['unit_sales'].apply(pd.np.log1p) #logarithm conversion

trainingSet = trainingSet.fillna(-1)
testingSet = testingSet.fillna(-1)


feature_names = [c for c in trainingSet.columns if c not in ['date','unit_sales','CVindices']]

del train_new, validation, data, date_index, date_mask

lgbm_params = {
          'task'              : 'train',
          'boosting_type'     : 'gbdt',
          'objective'         : 'regression',
          'metric'            : 'rmse',
          'num_leaves'        : 2**7, #2**4,
          'feature_fraction'  : 0.8,
          'bagging_fraction'  : 0.8,
          'bagging_freq'      : 10,#2
          'learning_rate'     : 0.01,
          'tree_method'       : 'exact',
          'min_data_in_leaf'  : 500,             
          'nthread'           : 25,
          'silent'            : False,
          'seed'              : 2017,
         }
lgbm_num_round = 1150
lgbm_early_stopping_rounds = 250

seed = 2017
folds = 1
nbags = 1
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2017
verboseeval=250

lgbmModelName = 'lgbm001'
xgbModelName  = 'xgb002'
rfModelName   = 'rf002'
etModelName   = 'et002'
fmModelName   = 'fm100'
adaModelName  = 'ada001'
gbdtModelName = 'gbdt001'

for i in range(1, folds+1):
        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)
        fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)

train = pd.read_csv(inDir + '/input/train.csv', index_col='id', parse_dates=['date'], dtype=dtypes)

# If done on all train data, results in 367m rows. So, we're taking a small sample:
date_mask = (train['date'] >= '2017-04-01') & (train['date'] <= '2017-08-15')
print(train.shape)
train = train[date_mask]
print(train.shape)

orig_train_index = train.index # will need later

# Bracket the dates
max_date = train['date'].max()  
min_date = train['date'].min()  
days = (max_date - min_date).days + 1 

# Master list of dates
dates = [min_date + datetime.timedelta(days=x) for x in range(days)]
dates.sort()

# Master list of stores
unique_stores = list(set(train['store_nbr'].unique())) # | set(test['Store'].unique()))
unique_stores.sort()
num_unique_stores = len(unique_stores)

# Master list of Items
unique_items = list(set(train['item_nbr'].unique())) # | set(test['item_nbr'].unique()))
unique_items.sort()
num_unique_items = len(unique_items)

# Unique Date / Store index
date_index = np.repeat(dates, num_unique_stores * num_unique_items) # num dates * num stores * num items
store_index = np.concatenate([np.repeat(unique_stores, num_unique_items)] * days)
item_index = np.concatenate([unique_items] * days * num_unique_stores)

print(len(date_index))
print(len(store_index))
print(len(item_index))

start = train.index.tolist()[0]
new_train_index = list(range(len(item_index)))
new_train_index = new_train_index + start

train_new = pd.DataFrame(index=new_train_index, columns=train.columns)

train_new['date'] = date_index
train_new['store_nbr'] = store_index
train_new['item_nbr'] = item_index

train_new.index.name = 'id'

# Set the indexes (makes it easy to insert data into new)
train_new.set_index(['date', 'store_nbr', 'item_nbr'], drop=True, inplace=True)
train.set_index(['date', 'store_nbr', 'item_nbr'], drop=True, inplace=True)

# Update the master index with and train
train_new.update(train)
train_new.reset_index(inplace=True)

# Return the original train back to normal
train.reset_index(inplace=True)
train.set_index(orig_train_index, inplace=True)

# Fill the created unit sales with zero
train_new['unit_sales'].fillna(0, inplace=True)

print(train.shape)
print(train_new.shape)
del train

trainingSet = pd.merge(train_new, stores, how='left', on='store_nbr')
trainingSet = pd.merge(trainingSet, items, how='left', on='item_nbr')
trainingSet = pd.merge(trainingSet, oil, how='left', on='date')
trainingSet = pd.merge(trainingSet, holidays_events, how='left', on='date')


trainingSet['onpromotion'] = trainingSet['onpromotion'].fillna('')
trainingSet['dcoilwtico'] = trainingSet['dcoilwtico'].fillna(0)

data=trainingSet['onpromotion']
le.fit(data.values)
trainingSet['onpromotion']=le.transform(trainingSet['onpromotion']) 

trainingSet['onpromotion'].unique()


trainingSet['Month'] = trainingSet['date'].dt.month
trainingSet['Day'] = trainingSet['date'].dt.day
trainingSet['WeekDay'] = trainingSet['date'].dt.weekday


trainingSet.loc[(trainingSet.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
trainingSet['unit_sales'] =  trainingSet['unit_sales'].apply(pd.np.log1p) #logarithm conversion

trainingSet = trainingSet.fillna(-1)

trainingSet.columns
fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)
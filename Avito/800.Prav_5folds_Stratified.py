# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:31:46 2017

@author: PA23309
"""

import pandas as pd
import numpy as np


random_state = 20180512
np.random.RandomState(random_state)

from sklearn.model_selection import StratifiedKFold


inDir = 'C:/Users/SriPrav/Documents/R/48Avito'

train = pd.read_csv(inDir+'/input/train.csv')
test  = pd.read_csv(inDir+'/input/test.csv')

train.shape #(1503424, 18)
test.shape  #(508438, 17)

train_subset = train[train['activation_date'] <= "2017-03-30"]

train_subset = train_subset.reset_index(drop=True)

train_subset['weekday'] = pd.to_datetime(train_subset['activation_date']).dt.weekday_name

train_subset.head()

train_subset.groupby(['activation_date','weekday']).size()

trainfoldSource = train_subset[['item_id','weekday']]

folds = 5
skf = StratifiedKFold(n_splits=folds, random_state = random_state, shuffle = True)

skf.get_n_splits(trainfoldSource, trainfoldSource['weekday'])

print(skf) 

Prav_CVindices = pd.DataFrame(columns=['index','CVindices'])

count = 1
for train_index, test_index in skf.split(trainfoldSource, trainfoldSource['weekday']):           
       df_index = pd.DataFrame(test_index,columns=['index']) 
       df_index['CVindices'] = count
       Prav_CVindices = Prav_CVindices.append(df_index)       
       count+=1
       
Prav_CVindices.set_index('index', inplace=True)
#Prav_CVindices.sort_index(inplace=True)

trainfoldSource = pd.merge(trainfoldSource, Prav_CVindices, left_index=True, right_index=True)

trainfoldSource.groupby(['CVindices','weekday'])[['weekday']].size()

del trainfoldSource['weekday']

trainfoldSource[['item_id','CVindices']].to_csv(inDir+"/input/Prav_5folds_CVindices_weekdayStratified.csv", index=False)

    
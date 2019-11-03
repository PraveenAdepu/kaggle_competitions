# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:31:46 2017

@author: PA23309
"""

import pandas as pd
import numpy as np


random_state = 2017
np.random.RandomState(random_state)
from sklearn.model_selection import StratifiedKFold, KFold


inDir =r'C:\Users\SriPrav\Documents\R\40Recruit'

source = pd.read_csv(inDir+"/input/air_visit_data.csv")


source_pivot = pd.pivot_table(source, values = 'visitors', index=['air_store_id'], columns = 'visit_date').reset_index()
source_pivot.fillna(0, inplace = True)


trainfoldSource = pd.DataFrame(source_pivot["air_store_id"])

folds = 5
skf = KFold(n_splits=folds, random_state = random_state, shuffle = True)

skf.get_n_splits(trainfoldSource)

print(skf) 

Prav_CVindices = pd.DataFrame(columns=['index','CVindices'])

count = 1
for train_index, test_index in skf.split(trainfoldSource):           
       df_index = pd.DataFrame(test_index,columns=['index']) 
       df_index['CVindices'] = count
       Prav_CVindices = Prav_CVindices.append(df_index)       
       count+=1
       
Prav_CVindices.set_index('index', inplace=True)

trainfoldSource = pd.merge(trainfoldSource, Prav_CVindices, left_index=True, right_index=True)

trainfoldSource.groupby(['CVindices']).count()


trainfoldSource.to_csv(inDir+"/input/Prav_5folds_CVindices.csv", index=False)

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:31:46 2017

@author: PA23309
"""

import pandas as pd
import numpy as np


random_state = 2017
np.random.RandomState(random_state)
from sklearn.model_selection import StratifiedKFold

inDir = 'C:\\Users\\SriPrav\\Documents\R\\37TFSpeech'

train = pd.read_csv(inDir+'/input/images_train.csv')

trainfoldSource = train[['image_path','image_category']]

folds = 10
skf = StratifiedKFold(n_splits=folds, random_state = random_state, shuffle = True)

skf.get_n_splits(trainfoldSource, trainfoldSource['image_category'])

print(skf) 

Prav_CVindices = pd.DataFrame(columns=['index','CVindices'])

count = 1
for train_index, test_index in skf.split(trainfoldSource, trainfoldSource['image_category']):           
       df_index = pd.DataFrame(test_index,columns=['index']) 
       df_index['CVindices'] = count
       Prav_CVindices = Prav_CVindices.append(df_index)       
       count+=1
       
Prav_CVindices.set_index('index', inplace=True)

trainfoldSource = pd.merge(trainfoldSource, Prav_CVindices, left_index=True, right_index=True)

trainfoldSource.groupby(['CVindices','image_category']).count()

del trainfoldSource['image_category']

trainingSet = pd.merge(train, trainfoldSource, how='left',on="image_path")

trainingSet.to_csv(inDir+"/input/Prav_10folds_CVindices.csv", index=False)

    
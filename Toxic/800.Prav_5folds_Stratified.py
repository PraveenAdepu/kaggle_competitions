# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:31:46 2017

@author: PA23309
"""

import pandas as pd
import numpy as np


random_state = 201802
np.random.RandomState(random_state)

from sklearn.model_selection import StratifiedKFold


inDir = 'C:/Users/SriPrav/Documents/R/42Toxic'

train = pd.read_csv(inDir+'/input/train.csv')
test  = pd.read_csv(inDir+'/input/test.csv')

train.shape #(95851, 8)
test.shape  #(226998, 2)

train.head()
target_columns = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
train["TotalToxic"] = train[target_columns].sum(axis=1)
train.head()

train["TotalToxic"].describe()

train[train["TotalToxic"]==6]

import seaborn as sns
sns.set_style("whitegrid")
tips = sns.load_dataset("tips")
ax = sns.countplot(x="TotalToxic", data=train)

train.groupby(['TotalToxic']).size()

trainfoldSource = train[['id','TotalToxic']]

folds = 5
skf = StratifiedKFold(n_splits=folds, random_state = random_state, shuffle = True)

skf.get_n_splits(trainfoldSource, trainfoldSource['TotalToxic'])

print(skf) 

Prav_CVindices = pd.DataFrame(columns=['index','CVindices'])

count = 1
for train_index, test_index in skf.split(trainfoldSource, trainfoldSource['TotalToxic']):           
       df_index = pd.DataFrame(test_index,columns=['index']) 
       df_index['CVindices'] = count
       Prav_CVindices = Prav_CVindices.append(df_index)       
       count+=1
       
Prav_CVindices.set_index('index', inplace=True)

trainfoldSource = pd.merge(trainfoldSource, Prav_CVindices, left_index=True, right_index=True)

trainfoldSource.groupby(['CVindices','TotalToxic'])[['TotalToxic']].size()

del trainfoldSource['TotalToxic']

trainfoldSource[['id','CVindices']].to_csv(inDir+"/input/Prav_5folds_CVindices.csv", index=False)

    
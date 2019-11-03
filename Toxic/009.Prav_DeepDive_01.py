# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 19:05:18 2017

@author: SriPrav
"""

import pandas as pd
import numpy as np

random_state = 201802
np.random.seed(201802)
np.random.RandomState(random_state)


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


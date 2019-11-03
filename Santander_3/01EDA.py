# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 20:15:45 2019

@author   : Prav
Objective : Kaggle Competition - Santander
          : Develop EDA and Modeling Frameworks
          : Try for Top 50 Rank
"""

inDir = r'C:\Users\SriPrav\Documents\R\58Santander'

import pandas as pd
import numpy as np

random_state = 201902
from Prav_Framework_GenericFunctions import *

train = pd.read_csv(inDir+'/input/train.csv')
test  = pd.read_csv(inDir+'/input/test.csv')

train.groupby(['target']).size()    

generate_CVindices = False

# Prav - caution on over-written CVindices files 

if generate_CVindices: 
    Prav_CVindices = StratifiedFolds(df=train[['ID_code','target']].rename(columns={'ID_code': 'id', 'target': 'target'}),folds=5, random_state=random_state)
    Prav_CVindices.groupby(['CVindices','target'])[['target']].size()
    del Prav_CVindices['target']
    Prav_CVindices.rename(columns={'id':'ID_code'}, inplace=True)
    Prav_CVindices[['ID_code','CVindices']].to_csv(inDir+"/input/Prav_5folds_CVindices.csv", index=False)

# Prav - Take care of this for re-produciability


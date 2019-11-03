# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 20:21:57 2019

@author: SriPrav
"""

import pandas as pd
import numpy as np


def StratifiedFolds(df, folds, random_state):
    np.random.RandomState(random_state)
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=folds, random_state = random_state, shuffle = True)
    
    skf.get_n_splits(df, df['target'])
    
    Prav_CVindices = pd.DataFrame(columns=['index','CVindices'])

    count = 1
    for train_index, test_index in skf.split(df, df['target']):           
           df_index = pd.DataFrame(test_index,columns=['index']) 
           df_index['CVindices'] = count
           Prav_CVindices = Prav_CVindices.append(df_index)       
           count+=1
           
    Prav_CVindices.set_index('index', inplace=True)
    
    df = pd.merge(df, Prav_CVindices, left_index=True, right_index=True)  
    
    return df

'''
# Prav - usage of this function

generate_CVindices = True

if generate_CVindices: 
    Prav_CVindices = StratifiedFolds(df=train[['ID_code','target']].rename(columns={'ID_code': 'id', 'target': 'target'}),folds=5, random_state=random_state)
    Prav_CVindices.groupby(['CVindices','target'])[['target']].size()
    del Prav_CVindices['target']
    Prav_CVindices.rename(columns={'id':'ID_code'}, inplace=True)
    Prav_CVindices[['ID_code','CVindices']].to_csv(inDir+"/input/Prav_5folds_CVindices.csv", index=False)

# Prav - Take care of this for re-produciability
'''
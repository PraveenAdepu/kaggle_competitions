# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 05:56:17 2017

@author: SriPrav
"""

import numpy as np
random_state = 2017
np.random.seed(random_state)

import pandas as pd


inDir = 'C:/Users/SriPrav/Documents/R/43IEEE'

first_model = "resnet_05"

train_models = pd.DataFrame()
current_model = pd.DataFrame()

for i in range(5):    
    fold = i + 1
    current_file = inDir + "\submissions\\Prav."+first_model+".fold1-test.csv"    
    df = pd.read_csv(current_file)    
    current_model= current_model.append(df)

current_model.head()
# Group the data frame by month and item and extract a number of stats from each group
model_folds_mean = current_model.groupby(['fname']).agg({'HTC-1-M7': np.mean,      # find the min, max, and sum of the duration column
                                     'iPhone-4s': np.mean,  # find the number of network type entries
                                     'iPhone-6': np.mean,
                                     'LG-Nexus-5x': np.mean,
                                     'Motorola-Droid-Maxx': np.mean,
                                     'Motorola-Nexus-6': np.mean,
                                     'Motorola-X': np.mean,
                                     'Samsung-Galaxy-Note3': np.mean,
                                     'Samsung-Galaxy-S4': np.mean,
                                     'Sony-NEX-7': np.mean})    # get the min, first, and number of
    
    
    
model_folds_mean['camera'] = model_folds_mean[['HTC-1-M7', 'iPhone-4s', 'iPhone-6', 'LG-Nexus-5x',
                                   'Motorola-Droid-Maxx', 'Motorola-Nexus-6', 'Motorola-X',
                                   'Samsung-Galaxy-Note3', 'Samsung-Galaxy-S4', 'Sony-NEX-7']].idxmax(axis=1)
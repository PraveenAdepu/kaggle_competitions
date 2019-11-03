# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:32:04 2017

@author: PA23309
"""
import pandas as pd
import glob

inDir =r'C:\Users\PA23309\Documents\Prav-Development\01.Model References\05.Research\Driver' # use your path

first_model = "et001"
level01_models = ["et002","et003","lgbm001"]

train_models = pd.DataFrame()
current_model = pd.DataFrame()

for i in range(5):    
    fold = i + 1
    current_file = inDir + "\submissions\\Prav."+first_model+".fold"+str(fold)+".csv"    
    df = pd.read_csv(current_file)    
    current_model= current_model.append(df)

current_model.rename(columns={'target': first_model}, inplace=True)
train_models = pd.concat([train_models,current_model], axis=1)
 
for model in level01_models:
    current_model = pd.DataFrame()
    for i in range(5):    
        fold = i + 1
        current_fil = inDir + "\submissions\\Prav."+model+".fold"+str(fold)+".csv"    
        df = pd.read_csv(current_fil)    
        current_model= current_model.append(df)
    current_model.rename(columns={'target': model}, inplace=True)
    train_models = pd.merge(train_models, current_model, how="left", on="id")
    
test_models = pd.DataFrame()
current_file = inDir + "\submissions\\Prav."+first_model+".full.csv"
test_models = pd.read_csv(current_file)
test_models.rename(columns={'target': first_model}, inplace=True)

for model in level01_models:
    current_model = pd.DataFrame()    
    current_fil = inDir + "\submissions\\Prav."+model+".full.csv"    
    current_model = pd.read_csv(current_fil)    
    current_model.rename(columns={'target': model}, inplace=True)
    test_models = pd.merge(test_models, current_model, how="left", on="id")

train_models.corr()    
test_models.corr()





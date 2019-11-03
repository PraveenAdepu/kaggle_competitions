# -*- coding: utf-8 -*-
"""
Created on Tue May 15 22:48:17 2018

@author: SriPrav
"""

import numpy as np 
import pandas as pd 

import scipy.stats

inDir = 'C:/Users/SriPrav/Documents/R/50Santander'
####################################################################################################################### 

Model01 = pd.read_csv(inDir+'/submissions/Prav.xgb003.full.csv') # LB : 1.40
Model02 = pd.read_csv(inDir+'/submissions/Prav.xgb004.full.csv') # LB : 1.41

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'ID')

Ensemble[["target_x","target_y"]].corr()

a = scipy.stats.spearmanr(Ensemble[["target_x","target_y"]])
a[0]

Ensemble["target"] = Ensemble["target_x"] * 0.5 + Ensemble["target_y"] * 0.5

ModelName = 'Ensemble01'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'

Ensemble[["ID","target"]].to_csv(sub_file, index=False)

#######################################################################################################################   


Model01 = pd.read_csv(inDir+'/submissions/Prav.xgb005.full.csv') # LB : 1.41
Model02 = pd.read_csv(inDir+'/submissions/Prav.xgb004.full.csv') # LB : 1.41

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'ID')

Ensemble[["target_x","target_y"]].corr()

a = scipy.stats.spearmanr(Ensemble[["target_x","target_y"]])
a[0]

Ensemble["target"] = Ensemble["target_x"] * 0.5 + Ensemble["target_y"] * 0.5

ModelName = 'Ensemble_xgb004005'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'

Ensemble[["ID","target"]].to_csv(sub_file, index=False)
####################################################################################################################### 

Model01 = pd.read_csv(inDir+'/submissions/Prav.xgb003.full.csv') # LB : 1.40
Model02 = pd.read_csv(inDir+'/submissions/Prav.Ensemble_xgb004005.csv') # LB : 1.41

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'ID')

Ensemble[["target_x","target_y"]].corr()

a = scipy.stats.spearmanr(Ensemble[["target_x","target_y"]])
a[0]

Ensemble["target"] = Ensemble["target_x"] * 0.5 + Ensemble["target_y"] * 0.5

ModelName = 'Ensemble02'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'

Ensemble[["ID","target"]].to_csv(sub_file, index=False)

####################################################################################################################### 

####################################################################################################################### 

Model01 = pd.read_csv(inDir+'/submissions/Prav.xgb003.full.csv') # LB : 1.40
Model02 = pd.read_csv(inDir+'/submissions/pipeline_kernel_cv1.0.csv') # LB : 1.39

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'ID')

Ensemble[["target_x","target_y"]].corr()

a = scipy.stats.spearmanr(Ensemble[["target_x","target_y"]])
a[0]

Ensemble["target"] = Ensemble["target_x"] * 0.5 + Ensemble["target_y"] * 0.5

ModelName = 'Ensemble_xgb003_ref'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'

Ensemble[["ID","target"]].to_csv(sub_file, index=False)

#######################################################################################################################  

####################################################################################################################### 

Model01 = pd.read_csv(inDir+'/submissions/Prav.lgbm001.full.csv') # LB : 1.40
Model02 = pd.read_csv(inDir+'/submissions/pipeline_kernel_cv1.0.csv') # LB : 1.39

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'ID')

Ensemble[["target_x","target_y"]].corr()

a = scipy.stats.spearmanr(Ensemble[["target_x","target_y"]])
a[0]

Ensemble["target"] = Ensemble["target_x"] * 0.5 + Ensemble["target_y"] * 0.5

ModelName = 'Ensemble_lgbm001_ref'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'

Ensemble[["ID","target"]].to_csv(sub_file, index=False)

#######################################################################################################################   

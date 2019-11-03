# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 09:13:38 2017

@author: PAdepu
"""
import pandas as pd
import numpy as np
import glob
import scipy
import scipy.stats as ss


inDir = "C:\Users\SriPrav\Documents\R\\24DSM"

models = ['xgb101']
for model in models:    
    train_path = inDir + "\\submissions\\prav." + model + ".fold*[0-9].csv"
    train_from_each_file = (pd.read_csv(f) for f in glob.glob(train_path))
    train_from_each_file_df   = pd.concat(train_from_each_file)
    train_from_each_file_df.rename(columns={'Diabetes': model}, inplace = True)  
    train = train_from_each_file_df
    
    test_path = inDir + "\\submissions\\prav." + model + ".full.csv"
    test_from_each_file = (pd.read_csv(f) for f in glob.glob(test_path))
    test_from_each_file_df   = pd.concat(test_from_each_file)
    test_from_each_file_df.rename(columns={'Diabetes': model}, inplace = True)  
    test = test_from_each_file_df
#L2_xgb08 = ['xgb12','rf01', 'xgb20','lgbm01','xgb21','lgbm02','xgb30','rf20','et20','nn21','xgb22','et22','nn20','xgb31','xgb32'] 
models = ['lgbm01','xgb21','lgbm02','xgb30','rf20','et20','nn21','xgb22','et22','nn20','xgb31','xgb32','xgb33','rgf30','rf30','et30', 'xgb20']#,,'xgb12','rf01','rf31', not sure of rf31 # rf22 never use
for model in models:    
    train_path = inDir + "\\submissions\\prav." + model + ".fold*[0-9].csv"
    train_from_each_file = (pd.read_csv(f) for f in glob.glob(train_path))
    train_from_each_file_df   = pd.concat(train_from_each_file)
    train_from_each_file_df.rename(columns={'Diabetes': model}, inplace = True)   
    train = pd.merge(train, train_from_each_file_df, on = "Patient_ID", how = "left")
    
    
    test_path = inDir + "\\submissions\\prav." + model + ".full.csv"
    test_from_each_file_df = pd.read_csv(test_path)    
    test_from_each_file_df.rename(columns={'Diabetes': model}, inplace = True)  
    test = pd.merge(test, test_from_each_file_df, on = 'Patient_ID',how = 'left')
    
cv_file = inDir + "/CVSchema/Prav_CVindices_5folds_10.csv"
CV_Schema = pd.read_csv(cv_file)
CV_Schema.head(2)


trainingSet = pd.merge(train, CV_Schema, how = 'left', on = 'Patient_ID')
testingSet = test
feature_names = [col for col in trainingSet.columns if col not in ['Patient_ID', 'CVindices','DiabetesDispense']]

print trainingSet.head()

print 'ranking...'
for col in feature_names:
    print col
    trainingSet[col] = ss.rankdata( trainingSet[col] )
print trainingSet.head()

print 'ranking...'
for col in feature_names:
    print col
    testingSet[col] = ss.rankdata( testingSet[col] )
print testingSet.head()

trainingSet[feature_names] = trainingSet[feature_names].apply(lambda x:x / np.max(x))
testingSet[feature_names]  = testingSet[feature_names].apply(lambda x:x / np.max(x))

ss.spearmanr(trainingSet[feature_names])
#############################################################################################################################
folds = 5
nbags = 5
nn_epoch = 10
full_nn_epoch = 12
current_seed = 2017

param = {}
param['seed'] = 2017
param['objective'] = 'reg:linear'
param['eta'] = 0.01
param['max_depth'] = 4
param['silent'] = 1
param['eval_metric'] = "auc"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 3
param['print_every_n'] = 50
xgb_num_rounds = 210

xgbParameters = list(param.items())

 
#xgbParameters[7] = ('seed', current_seed + 1)
xgbModelName  = 'L2_xgb09'
#lgbmModelName = 'lgbm02'
#rfModelName   = 'rf21'
#adaModelName  = 'ada20'
#gbdtModelName = 'gbdt20'
#etModelName   = 'et21'
#lrModelName   = 'lr20'
nnModelName   = 'L2_nn08'
#knnModelName  = 'knn20'

if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, xgb_num_rounds)
    fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,xgbModelName,xgbParameters, xgb_num_rounds)
    
    for i in range(1, folds+1):
        train_nn_regression(trainingSet, testingSet,feature_names,i,nbags,nnModelName,nn_epoch)
    fulltrain_nn_regression(trainingSet, testingSet,feature_names,nbags,nnModelName,full_nn_epoch)

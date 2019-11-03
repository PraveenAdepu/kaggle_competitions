# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:32:04 2017

@author: PA23309
"""
import pandas as pd
import glob

inDir = 'C:\\Users\\SriPrav\\Documents\\R\\36Driver'

first_model = "xgb002"
level01_models = ["xgb002_02","xgb002_03","xgb002_04","xgb003","xgb003_01","lgbm002","lgbm002_02","lgbm002_03","lgbm002_04","lgbm003","nn01","rf002","et002","rf003","et003","libffm"]#,"rgf001" 

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


trainfoldSource = pd.read_csv(inDir+'/input/Prav_5folds_CVindices.csv')
train = pd.read_csv(inDir+'/input/train.csv')
train = train[['id','target']]
train = pd.merge(train, trainfoldSource, how='left',on="id")

trainingSet = pd.merge(train_models, train, how='left',on='id')

testingSet = test_models

feature_names = [c for c in trainingSet.columns if c not in ['id','target','CVindices']]

seed = 2017
folds = 5
nbags = 5
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2017
verboseeval=250

rfStackingModelName = 'rfStacking01'
for i in range(1, folds+1):
    train_regression("rf",trainingSet, testingSet,feature_names,i,nbags,rfStackingModelName,current_seed)
fulltrain_regression("rf",trainingSet, testingSet,feature_names,nbags,rfStackingModelName,current_seed)


param = {}
param['seed'] = 2017
param['objective'] = 'binary:logistic'#'reg:linear'
param['eta'] = 0.001
param['max_depth'] = 4
param['silent'] = 1
param['min_child_weight'] = 3
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 16
param['print_every_n'] = 25
param['booster'] = 'gbtree'
#param['base_score'] = y_mean
#param['eval_metric'] = "auc"
xgb_num_rounds = 340

xgbParameters = list(param.items())


seed = 2017
folds = 5
nbags = 5
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2017
verboseeval=20

xgbModelName  = 'xgbStacking02'

for i in range(1, folds+1):
    train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)

    
lgbm_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        #'metric': 'auc',
        'max_depth': 3,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 10,        
        'nthread': 30,         
        'seed': 2017,
    } 

lgbm_num_round = 200
lgbm_early_stopping_rounds = 50

seed = 2017
folds = 5
nbags = 5
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2017
verboseeval=20

lgbmModelName = 'lgbmStacking02'
lrModelName = 'lrStacking01'

for i in range(1, folds+1):
        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)
fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)

for i in range(1, folds+1):
    train_regression("lr",trainingSet, testingSet,feature_names,i,nbags,lrModelName,current_seed)
fulltrain_regression("lr",trainingSet, testingSet,feature_names,nbags,lrModelName,current_seed)
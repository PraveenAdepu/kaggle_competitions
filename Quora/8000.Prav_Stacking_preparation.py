# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 09:13:38 2017

@author: PAdepu
"""
import pandas as pd
import numpy as np
import glob
inDir = "C:\Users\SriPrav\Documents\R\\23Quora"

models = ['xgb50']
for model in models:    
    train_path = inDir + "\\submissions\\prav." + model + ".fold*[0-9].csv"
    train_from_each_file = (pd.read_csv(f) for f in glob.glob(train_path))
    train_from_each_file_df   = pd.concat(train_from_each_file)
#    train_from_each_file_df = train_from_each_file_df.groupby(['id'])['is_duplicate'].mean().reset_index(name='is_duplicate')
    train_from_each_file_df.rename(columns={'is_duplicate': model}, inplace = True)
    train = train_from_each_file_df
    
    test_path = inDir + "\\submissions\\prav." + model + ".full.csv"
    test_from_each_file = (pd.read_csv(f) for f in glob.glob(test_path))
    test_from_each_file_df   = pd.concat(test_from_each_file)
    test_from_each_file_df.rename(columns={'is_duplicate': model}, inplace = True)  
    test = test_from_each_file_df
 
models = ['nn02','lgbm01','nn50','xgb52','lgbm02','xgb53','et51','rf50','nn51','xgb57','xgb58','xgb103','lgbm100','nn100','rf100','et100','lgbm103','nn101','lgbm105','xgb106','nn105']#,'et50','nn102'
for model in models:    
    train_path = inDir + "\\submissions\\prav." + model + ".fold*[0-9].csv"
    train_from_each_file = (pd.read_csv(f) for f in glob.glob(train_path))
    train_from_each_file_df   = pd.concat(train_from_each_file)
#    train_from_each_file_df = train_from_each_file_df.groupby(['id'])['is_duplicate'].mean().reset_index(name='is_duplicate')
    train_from_each_file_df.rename(columns={'is_duplicate': model}, inplace = True)   
#    train = pd.merge(train, train_from_each_file_df, on = "id", how = "left")
    del train_from_each_file_df['id']
    train = pd.concat([train,train_from_each_file_df],  axis=1 )
    
    
    test_path = inDir + "\\submissions\\prav." + model + ".full.csv"
    test_from_each_file_df = pd.read_csv(test_path)    
    test_from_each_file_df.rename(columns={'is_duplicate': model}, inplace = True)  
    test = pd.merge(test, test_from_each_file_df, on = 'test_id',how = 'left')
    
cv_file = inDir + "/CVSchema/Prav_CVindices_5folds.csv"
CV_Schema = pd.read_csv(cv_file)
CV_Schema.head(2)
del CV_Schema['qid1']
del CV_Schema['qid2']



testingSet = test

train_file = inDir + "/input/train.csv"
train_file = pd.read_csv(train_file)

target_columns = ['id','is_duplicate']
train_target = train_file[target_columns]
del train_file
trainingSet = pd.merge(train, train_target, how = 'left', on = ['id'])
trainingSet = pd.merge(trainingSet, CV_Schema, how = 'left', on = ['id'])

feature_names = [col for col in trainingSet.columns if col not in ['id', 'CVindices','is_duplicate']]

#a = scipy.stats.spearmanr(trainingSet[feature_names])
#a[0]
#########################################################################################################################


param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.02
param['max_depth'] = 4
param['silent'] = 1
param['eval_metric'] = "logloss"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 25
param['seed'] = 2017
param['print_every_n'] = 50
xgb_num_rounds = 460

xgbParameters = list(param.items())



lgbm_params = {
          'boosting_type': 'gbdt',
          'objective': 'regression',
          'metric': 'binary_logloss',
          'min_child_weight': 1.5,
          'num_leaves': 2**4,
          'lambda_l2': 10,
          'feature_fraction': 0.7,
          'bagging_fraction': 0.7,
          'bagging_freq': 5,
          'learning_rate': 0.02,
          'tree_method': 'exact',          
          'nthread': 25,
          'silent': True,
          'seed': 2017,
         }
lgbm_num_round = 8010
lgbm_early_stopping_rounds = 100

seed = 2017
folds = 5
nbags = 1
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2017

lgbmModelName = 'L2_lgbm01'
xgbModelName  = 'L2_xgb12'

if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, xgb_num_rounds)
    fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,xgbModelName,xgbParameters, xgb_num_rounds)
    
    for i in range(1, folds+1):
        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbm_params, lgbm_num_round)
    fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round)
    
    
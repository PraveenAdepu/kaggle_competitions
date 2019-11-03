# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 09:13:38 2017

@author: PAdepu
"""
import pandas as pd
import numpy as np
import glob
from sklearn.metrics import r2_score

inDir = "C:\Users\SriPrav\Documents\R\\26Mercedes"

models = ['xgb002']
for model in models:    
    train_path = inDir + "\\submissions\\prav." + model + ".fold*[0-9].csv"
    train_from_each_file = (pd.read_csv(f) for f in glob.glob(train_path))
    train_from_each_file_df   = pd.concat(train_from_each_file)
#    train_from_each_file_df = train_from_each_file_df.groupby(['id'])['is_duplicate'].mean().reset_index(name='is_duplicate')
    train_from_each_file_df.rename(columns={'y': model}, inplace = True)
    train = train_from_each_file_df
    
    test_path = inDir + "\\submissions\\prav." + model + ".full.csv"
    test_from_each_file = (pd.read_csv(f) for f in glob.glob(test_path))
    test_from_each_file_df   = pd.concat(test_from_each_file)
    test_from_each_file_df.rename(columns={'y': model}, inplace = True)  
    test = test_from_each_file_df
 
models = ['lgbm003','rf101','et101','gbdt001','xgb003','xgb004','gbdt002']#,'et50','nn102'
for model in models:    
    train_path = inDir + "\\submissions\\prav." + model + ".fold*[0-9].csv"
    train_from_each_file = (pd.read_csv(f) for f in glob.glob(train_path))
    train_from_each_file_df   = pd.concat(train_from_each_file)
#    train_from_each_file_df = train_from_each_file_df.groupby(['id'])['is_duplicate'].mean().reset_index(name='is_duplicate')
    train_from_each_file_df.rename(columns={'y': model}, inplace = True)   
    train = pd.merge(train, train_from_each_file_df, on = "ID", how = "left")
#    del train_from_each_file_df['id']
#    train = pd.concat([train,train_from_each_file_df],  axis=1 )
    
    
    test_path = inDir + "\\submissions\\prav." + model + ".full.csv"
    test_from_each_file_df = pd.read_csv(test_path)    
    test_from_each_file_df.rename(columns={'y': model}, inplace = True)  
    test = pd.merge(test, test_from_each_file_df, on = 'ID',how = 'left')
    
cv_file = inDir + "/CVSchema/Prav_CVindices_10folds.csv"
CV_Schema = pd.read_csv(cv_file)


testingSet = test

train_file = inDir + "/input/train.csv"
train_file = pd.read_csv(train_file)

target_columns = ['ID','y']
train_target = train_file[target_columns]
del train_file
trainingSet = pd.merge(train, train_target, how = 'left', on = ['ID'])
trainingSet = pd.merge(trainingSet, CV_Schema, how = 'left', on = ['ID'])

feature_names = cols = [col for col in trainingSet.columns if col not in ['ID','y','CVindices']] #

y_mean = np.mean(trainingSet["y"])

#a = scipy.stats.spearmanr(trainingSet[feature_names])
#a[0]
#########################################################################################################################


def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

param = {}
param['seed'] = 2017
param['objective'] = 'reg:linear'
param['eta'] = 0.01
param['max_depth'] = 4
param['silent'] = 1
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 10
param['print_every_n'] = 100
param['base_score'] = y_mean
#param['eval_metric'] = "logloss"
xgb_num_rounds = 510

xgbParameters = list(param.items())



lgbm_params = {
          'task': 'train',
          'boosting_type': 'gbdt',
          'objective': 'regression',
          'metric': 'l2',
          'min_child_weight': 16,
          'num_leaves': 2**4, #2**4,
          #'lambda_l2': 2,
          'feature_fraction': 0.7,
          'bagging_fraction': 0.9,#0.7
          'bagging_freq': 4,#2
          'learning_rate': 0.01,
          'tree_method': 'exact',
          'min_data_in_leaf':4,
          'min_sum_hessian_in_leaf': 0.8,          
          'nthread': 10,
          'silent': True,
          'seed': 2017,
         }
lgbm_num_round = 610
lgbm_early_stopping_rounds = 100



seed = 2017
folds = 10
nbags = 5
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2017
verboseeval=300

lgbmModelName = 'L2_lgbm01'
xgbModelName  = 'L2_xgb03'

if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
    fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
    
    for i in range(1, folds+1):
        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbm_params, lgbm_num_round)
    fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round)
    
    
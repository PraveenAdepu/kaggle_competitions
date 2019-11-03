# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:31:46 2017

@author: PA23309
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgbm

random_state = 2017
np.random.RandomState(random_state)
from sklearn.model_selection import StratifiedKFold

inDir = 'C:\\Users\\SriPrav\\Documents\\R\\36Driver'
train = pd.read_csv(inDir+'/input/train.csv')
test = pd.read_csv(inDir+'/input/test.csv')
trainfoldSource = train[['id','target']]

folds = 5
skf = StratifiedKFold(n_splits=folds, random_state = random_state, shuffle = True)

skf.get_n_splits(trainfoldSource, trainfoldSource['target'])

print(skf) 

Prav_CVindices = pd.DataFrame(columns=['index','CVindices'])

count = 1
for train_index, test_index in skf.split(trainfoldSource, trainfoldSource['target']):           
       df_index = pd.DataFrame(test_index,columns=['index']) 
       df_index['CVindices'] = count
       Prav_CVindices = Prav_CVindices.append(df_index)       
       count+=1
       
Prav_CVindices.set_index('index', inplace=True)

trainfoldSource = pd.merge(trainfoldSource, Prav_CVindices, left_index=True, right_index=True)

trainfoldSource.groupby(['CVindices'])[['target']].sum()

del trainfoldSource['target']

trainingSet = pd.merge(train, trainfoldSource, how='left',on="id")
testingSet = test


trainingSet[['id','CVindices']].to_csv(inDir+"/input/Prav_5folds_CVindices.csv", index=False)

feature_names = cols = [col for col in trainingSet.columns if col not in ['id','target','CVindices']] #'ID',

#y_mean = np.mean(trainingSet["target"])

param = {}
param['seed'] = 2017
param['objective'] = 'reg:linear'
param['eta'] = 0.05
param['max_depth'] = 4
param['silent'] = 1
param['min_child_weight'] = 3
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 20
param['print_every_n'] = 20
param['booster'] = 'gbtree'
#param['base_score'] = y_mean
#param['eval_metric'] = "logloss"
xgb_num_rounds = 160

xgbParameters = list(param.items())

lgbm_params = {
          'task'              : 'train',
          'boosting_type'     : 'gbdt',
          'objective'         : 'regression',
          'metric'            : 'l2',
          'min_child_weight'  : 16,
          'num_leaves'        : 2**4, #2**4,
          #'lambda_l2'        : 2,
          'feature_fraction'  : 0.7,
          'bagging_fraction'  : 0.7,#0.7
          'bagging_freq'      : 4,#2
          'learning_rate'     : 0.05,
          'tree_method'       : 'exact',
          'min_data_in_leaf'  : 4,
          'min_sum_hessian_in_leaf': 0.8,          
          'nthread'           : 20,
          'silent'            : False,
          'seed'              : 2017,
         }
lgbm_num_round = 160
lgbm_early_stopping_rounds = 40

seed = 2017
folds = 5
nbags = 5
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2017
verboseeval=40

lgbmModelName = 'lgbm001'
xgbModelName  = 'xgb001'
rfModelName   = 'rf001'
etModelName   = 'et001'
fmModelName   = 'fm100'
adaModelName  = 'ada001'
gbdtModelName = 'gbdt001'


if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
    fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
    
    for i in range(1, folds+1):
        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)
    fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)

    for i in range(1, folds+1):
        train_regression("rf",trainingSet, testingSet,feature_names,i,nbags,rfModelName,current_seed)
    fulltrain_regression("rf",trainingSet, testingSet,feature_names,nbags,rfModelName,current_seed)
    
    for i in range(1, folds+1):
        train_regression("et",trainingSet, testingSet,feature_names,i,nbags,etModelName,current_seed)
    fulltrain_regression("et",trainingSet, testingSet,feature_names,nbags,etModelName,current_seed)
    
#####################################################################################################################################
#####################################################################################################################################
# Exclude top 10 xgb features and start all models for second run
feature_names = cols = [col for col in trainingSet.columns if col not in ['id','target','CVindices']] #'ID',
FirstXGBTop10features = [   'ps_car_13',
                            'ps_ind_03',
                            'ps_reg_03',
                            'ps_ind_05_cat',
                            'ps_ind_15',
                            'ps_ind_01',
                            'ps_car_01_cat',
                            'ps_reg_02',
                            'ps_car_14',
                            'ps_ind_17_bin'
                            ]
len(feature_names)
feature_names = list(set(feature_names) - set(FirstXGBTop10features))
len(feature_names)
#y_mean = np.mean(trainingSet["y"])

param = {}
param['seed'] = 2018
param['objective'] = 'reg:linear'
param['eta'] = 0.05
param['max_depth'] = 4
param['silent'] = 1
param['min_child_weight'] = 3
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 10
param['print_every_n'] = 50
#param['base_score'] = y_mean
#param['eval_metric'] = "logloss"
xgb_num_rounds = 140

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
          'bagging_fraction': 0.7,#0.7
          'bagging_freq': 4,#2
          'learning_rate': 0.05,
          'tree_method': 'exact',
          'min_data_in_leaf':4,
          'min_sum_hessian_in_leaf': 0.8,          
          'nthread': 3,
          'silent': False,
          'seed': 2018,
         }
lgbm_num_round = 140
lgbm_early_stopping_rounds = 40

seed = 2017
folds = 5
nbags = 1
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2018
verboseeval=40

lgbmModelName = 'lgbm002'
xgbModelName  = 'xgb002'
rfModelName   = 'rf002'
etModelName   = 'et002'
fmModelName   = 'fm100'
adaModelName  = 'ada001'
gbdtModelName = 'gbdt001'
gc.collect()

if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
    fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
    
    for i in range(1, folds+1):
        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)
    fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)

    for i in range(1, folds+1):
        train_regression("rf",trainingSet, testingSet,feature_names,i,nbags,rfModelName,current_seed)
    fulltrain_regression("rf",trainingSet, testingSet,feature_names,nbags,rfModelName,current_seed)
    
    for i in range(1, folds+1):
        train_regression("et",trainingSet, testingSet,feature_names,i,nbags,etModelName,current_seed)
    fulltrain_regression("et",trainingSet, testingSet,feature_names,nbags,etModelName,current_seed)

#####################################################################################################################################
#####################################################################################################################################
# Exclude top 10 xgb features and start all models for second run
feature_names = cols = [col for col in trainingSet.columns if col not in ['id','target','CVindices']] #'ID',
FirstXGBBottom10features = ['ps_calc_04',
                            'ps_ind_13_bin',
                            'ps_car_05_cat',
                            'ps_calc_19_bin',
                            'ps_calc_17_bin',
                            'ps_calc_16_bin',
                            'ps_calc_18_bin',
                            'ps_ind_10_bin',
                            'ps_ind_11_bin',
                            'ps_calc_20_bin'
                            ]
len(feature_names)
feature_names = list(set(feature_names) - set(FirstXGBBottom10features))
len(feature_names)
#y_mean = np.mean(trainingSet["y"])

param = {}
param['seed'] = 2019
param['objective'] = 'reg:linear'
param['eta'] = 0.05
param['max_depth'] = 4
param['silent'] = 1
param['min_child_weight'] = 3
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 10
param['print_every_n'] = 50
#param['base_score'] = y_mean
#param['eval_metric'] = "logloss"
xgb_num_rounds = 140

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
          'bagging_fraction': 0.7,#0.7
          'bagging_freq': 4,#2
          'learning_rate': 0.05,
          'tree_method': 'exact',
          'min_data_in_leaf':4,
          'min_sum_hessian_in_leaf': 0.8,          
          'nthread': 3,
          'silent': False,
          'seed': 2018,
         }
lgbm_num_round = 140
lgbm_early_stopping_rounds = 40

seed = 2019
folds = 5
nbags = 1
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2019
verboseeval=40

lgbmModelName = 'lgbm003'
xgbModelName  = 'xgb003'
rfModelName   = 'rf003'
etModelName   = 'et003'
fmModelName   = 'fm100'
adaModelName  = 'ada001'
gbdtModelName = 'gbdt001'
gc.collect()

if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
    fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
    
    for i in range(1, folds+1):
        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)
    fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)

    for i in range(1, folds+1):
        train_regression("rf",trainingSet, testingSet,feature_names,i,nbags,rfModelName,current_seed)
    fulltrain_regression("rf",trainingSet, testingSet,feature_names,nbags,rfModelName,current_seed)
    
    for i in range(1, folds+1):
        train_regression("et",trainingSet, testingSet,feature_names,i,nbags,etModelName,current_seed)
    fulltrain_regression("et",trainingSet, testingSet,feature_names,nbags,etModelName,current_seed)
    
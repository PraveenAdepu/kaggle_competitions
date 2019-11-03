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


inDir = 'C:\\Users\\SriPrav\\Documents\\R\\36Driver'
train = pd.read_csv(inDir+'/input/train.csv')
test = pd.read_csv(inDir+'/input/test.csv')

trainfoldSource = pd.read_csv(inDir+'/input/Prav_5folds_CVindices.csv')
train = pd.merge(train, trainfoldSource, how='left',on="id")

train.groupby(['CVindices'])[['target']].sum()

# Preprocessing (Forza Baseline)
id_test = test['id'].values

col = [c for c in train.columns if c not in ['id','target']]
col = [c for c in col if not c.startswith('ps_calc_')]


def recon(reg):
    integer = int(np.round((40*reg)**2)) 
    for a in range(32):
        if (integer - a) % 31 == 0:
            A = a
    M = (integer - A)//31
    return A, M
train['ps_reg_A'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])
train['ps_reg_M'] = train['ps_reg_03'].apply(lambda x: recon(x)[1])
train['ps_reg_A'].replace(19,-1, inplace=True)
train['ps_reg_M'].replace(51,-1, inplace=True)
test['ps_reg_A'] = test['ps_reg_03'].apply(lambda x: recon(x)[0])
test['ps_reg_M'] = test['ps_reg_03'].apply(lambda x: recon(x)[1])
test['ps_reg_A'].replace(19,-1, inplace=True)
test['ps_reg_M'].replace(51,-1, inplace=True)

train = train.replace(-1, np.NaN)
d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
train = train.fillna(-1)
one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target','CVindices']}


def transform(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target','CVindices']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol:
        if '_bin' not in c:
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)

    for c in one_hot:
        if len(one_hot[c])>2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df



#cat("Feature engineering")
#data[, amount_nas := rowSums(data == -1, na.rm = T)]
#data[, high_nas := ifelse(amount_nas>4,1,0)]
#data[, ps_car_13_ps_reg_03 := ps_car_13*ps_reg_03]
#data[, ps_reg_mult := ps_reg_01*ps_reg_02*ps_reg_03]
#data[, ps_ind_bin_sum := ps_ind_06_bin+ps_ind_07_bin+ps_ind_08_bin+ps_ind_09_bin+ps_ind_10_bin+ps_ind_11_bin+ps_ind_12_bin+ps_ind_13_bin+ps_ind_16_bin+ps_ind_17_bin+ps_ind_18_bin]

train = transform(train)
test = transform(test)

col = [c for c in train.columns if c not in ['id','target','CVindices']]
col = [c for c in col if not c.startswith('ps_calc_')]

dups = train[train.duplicated(subset=col, keep=False)]

train = train[~(train['id'].isin(dups['id'].values))]

trainingSet = train
testingSet  = test


feature_names = col



param = {}
param['seed'] = 2017
param['objective'] = 'binary:logistic'#'reg:linear'
param['eta'] = 0.05
param['max_depth'] = 4
param['silent'] = 1
param['min_child_weight'] = 3
param['subsample'] = 0.8
param['colsample_bytree'] = 0.8
param['nthread'] = 16
param['print_every_n'] = 250
param['booster'] = 'gbtree'
#param['base_score'] = y_mean
#param['eval_metric'] = "logloss"
xgb_num_rounds = 1000

xgbParameters = list(param.items())

#lgb_params_1 = {
#    'learning_rate': 0.01,
#    'n_estimators': 1250,
#    'max_bin': 10,
#    'subsample': 0.8,
#    'subsample_freq': 10,
#    'colsample_bytree': 0.8, 
#    'min_child_samples': 500
#}

lgbm_params = {
          'task'              : 'train',
          'boosting_type'     : 'gbdt',
          'objective'         : 'regression',
          'num_leaves'        : 2**4, #2**4,
          'feature_fraction'  : 0.8,
          'bagging_fraction'  : 0.8,
          'bagging_freq'      : 10,#2
          'learning_rate'     : 0.01,
          'tree_method'       : 'exact',
          'min_data_in_leaf'  : 500,             
          'nthread'           : 30,
          'silent'            : False,
          'seed'              : 2017,
         }
lgbm_num_round = 1150
lgbm_early_stopping_rounds = 250

seed = 2017
folds = 5
nbags = 5
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2017
verboseeval=250

lgbmModelName = 'lgbm002'
xgbModelName  = 'xgb002'
rfModelName   = 'rf002'
etModelName   = 'et002'
fmModelName   = 'fm100'
adaModelName  = 'ada001'
gbdtModelName = 'gbdt001'
lrModelName = 'lr001'

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

    for i in range(1, folds+1):
        train_regression("lr",trainingSet, testingSet,feature_names,i,nbags,lrModelName,current_seed)
    fulltrain_regression("lr",trainingSet, testingSet,feature_names,nbags,lrModelName,current_seed)
lgbm_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 3,
        'learning_rate': 0.05,
        'feature_fraction': 1,
        'bagging_fraction': 1,
        'bagging_freq': 10,        
        'nthread': 30,         
        'seed': 2017,
    }
lgbm_num_round = 510
lgbm_early_stopping_rounds = 250

seed = 2017
current_seed = 2017
verboseeval=250
    
lgbmModelName = 'lgbm002_02'

for i in range(1, folds+1):
        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)
        fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)

lgbm_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 4,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 2,       
        'nthread': 30,         
        'seed': 2017,
    }
lgbm_num_round = 250
lgbm_early_stopping_rounds = 250

seed = 2017
current_seed = 2017
verboseeval=250
    
lgbmModelName = 'lgbm002_03'

for i in range(1, folds+1):
        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)
        fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)
    
lgbm_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 5,
        'learning_rate': 0.05,
        'feature_fraction': 0.3,
        'bagging_fraction': 0.7,
        'bagging_freq': 10,        
        'nthread': 30,         
        'seed': 2017,
    } 
lgbm_num_round = 250
lgbm_early_stopping_rounds = 250

seed = 2017
current_seed = 2017
verboseeval=250
    
lgbmModelName = 'lgbm002_04'

for i in range(1, folds+1):
        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)
        fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)
    

param = {}
param['seed'] = 2017
param['objective'] = 'binary:logistic'#'reg:linear'
param['eta'] = 0.025
param['max_depth'] = 4
param['silent'] = 1
param['min_child_weight'] = 100
param['subsample'] = 0.9
param['colsample_bytree'] = 0.7
param['colsample_bylevel']= 0.7
param['nthread'] = 16
param['print_every_n'] = 250
param['booster'] = 'gbtree'
param['alpha'] = 4
#param['base_score'] = y_mean
param['eval_metric'] = "auc"
xgb_num_rounds = 1050
xgbParameters = list(param.items())

seed = 2017
folds = 5
nbags = 5
current_seed = 2017
verboseeval=250

xgbModelName  = 'xgb003_01'

if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
    fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
    
param = {}
param['seed'] = 2017
param['objective'] = 'binary:logistic'#'reg:linear'
param['eta'] = 0.01
param['max_depth'] = 4
param['silent'] = 1
param['min_child_weight'] = 8
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 16
param['print_every_n'] = 250
param['booster'] = 'gbtree'
#param['base_score'] = y_mean
param['eval_metric'] = "auc"
xgb_num_rounds = 1250
xgbParameters = list(param.items())

seed = 2017
folds = 5
nbags = 5
current_seed = 2017
verboseeval=250

xgbModelName  = 'xgb002_03'

if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
    fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
    
    param = {}
param['seed'] = 2017
param['objective'] = 'binary:logistic'#'reg:linear'
param['eta'] = 0.01
param['max_depth'] = 5
param['silent'] = 1
param['min_child_weight'] = 10
param['subsample'] = 0.7
param['colsample_bytree'] = 0.3
param['nthread'] = 16
param['print_every_n'] = 250
param['booster'] = 'gbtree'
#param['base_score'] = y_mean
param['eval_metric'] = "auc"
xgb_num_rounds = 1250
xgbParameters = list(param.items())

seed = 2017
folds = 5
nbags = 5
current_seed = 2017
verboseeval=250

xgbModelName  = 'xgb002_04'

if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
    fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
       
       
#####################################################################################################################################
#####################################################################################################################################
# Exclude top 10 xgb features and start all models for second run
feature_names = cols = [col for col in trainingSet.columns if col not in ['id','target','CVindices']] #'ID',
FirstXGBTop10features = [    'ps_ind_14_oh_2'
                            ,'ps_ind_14_oh_3'
                            ,'ps_car_09_cat_median_range'
                            ,'ps_ind_02_cat_mean_range'
                            ,'ps_car_14_median_range'
                            ,'ps_ind_02_cat_oh_-1.0'
                            ,'ps_car_14_mean_range'
                            ,'ps_ind_02_cat_median_range'
                            ,'ps_reg_03_mean_range'
                            ,'ps_reg_03_median_range'
                            ,'ps_car_13_median_range'
                            ,'ps_car_13_mean_range'
                            ,'ps_car_10_cat_oh_2'
                            ,'ps_car_10_cat_median_range'
                            ,'ps_ind_14_oh_4'
                            ,'ps_ind_14_mean_range'
                            ,'ps_ind_14_oh_0'
                            ,'ps_ind_14_median_range'
                            ,'ps_car_02_cat_median_range'
                            ,'ps_car_03_cat_median_range'
                            ,'ps_car_07_cat_median_range'
                            ,'ps_car_11_oh_-1.0'
                            ,'ps_car_08_cat_median_range'
                            ,'ps_car_11_median_range'
                            ,'ps_car_02_cat_oh_-1.0'
                            ,'ps_car_05_cat_median_range'

                            ]

feature_names= ['ps_car_13_x_ps_reg_03'
,'ps_car_13'
,'ps_reg_03'
,'ps_car_14'
,'ps_ind_15'
,'ps_ind_05_cat'
,'ps_ind_03'
,'ps_ind_17_bin'
,'ps_reg_02'
,'ps_car_11_cat'
,'ps_car_12'
,'ps_ind_01'
,'ps_car_06_cat'
,'ps_car_01_cat'
,'ps_car_15'
,'negative_one_vals'
,'ps_reg_01'
,'ps_ind_07_bin'
,'ps_ind_05_cat_mean_range'
,'ps_car_09_cat_oh_1.0'
,'ps_ind_06_bin'
,'ps_ind_05_cat_median_range'
,'ps_car_09_cat'
,'ps_car_04_cat'
,'ps_car_07_cat'
,'ps_car_11'
,'ps_ind_02_cat'
,'ps_ind_16_bin'
,'ps_ind_09_bin'
,'ps_ind_04_cat'
,'ps_ind_08_bin'
,'ps_car_07_cat_oh_1.0'
,'ps_car_05_cat_oh_0.0'
,'ps_car_03_cat'
,'ps_car_07_cat_mean_range'
,'ps_car_11_oh_2.0'
,'ps_car_11_oh_0.0'
,'ps_car_03_cat_oh_0.0'
,'ps_car_09_cat_oh_-1.0'
,'ps_car_05_cat'
,'ps_car_08_cat_mean_range'
,'ps_car_07_cat_oh_0.0'
,'ps_ind_04_cat_oh_-1.0'
,'ps_ind_14'
,'ps_ind_18_bin'
,'ps_car_03_cat_mean_range'
,'ps_ind_15_mean_range'
,'ps_ind_15_median_range'
,'ps_car_08_cat'
,'ps_car_03_cat_oh_1.0'
,'ps_ind_01_median_range'
,'ps_car_09_cat_oh_2.0'
,'ps_ind_01_mean_range'
,'ps_ind_02_cat_oh_4.0'
,'ps_car_01_cat_median_range'
,'ps_car_01_cat_mean_range'
,'ps_ind_02_cat_oh_2.0'
,'ps_car_09_cat_oh_4.0'
,'ps_ind_13_bin'
,'ps_car_11_oh_1.0'
,'ps_car_10_cat'
,'ps_ind_03_median_range'
,'ps_ind_02_cat_oh_3.0'
,'ps_reg_01_mean_range'
,'ps_reg_01_median_range'
,'ps_ind_03_mean_range'
,'ps_ind_10_bin'
,'ps_car_09_cat_oh_3.0'
,'ps_car_07_cat_oh_-1.0'
,'ps_car_15_mean_range'
,'ps_car_09_cat_oh_0.0'
,'ps_car_05_cat_oh_1.0'
,'ps_ind_11_bin'
,'ps_car_03_cat_oh_-1.0'
,'ps_reg_02_mean_range'
,'ps_ind_12_bin'
,'ps_car_05_cat_mean_range'
,'ps_car_15_median_range'
,'ps_car_10_cat_mean_range'
,'ps_ind_04_cat_median_range'
,'ps_car_10_cat_oh_1'
,'ps_car_06_cat_mean_range'
,'ps_ind_04_cat_oh_0.0'
,'ps_ind_14_oh_1'
,'ps_car_10_cat_oh_0'
,'ps_ind_04_cat_mean_range'
,'ps_car_06_cat_median_range'
,'ps_car_11_oh_3.0'
,'ps_car_09_cat_mean_range'
,'ps_ind_04_cat_oh_1.0'
,'ps_car_11_mean_range'
,'ps_car_02_cat_mean_range'
,'ps_car_02_cat'
,'ps_car_04_cat_mean_range'
,'ps_car_02_cat_oh_0.0'
,'ps_reg_02_median_range'
,'ps_car_11_cat_median_range'
,'ps_car_11_cat_mean_range'
,'ps_car_12_mean_range'
,'ps_car_04_cat_median_range'
,'ps_car_02_cat_oh_1.0'
,'ps_ind_02_cat_oh_1.0'
,'ps_car_12_median_range'
,'ps_car_05_cat_oh_-1.0']
#len(feature_names)
#feature_names = list(set(feature_names) - set(FirstXGBTop10features))
#len(feature_names)
#y_mean = np.mean(trainingSet["y"])

param = {}
param['seed'] = 2018
param['objective'] = 'binary:logistic'#'reg:linear'
param['eta'] = 0.01
param['max_depth'] = 3
param['silent'] = 1
param['min_child_weight'] = 12
param['subsample'] = 0.7
param['colsample_bytree'] = 0.5
param['nthread'] = 16
param['print_every_n'] = 250
param['booster'] = 'gbtree'
#param['base_score'] = y_mean
#param['eval_metric'] = "logloss"
xgb_num_rounds = 1150

xgbParameters = list(param.items())

#lgb_params_1 = {
#    'learning_rate': 0.01,
#    'n_estimators': 1250,
#    'max_bin': 10,
#    'subsample': 0.8,
#    'subsample_freq': 10,
#    'colsample_bytree': 0.8, 
#    'min_child_samples': 500
#}

lgbm_params = {
          'task'              : 'train',
          'boosting_type'     : 'gbdt',
          'objective'         : 'regression',
          'num_leaves'        : 2**4, #2**4,
          'feature_fraction'  : 0.8,
          'bagging_fraction'  : 0.8,
          'bagging_freq'      : 8,#2
          'learning_rate'     : 0.01,
          'tree_method'       : 'exact',
          'min_data_in_leaf'  : 400,             
          'nthread'           : 30,
          'silent'            : False,
          'seed'              : 2017,
         }
lgbm_num_round = 1150
lgbm_early_stopping_rounds = 250

seed = 2017
folds = 5
nbags = 5
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2018
verboseeval=250

lgbmModelName = 'lgbm003'
xgbModelName  = 'xgb003'
rfModelName   = 'rf003'
etModelName   = 'et003'
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
    
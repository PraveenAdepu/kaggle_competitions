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
from numba import jit

inDir = 'C:\\Users\\SriPrav\\Documents\\R\\36Driver'
inDir = 'C:/Users/SriPrav/Documents/R/30Caesars'

X_build = pd.read_csv(inDir + "/input/X_build.csv")
X_valid = pd.read_csv(inDir + "/input/X_val.csv")


feature_names = [col for col in X_build.columns if col  not in ["customer_id", "date" ,"target","id","f_19","f_29"
                                                                 ,"customer_target_median1"
                                                                 ,"customer_target_lag_target_diff_flag"
                                                                 ,"market_target_median1","roll"
                                                                 ,"f0_target_lag1"               
                                                                 ,"f0_target_lead1"
                                                                 ,"f2_target_lag1"
                                                                 ,"f2_target_lead1"
                                                                 ,"marketf33_target_lag1"
                                                                 ,"marketf23_target_lag1"
                                                                 ,"f33_f23_target_lag1"
                                                                 ,"customer_visit_date_lag1"]]





@jit
def qwk(a1, a2, max_rat):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    w = np.zeros((max_rat + 1, max_rat + 1))
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            w[i, j] = (i - j) * (i - j)/ (max_rat * max_rat)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for i, j in zip(a1, a2):
        hist1[i] += 1
        hist2[j] += 1
        o +=  w[i, j]

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * w[i, j]

    e = e / a1.shape[0]

    return 1 - o / e

def qwk_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = qwk(y, preds, max_rat=20) 
    return 'qwk', score, True
    

lgbmParameters = {
          'task'              : 'train',
          'boosting_type'     : 'gbdt',
          'objective'         : 'regression',
          'num_leaves'        : 2**8, #2**4,
          'feature_fraction'  : 0.7,
          'bagging_fraction'  : 0.8,
          'bagging_freq'      : 5,#2
          'learning_rate'     : 0.05,
          'tree_method'       : 'exact',
          'min_data_in_leaf'  : 50,             
          'nthread'           : 30,
          'silent'            : False,
          'seed'              : 2017,
         }
lgbm_num_rounds = 1150
lgbm_early_stopping_rounds = 250

seed = 2017
folds = 5
nbags = 5
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2017
verboseeval=250

ModelName = 'lgbm01'

     
lgbmbuild = lgbm.Dataset(X_build[feature_names], X_build['target'])
lgbmval   = lgbm.Dataset(X_valid[feature_names], X_valid['target'])
             
pred_cv = np.zeros(X_valid.shape[0])

for j in range(1,nbags+1):
    print('bag ', j , ' Processing')
    bag_cv = np.zeros(X_valid.shape[0])
    lgbmParameters['seed'] =  current_seed + j         
    model = lgbm.train(lgbmParameters, lgbmbuild,lgbm_num_rounds,feval=qwk_lgb
                           ,valid_sets=[lgbmval,lgbmbuild],verbose_eval = verboseeval
                              #,early_stopping_rounds=early_stopping_rounds
                              )
    bag_cv= model.predict(X_valid[feature_names] #,num_iteration=model.best_iteration
                         )#.reshape(-1,1)
    
    pred_cv += bag_cv
    bag_score = qwk(X_valid['target'], bag_cv,20)
    print('bag ', j, '- qwk :', bag_score)
pred_cv /= nbags

fold_score = qwk(X_valid['target'], pred_cv)
print('Fold ', '- qwk :', fold_score)

pred_cv = pd.DataFrame(pred_cv)
pred_cv.columns = ["target"]
pred_cv["id"] = X_valid.id.values

sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + '.csv'
pred_cv[["id","target"]].to_csv(sub_valfile, index=False)

del pred_cv



def fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,ModelName,lgbmParameters, lgbm_num_rounds,current_seed,verboseeval):
    
    full_lgbm_num_rounds = int(lgbm_num_rounds * 1.2)
    lgbmtrain = lightgbm.Dataset(trainingSet[feature_names], trainingSet['target'])

    predfull_test = np.zeros(testingSet.shape[0]) 
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        lgbmParameters['seed'] =  current_seed + j           
        fullmodel = lightgbm.train(lgbmParameters, lgbmtrain,full_lgbm_num_rounds,feval=gini_lgb
                                   ,verbose_eval = verboseeval
                                  #,early_stopping_rounds=early_stopping_rounds
                                  )
    
        predfull_test += fullmodel.predict(testingSet[feature_names])
    predfull_test/= nbags
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["target"]
    predfull_test["id"] = testingSet.id.values
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
    predfull_test[["id","target"]].to_csv(sub_file, index=False)






param = {}
param['seed'] = 2017
param['objective'] = 'binary:logistic'#'reg:linear'
param['eta'] = 0.01
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
xgb_num_rounds = 1250

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
param['eta'] = 0.01
param['max_depth'] = 3
param['silent'] = 1
param['min_child_weight'] = 3
param['subsample'] = 0.95
param['colsample_bytree'] = 0.95
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

xgbModelName  = 'xgb002_02'

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
    
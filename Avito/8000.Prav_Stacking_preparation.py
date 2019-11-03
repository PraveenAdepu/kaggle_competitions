# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 09:13:38 2017

@author: PAdepu
"""
import pandas as pd
import numpy as np
import glob
#from sklearn.metrics import r2_score

inDir = 'C:/Users/SriPrav/Documents/R/48Avito'
# item_id,deal_probability
models = ['xgb2']
for model in models:    
    train_path = inDir + "\\submissions\\prav." + model + ".fold*[0-5].csv"
    train_from_each_file = (pd.read_csv(f) for f in glob.glob(train_path))
    train_from_each_file_df   = pd.concat(train_from_each_file)
#    train_from_each_file_df = train_from_each_file_df.groupby(['id'])['is_duplicate'].mean().reset_index(name='is_duplicate')
    train_from_each_file_df.rename(columns={'deal_probability': model}, inplace = True)
    train = train_from_each_file_df
    
    test_path = inDir + "\\submissions\\prav." + model + ".full.csv"
    test_from_each_file = (pd.read_csv(f) for f in glob.glob(test_path))
    test_from_each_file_df   = pd.concat(test_from_each_file)
    test_from_each_file_df.rename(columns={'deal_probability': model}, inplace = True)  
    test = test_from_each_file_df
 
models = ['nn06','nn07','ridge01','lgbm07','lgbm08','nn08','xgb3','nn09','nn10','nn11','nn12']#,'et50','nn102'
for model in models:    
    train_path = inDir + "\\submissions\\prav." + model + ".fold*[0-5].csv"
    train_from_each_file = (pd.read_csv(f) for f in glob.glob(train_path))
    train_from_each_file_df   = pd.concat(train_from_each_file)
#    train_from_each_file_df = train_from_each_file_df.groupby(['id'])['is_duplicate'].mean().reset_index(name='is_duplicate')
    train_from_each_file_df.rename(columns={'deal_probability': model}, inplace = True)   
    train = pd.merge(train, train_from_each_file_df, on = "item_id", how = "left")
#    del train_from_each_file_df['id']
#    train = pd.concat([train,train_from_each_file_df],  axis=1 )
    
    
    test_path = inDir + "\\submissions\\prav." + model + ".full.csv"
    test_from_each_file_df = pd.read_csv(test_path)    
    test_from_each_file_df.rename(columns={'deal_probability': model}, inplace = True)  
    test = pd.merge(test, test_from_each_file_df, on = 'item_id',how = 'left')
    
cv_file = inDir+'./input/Prav_5folds_CVindices_weekdayStratified.csv'
CV_Schema = pd.read_csv(cv_file)


testingSet = test


train_file = pd.read_csv(inDir+'/input/train.csv', parse_dates = ["activation_date"])
Prav_5folds_CVIndices = pd.read_csv(inDir+'./input/Prav_5folds_CVindices_weekdayStratified.csv')

train_file = pd.merge(train_file, Prav_5folds_CVIndices, how = 'inner', on = 'item_id')

train_file = train_file.reset_index(drop=True)

traindex = train_file.index

target_columns = ['item_id','deal_probability','CVindices']
train_target = train_file[target_columns]
del train_file
trainingSet = pd.merge(train, train_target, how = 'left', on = ['item_id'])


feature_names = cols = [col for col in trainingSet.columns if col not in ['item_id','deal_probability','CVindices']] #

trainingSet[feature_names].corr()

#a = scipy.stats.spearmanr(trainingSet[feature_names])
#a[0]
#########################################################################################################################



param = {}
param['seed'] = 201801
param['objective'] = 'reg:logistic'
param['eta'] = 0.01
param['max_depth'] = 4
param['silent'] = 1
param['min_child_weight'] = 3
param['subsample'] = 0.8
param['colsample_bytree'] = 0.8
param['nthread'] = 30
param['print_every_n'] = 100
param['eval_metric'] = "rmse"
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
folds = 5
nbags = 2
nn_epoch = 12
full_nn_epoch = 12
current_seed = 201801
verboseeval=100

lgbmModelName = 'L2_lgbm01'
xgbModelName  = 'L2_xgb05'

if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
    fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
    
#    for i in range(1, folds+1):
#        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbm_params, lgbm_num_round)
#    fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round)
    
    
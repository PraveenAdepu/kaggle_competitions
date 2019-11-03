# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:31:48 2018

@author: SriPrav
"""

import pandas as pd
import numpy as np


random_state = 20180512
np.random.RandomState(random_state)

import matplotlib.pyplot as plt
import seaborn as sns


inDir = 'C:/Users/SriPrav/Documents/R/50Santander'

train = pd.read_csv(inDir+'/input/train.csv.zip', compression='zip', header=0, sep=',')

test = pd.read_csv(inDir+'/input/test.csv.zip', compression='zip', header=0, sep=',')

Prav_5folds_CVIndices = pd.read_csv(inDir+'./input/Prav_5folds_CVindices.csv')

train = pd.merge(train, Prav_5folds_CVIndices, how = 'inner', on = 'ID')

print("Train rows and columns : ", train.shape)
print("Test rows and columns : ", test.shape)

#plt.figure(figsize=(8,6))
#plt.scatter(range(train.shape[0]), np.sort(train['target'].values))
#plt.xlabel('index', fontsize=12)
#plt.ylabel('Target', fontsize=12)
#plt.title("Target Distribution", fontsize=14)
#plt.show()
#
#plt.figure(figsize=(12,8))
#sns.distplot(train["target"].values, bins=50, kde=False)
#plt.xlabel('Target', fontsize=12)
#plt.title("Target Histogram", fontsize=14)
#plt.show()
#
#plt.figure(figsize=(12,8))
#sns.distplot( np.log1p(train["target"].values), bins=50, kde=False)
#plt.xlabel('Target', fontsize=12)
#plt.title("Log of Target Histogram", fontsize=14)
#plt.show()
#
#missing_df = train.isnull().sum(axis=0).reset_index()
#missing_df.columns = ['column_name', 'missing_count']
#missing_df = missing_df[missing_df['missing_count']>0]
#missing_df = missing_df.sort_values(by='missing_count')
#missing_df
#
#dtype_df = train.dtypes.reset_index()
#dtype_df.columns = ["Count", "Column Type"]
#dtype_df.groupby("Column Type").aggregate('count').reset_index()

unique_df = train.nunique().reset_index()
unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"]==1]
constant_df.shape

str(constant_df.col_name.tolist())

constant_columns = constant_df.col_name.tolist()

### Get the X and y variables for building model ###
train = train.drop(constant_df.col_name.tolist() , axis=1)
test = test.drop(constant_df.col_name.tolist(), axis=1)


feature_names = [c for c in train if c not in ['ID', 'target', 'CVindices']]

trainingSet = train.copy()
testingSet = test.copy()
trainingSet['target'] = np.log1p(trainingSet["target"])

param = {}
param['seed'] = 201801
param['objective'] = 'reg:linear'
param['eta'] = 0.01
param['max_depth'] = 5
param['silent'] = 1
param['min_child_weight'] = 3
param['subsample'] = 0.5
param['colsample_bytree'] = 0.5
param['nthread'] = 30
param['print_every_n'] = 100
param['eval_metric'] = "rmse"
xgb_num_rounds = 510

xgbParameters = list(param.items())

xgbModelName  = 'xgb001'

nbags = 5
current_seed = 201801
verboseeval = 50
folds = 5


if __name__ == '__main__':
    model_results = []
    model_results.append(xgbModelName)  
    for i in range(1, folds+1):
        fold_score = train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
        model_results.append(fold_score)
    LB = 0
    model_results.append(LB)
    ScoreDiff = 0
    model_results.append(ScoreDiff)
    model_results.append(xgbParameters)
    
    results = pd.DataFrame(model_results).T
    results.columns = ["ModelName","fold1","fold2", "fold3", "fold4","fold5","LB","ScoreDiff","Parameters"]
    results['folds_mean'] = results[["fold1","fold2", "fold3", "fold4","fold5"]].mean(axis=1)
    sub_results = inDir + '/ModelLogs/Prav.Modellog.csv'
    results[["ModelName","fold1","fold2", "fold3", "fold4","fold5","folds_mean","LB","ScoreDiff","Parameters"]].to_csv(sub_results, mode='a', header=False, index=False)
    
    fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)


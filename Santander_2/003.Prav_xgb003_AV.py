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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

inDir = 'C:/Users/SriPrav/Documents/R/50Santander'

train = pd.read_csv(inDir+'/input/train.csv.zip', compression='zip', header=0, sep=',')

test = pd.read_csv(inDir+'/input/test.csv.zip', compression='zip', header=0, sep=',')

Prav_5folds_CVIndices = pd.read_csv(inDir+'./input/Prav_adversarial_validation.csv')

del Prav_5folds_CVIndices['p']

train = pd.merge(train, Prav_5folds_CVIndices, how = 'inner', on = 'ID')

print("Train rows and columns : ", train.shape)
print("Test rows and columns : ", test.shape)

######################################################################################################################
######################################################################################################################


unique_df = train.nunique().reset_index()
unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"]==1]
constant_df.shape

str(constant_df.col_name.tolist())

constant_columns = constant_df.col_name.tolist()

### Get the X and y variables for building model ###
train = train.drop(constant_df.col_name.tolist() , axis=1)
test = test.drop(constant_df.col_name.tolist(), axis=1)

# Check and remove duplicate columns
colsToRemove = []
colsScaned = []
dupList = {}

columns = train.columns

for i in range(len(columns)-1):
    v = train[columns[i]].values
    dupCols = []
    for j in range(i+1,len(columns)):
        if np.array_equal(v, train[columns[j]].values):
            colsToRemove.append(columns[j])
            if columns[j] not in colsScaned:
                dupCols.append(columns[j]) 
                colsScaned.append(columns[j])
                dupList[columns[i]] = dupCols
                
# remove duplicate columns in the training set
train.drop(colsToRemove, axis=1, inplace=True) 

# remove duplicate columns in the testing set
test.drop(colsToRemove, axis=1, inplace=True)

def drop_sparse(train, test):
    flist = [x for x in train.columns if not x in ['ID','target','CVindices']]
    for f in flist:
        if len(np.unique(train[f]))<2:
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
    return train, test

train, test = drop_sparse(train, test)
print("Removed `{}` Duplicate Columns\n".format(len(dupList)))
print(dupList)

######################################################################################################################
######################################################################################################################

def add_SumValues(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target','CVindices']]
    if 'SumValues' in features:
        train.insert(1, 'SumValues', (train[flist] != 0).astype(int).sum(axis=1))
        test.insert(1, 'SumValues', (test[flist] != 0).astype(int).sum(axis=1))
    flist = [x for x in train.columns if not x in ['ID','target','CVindices']]
    return train, test

train, test = add_SumValues(train, test, ['SumValues'])

def add_SumZeros(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target','CVindices','SumValues']]
    if 'SumZeros' in features:
        train.insert(1, 'SumZeros', (train[flist] == 0).astype(int).sum(axis=1))
        test.insert(1, 'SumZeros', (test[flist] == 0).astype(int).sum(axis=1))
    flist = [x for x in train.columns if not x in ['ID','target','CVindices','SumValues']]

    return train, test

train, test = add_SumZeros(train, test, ['SumZeros'])

def add_skewKurtosis(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target','CVindices','SumValues','SumZeros']]
    if 'skewKurtosis' in features:
        train['skewness'] = train[flist].skew(axis=1)
        train['kurtosis'] = train[flist].kurtosis(axis=1)
        
        test['skewness'] = test[flist].skew(axis=1)
        test['kurtosis'] = test[flist].kurtosis(axis=1)

    return train, test

train, test = add_skewKurtosis(train, test, ['skewKurtosis'])

def add_OtherAgg(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target','CVindices','SumZeros','SumValues','skewness','kurtosis']]
    train_fill = train[flist].replace(0,np.nan)
    test_fill = test[flist].replace(0,np.nan)
    if 'OtherAgg' in features:
        train['Mean'] = train_fill.mean(axis=1)  
        train['Median'] = train_fill.median(axis=1)
        #train['Mode'] = train_fill.mode(axis=1)
        train['Max'] = train_fill.max(axis=1)
        train['Var'] = train_fill.var(axis=1)
        train['Std'] = train_fill.std(axis=1)
#        train['nonskewness'] = train_fill.skew(axis=1)
#        train['nonkurtosis'] = train_fill.kurtosis(axis=1)
        
        test['Mean'] = test_fill.mean(axis=1)
        test['Median'] = test_fill.median(axis=1)
        #test['Mode'] = test_fill.mode(axis=1)
        test['Max'] = test_fill.max(axis=1)
        test['Var'] = test_fill.var(axis=1)
        test['Std'] = test_fill.std(axis=1)
#        test['nonskewness'] = test_fill.skew(axis=1)
#        test['nonkurtosis'] = test_fill.kurtosis(axis=1)
    return train, test

train, test = add_OtherAgg(train, test, ['OtherAgg'])

plt.figure(figsize=(12,8))
sns.distplot(train["Max"].values, bins=50, kde=False)
plt.xlabel('Target', fontsize=12)
plt.title("Target Histogram", fontsize=14)
plt.show()

plt.figure(figsize=(12,8))
sns.distplot( np.log1p(train["Max"].values), bins=50, kde=False)
plt.xlabel('Target', fontsize=12)
plt.title("Log of Target Histogram", fontsize=14)
plt.show()

plt.figure(figsize=(12,8))
sns.distplot(train["Mean"].values, bins=50, kde=False)
plt.xlabel('Target', fontsize=12)
plt.title("Target Histogram", fontsize=14)
plt.show()

plt.figure(figsize=(12,8))
sns.distplot( np.log1p(train["Mean"].values), bins=50, kde=False)
plt.xlabel('Target', fontsize=12)
plt.title("Log of Target Histogram", fontsize=14)
plt.show()

plt.figure(figsize=(12,8))
sns.distplot(train["Median"].values, bins=50, kde=False)
plt.xlabel('Target', fontsize=12)
plt.title("Target Histogram", fontsize=14)
plt.show()

plt.figure(figsize=(12,8))
sns.distplot( np.log1p(train["Median"].values), bins=50, kde=False)
plt.xlabel('Target', fontsize=12)
plt.title("Log of Target Histogram", fontsize=14)
plt.show()

train['Max'] = np.log1p(train["Max"])
train['Mean'] = np.log1p(train["Mean"])
train['Median'] = np.log1p(train["Median"])

test['Max'] = np.log1p(test["Max"])
test['Mean'] = np.log1p(test["Mean"])
test['Median'] = np.log1p(test["Median"])

train.fillna(0, inplace=True)
test.fillna(0, inplace=True)


flist = [x for x in train.columns if not x in ['ID','target','CVindices']]

flist_kmeans = []
for ncl in range(2,11):
    cls = KMeans(n_clusters=ncl)
    cls.fit_predict(train[flist].values)
    train['kmeans_cluster_'+str(ncl)] = cls.predict(train[flist].values)
    test['kmeans_cluster_'+str(ncl)] = cls.predict(test[flist].values)
    flist_kmeans.append('kmeans_cluster_'+str(ncl))
print(flist_kmeans)


######################################################################################################################
######################################################################################################################


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

xgbModelName  = 'xgb003_AV'

nbags = 10
current_seed = 201801
verboseeval = 50
folds = 1


if __name__ == '__main__':
    model_results = []
    model_results.append(xgbModelName)  
    for i in range(1, folds+1):
        fold_score = train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, xgb_num_rounds,current_seed,verboseeval)
        model_results.append(fold_score)
    LB = 0
    model_results.append(LB)
    model_results.append(LB)
    model_results.append(LB)
    model_results.append(LB)
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


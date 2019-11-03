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

inDir = 'C:/Users/SriPrav/Documents/R/50Santander'
######################################################################################################################
######################################################################################################################

train = pd.read_csv(inDir+'/input/train.csv.zip', compression='zip', header=0, sep=',')

test = pd.read_csv(inDir+'/input/test.csv.zip', compression='zip', header=0, sep=',')

print("Train rows and columns : ", train.shape)
print("Test rows and columns : ", test.shape)

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

def aggregate_row(row):
    non_zero_values = row.iloc[row.nonzero()]
    aggs = {'non_zero_mean': non_zero_values.mean(),
            'non_zero_max': non_zero_values.max(),
            'non_zero_min': non_zero_values.min(),
            'non_zero_std': non_zero_values.std(),
            'non_zero_sum': non_zero_values.sum(),
            'non_zero_count': non_zero_values.count(),
            'non_zero_fraction': non_zero_values.count() / row.count()
            }
    return pd.Series(aggs)


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
        train['Sum'] = train_fill.sum(axis=1)
        train['Min'] = train_fill.min(axis=1)
#        train['nonkurtosis'] = train_fill.kurtosis(axis=1)
        
        test['Mean'] = test_fill.mean(axis=1)
        test['Median'] = test_fill.median(axis=1)
        #test['Mode'] = test_fill.mode(axis=1)
        test['Max'] = test_fill.max(axis=1)
        test['Var'] = test_fill.var(axis=1)
        test['Std'] = test_fill.std(axis=1)
        test['Sum'] = test_fill.sum(axis=1)
        test['Min'] = test_fill.min(axis=1)
#        test['nonkurtosis'] = test_fill.kurtosis(axis=1)
        
    return train, test

train, test = add_OtherAgg(train, test, ['OtherAgg'])

train["NonZeroRatio"] = train['SumValues']/(train['SumValues']+train['SumZeros'])
test["NonZeroRatio"] = test['SumValues']/(test['SumValues']+test['SumZeros'])

train["MedianMeanRatio"] = train['Median']/train['Mean']
test["MedianMeanRatio"] = test['Median']/test['Mean']

train["MaxMinRatio"] = train['Max']/train['Min']
test["MaxMinRatio"] = test['Max']/test['Min']

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


####################################################################################################################################

train.to_hdf(inDir + '/input/train_2.h5','train_2') # 4746
test.to_hdf(inDir + '/input/test_2.h5','test_2')    # 4745

####################################################################################################################################

train = pd.read_hdf(inDir + '/input/train_2.h5','train_2')
test = pd.read_hdf(inDir + '/input/test_2.h5','test_2')
####################################################################################################################################



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

#plt.figure(figsize=(12,8))
#sns.distplot(train["Median"].values, bins=50, kde=False)
#plt.xlabel('Target', fontsize=12)
#plt.title("Target Histogram", fontsize=14)
#plt.show()

#plt.figure(figsize=(12,8))
#sns.distplot( np.log1p(train["Median"].values), bins=50, kde=False)
#plt.xlabel('Target', fontsize=12)
#plt.title("Log of Target Histogram", fontsize=14)
#plt.show()

#train['Max'] = np.log1p(train["Max"])
#train['Min'] = np.log1p(train["Min"])
#train['Mean'] = np.log1p(train["Mean"])
#train['Median'] = np.log1p(train["Median"])
#train['Sum'] = np.log1p(train["Sum"])
#
#test['Max'] = np.log1p(test["Max"])
#test['Min'] = np.log1p(test["Min"])
#test['Mean'] = np.log1p(test["Mean"])
#test['Median'] = np.log1p(test["Median"])
#test['Sum'] = np.log1p(test["Sum"])

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

from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

# 4730
flist = [x for x in train.columns if not x in ['ID','target','CVindices'
                                               ,'SumZeros','SumValues','skewness','kurtosis'
                                               ,'kmeans_cluster_2', 'kmeans_cluster_3', 'kmeans_cluster_4', 'kmeans_cluster_5', 'kmeans_cluster_6'
                                               , 'kmeans_cluster_7', 'kmeans_cluster_8', 'kmeans_cluster_9', 'kmeans_cluster_10'
                                               ,'Max','Mean','Median','Var','Std','Sum','NonZeroRatio','MedianMeanRatio','MaxMinRatio','Min']]


N_COMP = 50
print("\nTrain shape: {}\nTest shape: {}".format(train[flist].shape, test[flist].shape))

print("\nStart decomposition process...")
print("PCA")
pca = PCA(n_components=N_COMP, random_state=random_state)
pca_results_train = pca.fit_transform(train[flist])
pca_results_test = pca.transform(test[flist])

print("tSVD")
tsvd = TruncatedSVD(n_components=N_COMP, random_state=random_state)
tsvd_results_train = tsvd.fit_transform(train[flist])
tsvd_results_test = tsvd.transform(test[flist])

print("ICA")
ica = FastICA(n_components=N_COMP, random_state=random_state)
ica_results_train = ica.fit_transform(train[flist])
ica_results_test = ica.transform(test[flist])

print("GRP")
grp = GaussianRandomProjection(n_components=N_COMP, eps=0.1, random_state=random_state)
grp_results_train = grp.fit_transform(train[flist])
grp_results_test = grp.transform(test[flist])

print("SRP")
srp = SparseRandomProjection(n_components=N_COMP, dense_output=True, random_state=random_state)
srp_results_train = srp.fit_transform(train[flist])
srp_results_test = srp.transform(test[flist])

print("FA")
fa = FactorAnalysis(n_components=N_COMP, random_state=random_state)
fa_results_train = fa.fit_transform(train[flist])
fa_results_test = fa.transform(test[flist])


print("Append decomposition components to datasets...")
for i in range(1, N_COMP + 1):
    train['pca_' + str(i)] = pca_results_train[:, i - 1]
    test['pca_' + str(i)] = pca_results_test[:, i - 1]

    train['ica_' + str(i)] = ica_results_train[:, i - 1]
    test['ica_' + str(i)] = ica_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]
    
    train['fa_' + str(i)] = fa_results_train[:, i - 1]
    test['fa_' + str(i)] = fa_results_test[:, i - 1]
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))

######################################################################################################################
######################################################################################################################


Prav_5folds_CVIndices = pd.read_csv(inDir+'./input/Prav_adversarial_validation.csv')

del Prav_5folds_CVIndices['p']

train = pd.merge(train, Prav_5folds_CVIndices, how = 'inner', on = 'ID')    
    
#feature_names = [c for c in train if c not in ['ID', 'target', 'CVindices']+list(flist)]

pca_feature_names  = [col for col in train.columns if 'pca_' in col]
ica_feature_names  = [col for col in train.columns if 'ica_' in col]
tsvd_feature_names  = [col for col in train.columns if 'tsvd_' in col]
grp_feature_names  = [col for col in train.columns if 'grp_' in col]
srp_feature_names  = [col for col in train.columns if 'srp_' in col]
fa_feature_names  = [col for col in train.columns if 'fa_' in col]
cluster_feature_names  = [col for col in train.columns if 'kmeans_cluster' in col]

feature_names =  list(srp_feature_names) + list(fa_feature_names)  + ['SumZeros','Max','Mean','Var','Std','Sum','NonZeroRatio','Min','skewness','kurtosis','Median','MedianMeanRatio','MaxMinRatio','SumValues']#,
#
trainingSet = train.copy()
testingSet = test.copy()
trainingSet['target'] = np.log1p(trainingSet["target"])

# EXECUTE FRAMEWORK SCRIPT HERE #

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


xgbModelName  = 'xgb006_AV'

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

######################################################################################################################
######################################################################################################################

del train['CVindices']    
Prav_5folds_CVIndices = pd.read_csv(inDir+'./input/Prav_5folds_CVindices.csv')
train = pd.merge(train, Prav_5folds_CVIndices, how = 'inner', on = 'ID')

#feature_names = [c for c in train if c not in ['ID', 'target', 'CVindices']+list(flist)]

trainingSet = train.copy()
testingSet = test.copy()
trainingSet['target'] = np.log1p(trainingSet["target"])


param = {}
param['seed'] = 201801
param['objective'] = 'reg:linear'
param['eta'] = 0.01
param['max_depth'] = 5
param['silent'] = 1
param['min_child_weight'] = 12
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 30
param['print_every_n'] = 100
param['eval_metric'] = "rmse"
xgb_num_rounds = 510

xgbParameters = list(param.items())


xgbModelName  = 'xgb006'

nbags = 10
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

######################################################################################################################
######################################################################################################################

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 22:15:27 2017

@author: SriPrav
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

user = 'SriPrav'
inDir = 'C:/Users/'+user+'/Documents/R/26Mercedes'

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
trainingSet = pd.read_csv(train_file)
testingSet = pd.read_csv(test_file)
print(trainingSet.shape) # (4209, 378)
print(testingSet.shape)  # (4209, 377)

for c in trainingSet.columns:
    if trainingSet[c].dtype == 'object':
        print('object column ', c , ' Processing') 
        lbl = LabelEncoder() 
        lbl.fit(list(trainingSet[c].values) + list(testingSet[c].values)) 
        trainingSet[c] = lbl.transform(list(trainingSet[c].values))
        testingSet[c] = lbl.transform(list(testingSet[c].values))

y_mean = np.mean(trainingSet["y"])

# shape        
print('Shape train: {}\nShape test: {}'.format(trainingSet.shape, testingSet.shape))


##Add decomposed components: PCA / ICA etc.
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=42)
tsvd_results_train = tsvd.fit_transform(trainingSet.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(testingSet)

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(trainingSet.drop(["y"], axis=1))
pca2_results_test = pca.transform(testingSet)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(trainingSet.drop(["y"], axis=1))
ica2_results_test = ica.transform(testingSet)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(trainingSet.drop(["y"], axis=1))
grp_results_test = grp.transform(testingSet)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(trainingSet.drop(["y"], axis=1))
srp_results_test = srp.transform(testingSet)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    trainingSet['pca_' + str(i)] = pca2_results_train[:,i-1]
    testingSet['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    trainingSet['ica_' + str(i)] = ica2_results_train[:,i-1]
    testingSet['ica_' + str(i)] = ica2_results_test[:, i-1]
    
#    trainingSet['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
#    testingSet['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]
#
#    trainingSet['grp_' + str(i)] = grp_results_train[:, i - 1]
#    testingSet['grp_' + str(i)] = grp_results_test[:, i - 1]
#
#    trainingSet['srp_' + str(i)] = srp_results_train[:, i - 1]
#    testingSet['srp_' + str(i)] = srp_results_test[:, i - 1]
# shape        
print('Shape train: {}\nShape test: {}'.format(trainingSet.shape, testingSet.shape))
   
#    train['tsvd_' + str(i)] = tsvd_results_train[:,i-1]
#    test['tsvd_' + str(i)] = tsvd_results_test[:, i-1]


#train_features_12 = inDir + "/input/train_question_freq_features_from_porter_02.csv"
#test_features_12 = inDir + "/input/test_question_freq_features_from_porter_02.csv"
#train_features_12 = pd.read_csv(train_features_12)
#test_features_12 = pd.read_csv(test_features_12)
#print(train_features_12.shape) # (404290, 36)
#print(test_features_12.shape)  # (2345796, 34)
#
#del train_features_12['is_duplicate']
#test_features_12.rename(columns={'id': 'test_id'}, inplace=True)
#
#trainingSet = pd.merge(trainingSet, train_features_12, how = 'left', on = 'id')
#testingSet = pd.merge(testingSet, test_features_12, how = 'left', on = 'test_id')

cv_file = inDir + "/CVSchema/Prav_CVindices_10folds.csv"
CV_Schema = pd.read_csv(cv_file)

trainingSet = pd.merge(trainingSet, CV_Schema, how = 'left', on = ['ID'])

feature_names = cols = [col for col in trainingSet.columns if col not in ['y','CVindices','X0','X1','X2','X3','X4','X5','X6','X8']] #'ID',
                                                            
#trainingSet = trainingSet.replace(np.inf, np.nan) 
#testingSet = testingSet.replace(np.inf, np.nan)
#trainingSet = trainingSet.fillna(0)   
#testingSet = testingSet.fillna(0)

#trainingSet[feature_names].dtypes
#testingSet[feature_names].dtypes
#
#trainingSet[feature_names] =trainingSet[feature_names].apply(pd.to_numeric)
#testingSet[feature_names] =testingSet[feature_names].apply(pd.to_numeric)

#trainingSet[feature_names] =trainingSet[feature_names].astype(np.float64)
#testingSet[feature_names] =testingSet[feature_names].astype(np.float64)

##################################################################################################################################
##################################################################################################################################
#for column in features_to_use:
#    print(column)
#    print(' AUC:', roc_auc_score(trainingSet['is_duplicate'], trainingSet[column]))
##################################################################################################################################
##################################################################################################################################
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

param = {}
param['seed'] = 2017
param['objective'] = 'reg:linear'
param['eta'] = 0.005
param['max_depth'] = 4
param['silent'] = 1
param['min_child_weight'] = 3
param['subsample'] = 0.8
param['colsample_bytree'] = 0.7
param['nthread'] = 10
param['print_every_n'] = 100
param['base_score'] = y_mean
#param['eval_metric'] = "logloss"
xgb_num_rounds = 710

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
verboseeval=400

lgbmModelName = 'lgbm003'
xgbModelName  = 'xgb004'
rfModelName   = 'rf101'
etModelName   = 'et101'
fmModelName   = 'fm100'
adaModelName   = 'ada001'
gbdtModelName   = 'gbdt002'
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
    
    for i in range(1, folds+1):
        train_regression("ada",trainingSet, testingSet,feature_names,i,nbags,adaModelName,current_seed)
    fulltrain_regression("ada",trainingSet, testingSet,feature_names,nbags,adaModelName,current_seed)
    
    for i in range(1, folds+1):
        train_regression("gbdt",trainingSet, testingSet,feature_names,i,nbags,gbdtModelName,current_seed)
    fulltrain_regression("gbdt",trainingSet, testingSet,feature_names,nbags,gbdtModelName,current_seed)


#    dump_svmlight_file(trainingSet[feature_names],trainingSet['is_duplicate'],inDir+"/input/X_trainingSet.svm")
#    dump_svmlight_file(testingSet[feature_names],np.zeros(testingSet.shape[0]),inDir+"/input/X_testingSet.svm")
#
#    for i in range(1, folds+1):
#        train_libfm_regression(trainingSet, testingSet,feature_names,i,nbags,fmModelName,xgbParameters, xgb_num_rounds)
#    fulltrain_libfm_regression(trainingSet, testingSet,feature_names,nbags,fmModelName,xgbParameters, xgb_num_rounds)












x_train = pd.DataFrame()
x_test = pd.DataFrame()
x_train = trainingSet[features_to_use]
x_test  = testingSet[features_to_use]


x_train = x_train.apply(pd.to_numeric)
x_test = x_test.apply(pd.to_numeric)

y_train = trainingSet['is_duplicate'].apply(pd.to_numeric).values

from sklearn.cross_validation import train_test_split
x_build, x_valid, y_build, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

import xgboost as xgb

#params = {}
#params['objective'] = 'binary:logistic'
#params['eval_metric'] = 'logloss'
#params['eta'] = 0.02
#params['max_depth'] = 4

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
num_rounds = 8010
plst = list(param.items())


d_train = xgb.DMatrix(x_build, label=y_build)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

#bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
model = xgb.train(plst, 
                      d_train, 
                      num_rounds, 
                      watchlist, 
                      verbose_eval = 500 ,
                      early_stopping_rounds=20
                      )

#d_test = xgb.DMatrix(x_test)
#p_test = model.predict(d_test)
#
#sub = pd.DataFrame()
#sub['test_id'] = testingSet['test_id']
#sub['is_duplicate'] = p_test
#sub.to_csv('./submissions/Prav_xgb04.csv', index=False)
##########################################################################################
# Full model training
########################################################################################## 

fullnum_rounds = int(num_rounds * 1.2)
xgbtrain = xgb.DMatrix( x_train, label=y_train)
xgtest = xgb.DMatrix(x_test)
watchlistfull = [ (xgbtrain,'train') ]
                 
model = xgb.train(plst, 
                      d_train, 
                      num_rounds, 
                      watchlist, 
                      verbose_eval = 500 ,
                      early_stopping_rounds=20
                      )



fullmodel = xgb.train(plst, 
                          xgbtrain, 
                          fullnum_rounds, 
                          watchlistfull,
                          verbose_eval = 500,
                          )

predfull_test = fullmodel.predict(xgtest)
predfull_test = pd.DataFrame(predfull_test)
predfull_test.columns = ["is_duplicate"]
predfull_test["test_id"] = testingSet.test_id.values
predfull_test = predfull_test[['test_id','is_duplicate']]
sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.xgb07.full' + '.csv'
predfull_test.to_csv(sub_file, index=False)
    
def fulltrain_xgboost(bags):
    xgbtrain = xgb.DMatrix( x_train, label=y_train)
    watchlist = [ (xgbtrain,'train') ]
    fullmodel = xgb.train(plst, 
                              xgbtrain, 
                              fullnum_rounds, 
                              watchlist,
                              verbose_eval = 100,
                              )
    xgtest = xgb.DMatrix(x_test)
    predfull_test = fullmodel.predict(xgtest)
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["is_duplicate"]
    predfull_test["test_id"] = testingSet.test_id.values
    predfull_test = predfull_test[['test_id','is_duplicate']]
    sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.xgb07.full' + '.csv'
    predfull_test.to_csv(sub_file, index=False)
    
folds = 5
if __name__ == '__main__':
    #for i in range(1, folds+1):
        #train_xgboost(i)
    fulltrain_xgboost(folds)
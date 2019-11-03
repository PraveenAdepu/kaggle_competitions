
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 22:03:59 2017

@author: SriPrav
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb



pal = sns.color_palette()


import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, hstack
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string
from nltk.corpus import stopwords
from collections import Counter
np.seterr(divide='ignore', invalid='ignore')
stops = set(stopwords.words("english"))
from sklearn.metrics import roc_auc_score, log_loss


inDir = 'C:/Users/SriPrav/Documents/R/23Quora'

train_file = inDir + "/input/TrainingSet01.csv"
test_file = inDir + "/input/TestingSet01.csv"
train = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train.shape) # (404290, 6)
print(test_df.shape)  # (2345796, 3)

train_df = train # .groupby('id').first().reset_index()


train_features_06 = inDir + "/input/train_features_06.csv"
test_features_06 = inDir + "/input/test_features_06.csv"
train_features_06 = pd.read_csv(train_features_06)
test_features_06 = pd.read_csv(test_features_06)
print(train_features_06.shape) # (404290, 36)
print(test_features_06.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_06, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_06, how = 'left', on = 'test_id')

train_features_07 = inDir + "/input/train_features_07.csv"
test_features_07 = inDir + "/input/test_features_07.csv"
train_features_07 = pd.read_csv(train_features_07)
test_features_07 = pd.read_csv(test_features_07)
print(train_features_07.shape) # (404290, 36)
print(test_features_07.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_07, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_07, how = 'left', on = 'test_id')

cv_file = inDir + "/CVSchema/Prav_CVindices_5folds.csv"
CV_Schema = pd.read_csv(cv_file)

train_df = pd.merge(train_df, CV_Schema, how = 'left', on = ['id','qid1','qid2'])


#z_noun_match,z_match_ratio,z_word_match,z_tfidf_word_match
#print('Original AUC:', roc_auc_score(train_df['is_duplicate'], train_df['z_noun_match'].fillna(0)))
#print('   TFIDF AUC:', roc_auc_score(train_df['is_duplicate'], train_df['z_match_ratio'].fillna(0)))
#print('Original AUC:', roc_auc_score(train_df['is_duplicate'], train_df['z_word_match'].fillna(0)))
#print('   TFIDF AUC:', roc_auc_score(train_df['is_duplicate'], train_df['z_tfidf_word_match'].fillna(0)))

features_to_use = cols = [col for col in train_df.columns if col not in ['id','qid1','qid2','question1', 'question2','is_duplicate','CVindices','test_id'
                                                                       ,'q1_2pairwords_matched_q2_ratio','z_tfidf_word_match']] 

train_df = train_df.replace(np.inf, np.nan) 
test_df = test_df.replace(np.inf, np.nan)
train_df = train_df.fillna(0)   
test_df = test_df.fillna(0)

#trainfeatures_to_use = cols = [col for col in train_df.columns if col not in ['qid1','qid2','question1', 'question2',
#                                                                       'q1_2pairwords_matched_q2_ratio','z_tfidf_word_match']] 

##################################################################################################################################
##################################################################################################################################
#for column in features_to_use:
#    print(column)
#    print(' AUC:', roc_auc_score(train_df['is_duplicate'], train_df[column]))
##################################################################################################################################
##################################################################################################################################
#x_train = pd.DataFrame()
#x_test = pd.DataFrame()
#x_train = train_df[features_to_use]
#x_test  = test_df[features_to_use]
#
#
#train_df_xgb = train_df[trainfeatures_to_use].apply(pd.to_numeric)
#
#train_df_xgb['id'] = train_df_xgb['id'].apply(pd.to_numeric).values
#train_df_xgb['CVindices'] = train_df_xgb['CVindices'].apply(pd.to_numeric).values
#train_df_xgb['is_duplicate'] = train_df_xgb['is_duplicate'].apply(pd.to_numeric).values

x_test = test_df[features_to_use].apply(pd.to_numeric)


i = 1
def train_lgbm(i):
    print('Fold ', i , ' Processing')
    X_build = train_df[train_df['CVindices'] != i] # 636112
    #X_build = X_build.groupby('id').first().reset_index() # 331085
    X_val   = train_df[train_df['CVindices'] == i]
    
    print(X_build.shape) # (404290, 6)
    print(X_val.shape)  # (2345796, 3)

    X_train = X_build[features_to_use]
    X_valid = X_val[features_to_use]
    
    X_train = X_train.fillna(0) 
    X_valid = X_valid.fillna(0)
    
    X_train = X_train.apply(pd.to_numeric)
    X_valid = X_valid.apply(pd.to_numeric)

    X_trainy = X_build['is_duplicate']
    X_validy = X_val['is_duplicate']
    
    X_trainy = X_trainy.apply(pd.to_numeric).values
    X_validy = X_validy.apply(pd.to_numeric).values
    
    gbm = lgb.LGBMRegressor(   max_depth=6,
                               num_leaves=128, 
                               n_estimators=250,
                               min_child_weight=1, 
                               learning_rate=0.01, 
                               colsample_bytree=0.7,                                           
                               subsample=0.7,
                               nthread=24,
                               seed=2017)
    gbm.fit(X_train, X_trainy,
            eval_set=[(X_valid, X_validy)],
            eval_metric= {'binary_logloss','auc'},#'binary_logloss',
            verbose=True #,
            #early_stopping_rounds=5
            )
    pred_cv = gbm.predict(X_valid)
    fold_score = log_loss(X_validy, pred_cv)
    print('Fold ', i, '- logloss:', fold_score)
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = ["is_duplicate"]
    pred_cv["id"] = X_val.id.values
    pred_cv = pred_cv[['id','is_duplicate']]
    sub_valfile = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.lgbm51.fold' + str(i) + '.csv'
    pred_cv.to_csv(sub_valfile, index=False)
    
    
    pred_test = gbm.predict(x_test)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["is_duplicate"]
    pred_test["test_id"] = test_df.test_id.values
    pred_test = pred_test[['test_id','is_duplicate']]
    sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.lgbm51.fold' + str(i) + '-test' + '.csv'
    pred_test.to_csv(sub_file, index=False)
    del pred_cv
    del pred_test

##########################################################################################
# Full model training
########################################################################################## 

fullnum_rounds = int(num_rounds * 1.2)
   
def fulltrain_xgboost(nbags):
    predfull_test = np.zeros(x_test.shape[0]) 
    xgbtrain = xgb.DMatrix(train_df[features_to_use].apply(pd.to_numeric), label=train_df['is_duplicate'].apply(pd.to_numeric).values, missing = 0)
    watchlist = [ (xgbtrain,'train') ]
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        fullmodel = xgb.train(plst, 
                                  xgbtrain, 
                                  fullnum_rounds, 
                                  watchlist,
                                  verbose_eval = 1000,
                                  )
        xgtest = xgb.DMatrix(x_test, missing = 0)
        predfull_test += fullmodel.predict(xgtest)        
    predfull_test/= nbags
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["is_duplicate"]
    predfull_test["test_id"] = test_df.test_id.values
    predfull_test = predfull_test[['test_id','is_duplicate']]
    sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.xgb50.full' + '.csv'
    predfull_test.to_csv(sub_file, index=False)
    
folds = 5
nbags = 2
i = 1
if __name__ == '__main__':
    #for i in range(1, folds+1):
    train_lgbm(i)
    fulltrain_xgboost(nbags)
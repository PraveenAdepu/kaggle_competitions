# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 21:41:57 2017

@author: SriPrav
"""

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

inDir = 'C:/Users/SriPrav/Documents/R/23Quora'

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train_df.shape) # (404290, 6)
print(test_df.shape)  # (2345796, 3)

train_features_01 = inDir + "/input/train_features_01.csv"
test_features_01 = inDir + "/input/test_features_01.csv"
train_features_01 = pd.read_csv(train_features_01)
test_features_01 = pd.read_csv(test_features_01)
print(train_features_01.shape) # (404290, 36)
print(test_features_01.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_01, how = 'left', on = ['id','qid1','qid2'])

test_features_01.rename(columns={'id': 'test_id'}, inplace=True)

test_df = pd.merge(test_df, test_features_01, how = 'left', on = 'test_id')

train_features_02 = inDir + "/input/train_features_02.csv"
test_features_02 = inDir + "/input/test_features_02.csv"
train_features_02 = pd.read_csv(train_features_02)
test_features_02 = pd.read_csv(test_features_02)
print(train_features_02.shape) # (404290, 36)
print(test_features_02.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_02, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_02, how = 'left', on = 'test_id')

train_features_03 = inDir + "/input/train_features_03.csv"
test_features_03 = inDir + "/input/test_features_03.csv"
train_features_03 = pd.read_csv(train_features_03)
test_features_03 = pd.read_csv(test_features_03)
print(train_features_03.shape) # (404290, 36)
print(test_features_03.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_03, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_03, how = 'left', on = 'test_id')

cv_file = inDir + "/CVSchema/Prav_CVindices_5folds.csv"
CV_Schema = pd.read_csv(cv_file)

train_df = pd.merge(train_df, CV_Schema, how = 'left', on = ['id','qid1','qid2'])

#train_qs = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist()).astype(str)
#test_qs = pd.Series(test_df['question1'].tolist() + test_df['question2'].tolist()).astype(str)
#
#dist_train = train_qs.apply(lambda x: len(x.split(' ')))
#dist_test = test_qs.apply(lambda x: len(x.split(' ')))
#
## If a word appears only once, we ignore it completely (likely a typo)
## Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
#def get_weight(count, eps=10000, min_count=2):
#    if count < min_count:
#        return 0
#    else:
#        return 1 / (count + eps)
#
#eps = 5000 
#words = (" ".join(train_qs)).lower().split()
#counts = Counter(words)
#weights = {word: get_weight(count) for word, count in counts.items()}
#
#print('Most common words and weights: \n')
#print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
#print('\nLeast common words and weights: ')
#(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])
#           
#def word_match_share(row):
#    q1words = {}
#    q2words = {}
#    for word in str(row['question1']).lower().split():
#        if word not in stops:
#            q1words[word] = 1
#    for word in str(row['question2']).lower().split():
#        if word not in stops:
#            q2words[word] = 1
#    if len(q1words) == 0 or len(q2words) == 0:
#        # The computer-generated chaff includes a few questions that are nothing but stopwords
#        return 0
#    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
#    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
#    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/((len(q1words) + len(q2words)* 1.0))
#    return R
#
#
#def tfidf_word_match_share(row):
#    q1words = {}
#    q2words = {}
#    for word in str(row['question1']).lower().split():
#        if word not in stops:
#            q1words[word] = 1
#    for word in str(row['question2']).lower().split():
#        if word not in stops:
#            q2words[word] = 1
#    if len(q1words) == 0 or len(q2words) == 0:
#        # The computer-generated chaff includes a few questions that are nothing but stopwords
#        return 0
#    
#    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
#    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
##    print(np.sum(shared_weights))
##    print(np.sum(total_weights))
#    R = (np.sum(shared_weights) * 1.0) / (np.sum(total_weights)* 1.0 )
#    return R
#train_word_match = train_df.apply(word_match_share, axis=1, raw=True)
#tfidf_train_word_match = train_df.apply(tfidf_word_match_share, axis=1, raw=True)
#
#train_df['word_match'] = train_word_match
#train_df['tfidf_word_match'] = tfidf_train_word_match
#test_df['word_match'] = test_df.apply(word_match_share, axis=1, raw=True)
#test_df['tfidf_word_match'] = test_df.apply(tfidf_word_match_share, axis=1, raw=True)

#from sklearn.metrics import roc_auc_score
#print('Original AUC:', roc_auc_score(train_df['is_duplicate'],train_df['word_match']))
#print('   TFIDF AUC:', roc_auc_score(train_df['is_duplicate'], train_df['tfidf_word_match'].fillna(0)))
#  
train_df = train_df.fillna(-1)   
test_df = test_df.fillna(-1)

features_to_use = cols = [col for col in train_df.columns if col not in ['id','qid1','qid2','question1', 'question2','is_duplicate','CVindices','test_id']] 


    
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
num_rounds = 3300
plst = list(param.items())

test_X = test_df[features_to_use]


def train_xgboost(i):
    print('Fold ', i , ' Processing')
    X_build = train_df[train_df['CVindices'] != i]
    X_val   = train_df[train_df['CVindices'] == i]
    
    X_train = X_build[features_to_use]
    X_valid = X_val[features_to_use]
    
    X_trainy = X_build['is_duplicate']
    X_validy = X_val['is_duplicate']
    
    
    xgbbuild = xgb.DMatrix(X_train, label=X_trainy)
    xgbval = xgb.DMatrix(X_valid, label=X_validy)
    watchlist = [ (xgbbuild,'build'), (xgbval, 'valid') ]
                 
    model = xgb.train(plst, 
                      xgbbuild, 
                      num_rounds, 
                      watchlist, 
                      verbose_eval = 100 ,
                      early_stopping_rounds=20
                      )
#    pred_cv = model.predict(xgbval)
#    pred_cv = pd.DataFrame(pred_cv)
#    pred_cv.columns = ["is_duplicate"]
#    pred_cv["id"] = X_val.id.values
#    pred_cv = pred_cv[['id','is_duplicate']]
#    sub_valfile = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.xgb01.fold' + str(i) + '.csv'
#    pred_cv.to_csv(sub_valfile, index=False)
    
    xgtest = xgb.DMatrix(test_X)
    pred_test = model.predict(xgtest)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["is_duplicate"]
    pred_test["test_id"] = test_df.test_id.values
    pred_test = pred_test[['test_id','is_duplicate']]
    pred_test.head(10)
    sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.xgb01.fold' + str(i) + '-test' + '.csv'
    pred_test.to_csv(sub_file, index=False)
    del pred_cv
    del pred_test

fullnum_rounds = int(num_rounds * 1.2)

def fulltrain_xgboost(bags):
    xgbtrain = xgb.DMatrix( train_df[features_to_use], label=train_df['is_duplicate'])
    watchlist = [ (xgbtrain,'train') ]
    fullmodel = xgb.train(plst, 
                              xgbtrain, 
                              fullnum_rounds, 
                              watchlist,
                              verbose_eval = 100,
                              )
    xgtest = xgb.DMatrix(test_X)
    predfull_test = fullmodel.predict(xgtest)
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["is_duplicate"]
    predfull_test["id"] = test_df.id.values
    predfull_test = predfull_test[['id','is_duplicate']]
    sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.xgb01.full' + '.csv'
    predfull_test.to_csv(sub_file, index=False)

folds = 5
i = 1
if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost(i)
    fulltrain_xgboost(folds)
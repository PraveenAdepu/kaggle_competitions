# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 22:15:27 2017

@author: SriPrav
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns


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

inDir = 'C:/Users/SriPrav/Documents/R/23Quora'

train_file = inDir + "/input/TrainingSet01.csv"
test_file = inDir + "/input/TestingSet01.csv"
trainingSet = pd.read_csv(train_file)
testingSet = pd.read_csv(test_file)
print(trainingSet.shape) # (404290, 6)
print(testingSet.shape)  # (2345796, 3)


train_features_06 = inDir + "/input/train_features_06.csv"
test_features_06 = inDir + "/input/test_features_06.csv"
train_features_06 = pd.read_csv(train_features_06)
test_features_06 = pd.read_csv(test_features_06)
print(train_features_06.shape) # (404290, 36)
print(test_features_06.shape)  # (2345796, 34)

trainingSet = pd.merge(trainingSet, train_features_06, how = 'left', on = 'id')
testingSet = pd.merge(testingSet, test_features_06, how = 'left', on = 'test_id')

train_features_07 = inDir + "/input/train_features_07.csv"
test_features_07 = inDir + "/input/test_features_07.csv"
train_features_07 = pd.read_csv(train_features_07)
test_features_07 = pd.read_csv(test_features_07)
print(train_features_07.shape) # (404290, 36)
print(test_features_07.shape)  # (2345796, 34)

trainingSet = pd.merge(trainingSet, train_features_07, how = 'left', on = 'id')
testingSet = pd.merge(testingSet, test_features_07, how = 'left', on = 'test_id')



train_features_11 = inDir + "/input/train_question_freq_features.csv"
test_features_11 = inDir + "/input/test_question_freq_features.csv"
train_features_11 = pd.read_csv(train_features_11)
test_features_11 = pd.read_csv(test_features_11)
print(train_features_11.shape) # (404290, 36)
print(test_features_11.shape)  # (2345796, 34)

del train_features_11['is_duplicate']
test_features_11.rename(columns={'id': 'test_id'}, inplace=True)

trainingSet = pd.merge(trainingSet, train_features_11, how = 'left', on = 'id')
testingSet = pd.merge(testingSet, test_features_11, how = 'left', on = 'test_id')

train_features_30 = inDir + "/input/train_features_35.csv"
test_features_30 = inDir + "/input/test_features_35.csv"
train_features_30 = pd.read_csv(train_features_30)
test_features_30 = pd.read_csv(test_features_30)
print(train_features_30.shape) # (404290, 36)
print(test_features_30.shape)  # (2345796, 34)

trainingSet = pd.merge(trainingSet, train_features_30, how = 'left', on = 'id')
testingSet = pd.merge(testingSet, test_features_30, how = 'left', on = 'test_id')

train_features_31 = inDir + "/input/train_features_31.csv"
test_features_31 = inDir + "/input/test_features_31.csv"
train_features_31 = pd.read_csv(train_features_31)
test_features_31 = pd.read_csv(test_features_31)
print(train_features_31.shape) # (404290, 36)
print(test_features_31.shape)  # (2345796, 34)

trainingSet = pd.merge(trainingSet, train_features_31, how = 'left', on = 'id')
testingSet = pd.merge(testingSet, test_features_31, how = 'left', on = 'test_id')


train_features_32 = inDir + "/input/train_features_32.csv"
test_features_32 = inDir + "/input/test_features_32.csv"
train_features_32 = pd.read_csv(train_features_32)
test_features_32 = pd.read_csv(test_features_32)
print(train_features_32.shape) # (404290, 36)
print(test_features_32.shape)  # (2345796, 34)

trainingSet = pd.merge(trainingSet, train_features_32, how = 'left', on = 'id')
testingSet = pd.merge(testingSet, test_features_32, how = 'left', on = 'test_id')

train_features_33 = inDir + "/input/train_features_33.csv"
test_features_33 = inDir + "/input/test_features_33.csv"
train_features_33 = pd.read_csv(train_features_33)
test_features_33 = pd.read_csv(test_features_33)
print(train_features_33.shape) # (404290, 36)
print(test_features_33.shape)  # (2345796, 34)

trainingSet = pd.merge(trainingSet, train_features_33, how = 'left', on = 'id')
testingSet = pd.merge(testingSet, test_features_33, how = 'left', on = 'test_id')

#train_features_34 = inDir + "/input/train_features_34.csv"
#test_features_34 = inDir + "/input/test_features_34.csv"
#train_features_34 = pd.read_csv(train_features_34)
#test_features_34 = pd.read_csv(test_features_34)
#print(train_features_34.shape) # (404290, 36)
#print(test_features_34.shape)  # (2345796, 34)
#
#trainingSet = pd.merge(trainingSet, train_features_34, how = 'left', on = 'id')
#testingSet = pd.merge(testingSet, test_features_34, how = 'left', on = 'test_id')



train_features_52 = inDir + "/input/train_features_52.csv"
test_features_52 = inDir + "/input/test_features_52.csv"
train_features_52 = pd.read_csv(train_features_52)
test_features_52 = pd.read_csv(test_features_52)
print(train_features_52.shape) # (404290, 36)
print(test_features_52.shape)  # (2345796, 34)

trainingSet = pd.merge(trainingSet, train_features_52, how = 'left', on = 'id')
testingSet = pd.merge(testingSet, test_features_52, how = 'left', on = 'test_id')

train_features_36 = inDir + "/input/train_features_36.csv"
test_features_36 = inDir + "/input/test_features_36.csv"
train_features_36 = pd.read_csv(train_features_36)
test_features_36 = pd.read_csv(test_features_36)
print(train_features_36.shape) # (404290, 36)
print(test_features_36.shape)  # (2345796, 34)

trainingSet = pd.merge(trainingSet, train_features_36, how = 'left', on = 'id')
testingSet = pd.merge(testingSet, test_features_36, how = 'left', on = 'test_id')

train_features_37 = inDir + "/input/train_interaction_features.csv"
test_features_37 = inDir + "/input/test_interaction_features.csv"
train_features_37 = pd.read_csv(train_features_37)
test_features_37 = pd.read_csv(test_features_37)
print(train_features_37.shape) # (404290, 36)
print(test_features_37.shape)  # (2345796, 34)

trainingSet = pd.merge(trainingSet, train_features_37, how = 'left', on = 'id')
testingSet = pd.merge(testingSet, test_features_37, how = 'left', on = 'test_id')


train_features_40 = inDir + "/input/train_features_40.csv"
test_features_40 = inDir + "/input/test_features_40.csv"
train_features_40 = pd.read_csv(train_features_40)
test_features_40 = pd.read_csv(test_features_40)
print(train_features_40.shape) # (404290, 36)
print(test_features_40.shape)  # (2345796, 34)

trainingSet = pd.merge(trainingSet, train_features_40, how = 'left', on = 'id')
testingSet = pd.merge(testingSet, test_features_40, how = 'left', on = 'test_id')


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

cv_file = inDir + "/CVSchema/Prav_CVindices_5folds.csv"
CV_Schema = pd.read_csv(cv_file)

trainingSet = pd.merge(trainingSet, CV_Schema, how = 'left', on = ['id','qid1','qid2'])

from sklearn.metrics import roc_auc_score
#z_noun_match,z_match_ratio,z_word_match,z_tfidf_word_match
#print('Original AUC:', roc_auc_score(trainingSet['is_duplicate'], trainingSet['z_noun_match'].fillna(0)))
#print('   TFIDF AUC:', roc_auc_score(trainingSet['is_duplicate'], trainingSet['z_match_ratio'].fillna(0)))
#print('Original AUC:', roc_auc_score(trainingSet['is_duplicate'], trainingSet['z_word_match'].fillna(0)))
#print('   TFIDF AUC:', roc_auc_score(trainingSet['is_duplicate'], trainingSet['z_tfidf_word_match'].fillna(0)))

#feature_names = cols = [col for col in trainingSet.columns if col not in ['id','qid1','qid2','question1', 'question2','is_duplicate','CVindices','test_id'
#                                                                       'q1_2pairwords_matched_q2_ratio','z_tfidf_word_match', 'q1_hash',
#                                                                       'q2_hash']] 
                                                                       
# train_df$word_q1_2w        <- pmax(train_df$word_match/ train_df$q1_freq, train_df$q1_freq/train_df$word_match)
# train_df$q1_q2_freq_2w     <- pmax(train_df$q2_freq/ train_df$q1_freq, train_df$q1_freq/train_df$q2_freq)
# train_df$word_3gram_2w     <- pmax(train_df$word_match/ train_df$q_3gram_jaccard, train_df$q_3gram_jaccard/train_df$word_match)
# train_df$tfidf_q1_2w       <- pmax(train_df$tfidf_word_match/ train_df$q1_freq, train_df$q1_freq/train_df$tfidf_word_match)
# train_df$qpmin_q2_freq_2w  <- pmax(train_df$q2_freq/ train_df$q_MatchedWords_ratio_to_q_pmin, train_df$q_MatchedWords_ratio_to_q_pmin/train_df$q2_freq)
# train_df$word_q2_2w        <- pmax(train_df$word_match/ train_df$q2_freq, train_df$q2_freq/train_df$word_match)
# train_df$tfidf_q2_2w       <- pmax(train_df$tfidf_word_match/ train_df$q2_freq, train_df$q2_freq/train_df$tfidf_word_match)
# train_df$q1_3gram_2w       <- pmax(train_df$q1_freq/ train_df$q_3gram_jaccard, train_df$q_3gram_jaccard/train_df$q1_freq)
# train_df$qpmin_q1_freq_2w  <- pmax(train_df$q1_freq/ train_df$q_MatchedWords_ratio_to_q_pmin, train_df$q_MatchedWords_ratio_to_q_pmin/train_df$q1_freq)
# train_df$tfidf_4gram_2w    <- pmax(train_df$tfidf_word_match/ train_df$q_4gram_jaccard, train_df$q_4gram_jaccard/train_df$tfidf_word_match)
# 
# 
# test_df$word_q1_2w        <- pmax(test_df$word_match/ test_df$q1_freq, test_df$q1_freq/test_df$word_match)
# test_df$q1_q2_freq_2w     <- pmax(test_df$q2_freq/ test_df$q1_freq, test_df$q1_freq/test_df$q2_freq)
# test_df$word_3gram_2w     <- pmax(test_df$word_match/ test_df$q_3gram_jaccard, test_df$q_3gram_jaccard/test_df$word_match)
# test_df$tfidf_q1_2w       <- pmax(test_df$tfidf_word_match/ test_df$q1_freq, test_df$q1_freq/test_df$tfidf_word_match)
# test_df$qpmin_q2_freq_2w  <- pmax(test_df$q2_freq/ test_df$q_MatchedWords_ratio_to_q_pmin, test_df$q_MatchedWords_ratio_to_q_pmin/test_df$q2_freq)
# test_df$word_q2_2w        <- pmax(test_df$word_match/ test_df$q2_freq, test_df$q2_freq/test_df$word_match)
# test_df$tfidf_q2_2w       <- pmax(test_df$tfidf_word_match/ test_df$q2_freq, test_df$q2_freq/test_df$tfidf_word_match)
# test_df$q1_3gram_2w       <- pmax(test_df$q1_freq/ test_df$q_3gram_jaccard, test_df$q_3gram_jaccard/test_df$q1_freq)
# test_df$qpmin_q1_freq_2w  <- pmax(test_df$q1_freq/ test_df$q_MatchedWords_ratio_to_q_pmin, test_df$q_MatchedWords_ratio_to_q_pmin/test_df$q1_freq)
# test_df$tfidf_4gram_2w    <- pmax(test_df$tfidf_word_match/ test_df$q_4gram_jaccard, test_df$q_4gram_jaccard/test_df$tfidf_word_match)


feature_names = ['q_dist_soundex'
                        ,'q_dist_jarowinkler'
                        ,'q_dist_lcs'
                        ,'q1_nchar'
                        ,'q2_nchar'
                        ,'q1_EndsWith_q2'
                        ,'q1_EndsWith_Sound_q2'
                        ,'q_nchar_ratios_pmax'
                        ,'q_nchar_pmin'
                        ,'q_nchar_pmax'
                        ,'q1_StartsWith_Sound_q2'
                        ,'q1_StartsWith_q2'
                        ,'q1_nwords'
                        ,'q1_nwords_matched_q2'
                        ,'q1_MatchedWords_ratio_to_q2'
                        ,'q2_nwords'
                        ,'q2_nwords_matched_q1'
                        ,'q2_MatchedWords_ratio_to_q1'
                        ,'q_MatchedWords_ratio_to_q_ratios_pmax'
                        ,'q_MatchedWords_ratio_to_q_pmin'
                        ,'q_MatchedWords_ratio_to_q_pmax'
                        ,'q_2gram_jaccard'
                        ,'q_3gram_jaccard'
                        ,'q_4gram_jaccard'
                        ,'q_5gram_jaccard'
                        ,'q_dist_lv'
                        ,'q_dist_cosine'
                        ,'q1_PunctCount'
                        ,'q2_PunctCount'
                        ,'q_PunctCount_ratios_pmax'
                        ,'q_PunctCount_pmin'
                        ,'q_PunctCount_pmax'
                        ,'len_q1'
                        ,'len_q2'
                        ,'diff_len'
                        ,'len_char_q1'
                        ,'len_char_q2'
                        ,'len_word_q1'
                        ,'len_word_q2'
                        ,'common_words'
                        ,'fuzz_qratio'
                        ,'fuzz_WRatio'
                        ,'fuzz_partial_ratio'
                        ,'fuzz_partial_token_set_ratio'
                        ,'fuzz_partial_token_sort_ratio'
                        ,'fuzz_token_set_ratio'
                        ,'fuzz_token_sort_ratio'
                        ,'word_match'
                        ,'tfidf_word_match'
                        ,'cosine_distance'
                        ,'cityblock_distance'
                        ,'jaccard_distance'
                        ,'canberra_distance'
                        ,'euclidean_distance'
                        ,'minkowski_distance'
                        ,'braycurtis_distance'
                        ,'skew_q1vec'
                        ,'skew_q2vec'
                        ,'kur_q1vec'
                        ,'kur_q2vec',
                     'zbigrams_common_count',
                     'zbigrams_common_ratio',
                     'z_noun_match',
                     'z_match_ratio',
                     'z_word_match',                    
                     'q1_freq',
                     'q2_freq',"q1_q2_intersect","q1_q2_wm_ratio" ,"z_place_match_num","z_place_mismatch_num","qid1_max_kcore"
                     ,"qid2_max_kcore","max_kcore","m_q1_q2_tf_svd0_q1","m_q1_q2_tf_svd0_q2", "wmd" ,"norm_wmd",
                     "word_q1_2w",      
                    "q1_q2_freq_2w",   
                    "word_3gram_2w",   
                    "tfidf_q1_2w",     
                    "qpmin_q2_freq_2w",
                    "word_q2_2w",      
                    "tfidf_q2_2w",     
                    "q1_3gram_2w",     
                    "qpmin_q1_freq_2w",
                    "tfidf_4gram_2w" , "q1_pr","q2_pr"
                     ]#"q_unique_words","q_unique_words_excluding_stop","q_words_diff_unique_excluding_stop","q_char_diff_unique_excluding_stop"

                                                                                     
trainingSet = trainingSet.replace(np.inf, np.nan) 
testingSet = testingSet.replace(np.inf, np.nan)
trainingSet = trainingSet.fillna(0)   
testingSet = testingSet.fillna(0)

#trainingSet[feature_names].dtypes
#testingSet[feature_names].dtypes
#
#trainingSet[feature_names] =trainingSet[feature_names].apply(pd.to_numeric)
#testingSet[feature_names] =testingSet[feature_names].apply(pd.to_numeric)

trainingSet[feature_names] =trainingSet[feature_names].astype(np.float64)
testingSet[feature_names] =testingSet[feature_names].astype(np.float64)

##################################################################################################################################
##################################################################################################################################
#for column in features_to_use:
#    print(column)
#    print(' AUC:', roc_auc_score(trainingSet['is_duplicate'], trainingSet[column]))
##################################################################################################################################
##################################################################################################################################


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
param['print_every_n'] = 500
xgb_num_rounds = 5010

xgbParameters = list(param.items())



lgbm_params = {
          'boosting_type': 'gbdt',
          'objective': 'regression',
          'metric': 'binary_logloss',
          'min_child_weight': 10,
          'num_leaves': 2**4, #2**4,
          'lambda_l2': 2,
          'feature_fraction': 0.7,
          'bagging_fraction': 0.8,#0.7
          'bagging_freq': 5,#2
          'learning_rate': 0.02,
          'tree_method': 'exact',          
          'nthread': 25,
          'silent': True,
          'seed': 2017,
         }
lgbm_num_round = 5010
lgbm_early_stopping_rounds = 100

seed = 2017
folds = 5
nbags = 1
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2017

lgbmModelName = 'lgbm105'
xgbModelName  = 'xgb106'
rfModelName   = 'rf101'
etModelName   = 'et101'
fmModelName   = 'fm100'

gc.collect()
if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, xgb_num_rounds)
    fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,xgbModelName,xgbParameters, xgb_num_rounds)
    
    for i in range(1, folds+1):
        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbm_params, lgbm_num_round)
    fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round)

    for i in range(1, folds+1):
        train_regression("rf",trainingSet, testingSet,feature_names,i,nbags,rfModelName,current_seed)
    fulltrain_regression("rf",trainingSet, testingSet,feature_names,nbags,rfModelName,current_seed)
    
    
    for i in range(1, folds+1):
        train_regression("et",trainingSet, testingSet,feature_names,i,nbags,etModelName,current_seed)
    fulltrain_regression("et",trainingSet, testingSet,feature_names,nbags,etModelName,current_seed)


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
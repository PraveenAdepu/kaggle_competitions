
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

train_features_52 = inDir + "/input/train_features_52.csv"
test_features_52 = inDir + "/input/test_features_52.csv"
train_features_52 = pd.read_csv(train_features_52)
test_features_52 = pd.read_csv(test_features_52)
print(train_features_52.shape) # (404290, 36)
print(test_features_52.shape)  # (2345796, 34)

trainingSet = pd.merge(trainingSet, train_features_52, how = 'left', on = 'id')
testingSet = pd.merge(testingSet, test_features_52, how = 'left', on = 'test_id')
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

#z_noun_match,z_match_ratio,z_word_match,z_tfidf_word_match
print('Original AUC:', roc_auc_score(trainingSet['is_duplicate'], trainingSet['z_noun_match'].fillna(0)))
print('   TFIDF AUC:', roc_auc_score(trainingSet['is_duplicate'], trainingSet['z_match_ratio'].fillna(0)))
print('Original AUC:', roc_auc_score(trainingSet['is_duplicate'], trainingSet['z_word_match'].fillna(0)))
print('   TFIDF AUC:', roc_auc_score(trainingSet['is_duplicate'], trainingSet['z_tfidf_word_match'].fillna(0)))

#feature_names = cols = [col for col in trainingSet.columns if col not in ['id','qid1','qid2','question1', 'question2','is_duplicate','CVindices','test_id'
#                                                                       'q1_2pairwords_matched_q2_ratio','z_tfidf_word_match', 'q1_hash',
#                                                                       'q2_hash']] 
                                                                       


features_to_use  = ['q_dist_soundex'
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
                     'q2_freq', "wmd" ,"norm_wmd"] 

trainingSet = trainingSet.replace(np.inf, np.nan) 
testingSet = testingSet.replace(np.inf, np.nan)
trainingSet = trainingSet.fillna(0)   
testingSet = testingSet.fillna(0)


Normalize_features = ['q_dist_soundex'
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
                     'q2_freq', "wmd" ,"norm_wmd"]
                     
train_Normalize = trainingSet[Normalize_features]
test_Normalize = testingSet[Normalize_features]

# Scale train_X and test_X together
traintest_Normalize = np.vstack((train_Normalize, test_Normalize))
print(traintest_Normalize.shape)
traintest_Normalize = preprocessing.StandardScaler().fit_transform(traintest_Normalize)

train_Normalize_complete = traintest_Normalize[range(train_Normalize.shape[0])]
test_Normalize_complete = traintest_Normalize[range(train_Normalize.shape[0], traintest_Normalize.shape[0])]
print(train_Normalize_complete.shape)
print(test_Normalize_complete.shape) 

train_Normalize_complete = pd.DataFrame(train_Normalize_complete, columns = Normalize_features)
test_Normalize_complete = pd.DataFrame(test_Normalize_complete, columns = Normalize_features)

train_Normalize.head() 
train_Normalize_complete.head()

for col in Normalize_features:
    del trainingSet[col]

for col in Normalize_features:
    del testingSet[col]
       
trainingSet = pd.concat([trainingSet,train_Normalize_complete], axis=1)
testingSet = pd.concat([testingSet,test_Normalize_complete], axis=1)

del train_Normalize
del test_Normalize
del traintest_Normalize
del train_Normalize_complete
del test_Normalize_complete

gc.collect()

#trainfeatures_to_use = cols = [col for col in trainingSet.columns if col not in ['qid1','qid2','question1', 'question2',
#                                                                       'q1_2pairwords_matched_q2_ratio','z_tfidf_word_match']] 

##################################################################################################################################
##################################################################################################################################
#for column in features_to_use:
#    print(column)
#    print(' AUC:', roc_auc_score(trainingSet['is_duplicate'], trainingSet[column]))
##################################################################################################################################
##################################################################################################################################
features_to_use = ['q_dist_soundex'
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
                     'q2_freq', "wmd" ,"norm_wmd"]



import numpy as np
np.random.seed(123)
import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import MinMaxScaler # Extra
from keras.callbacks import Callback, ModelCheckpoint
MODEL_WEIGHTS_FILE = 'nn51_question_pairs_weights.h5'


## Scale train_X and test_X together
#traintest = np.vstack((train_X, test_X))
#print(traintest.shape)
#traintest = preprocessing.StandardScaler().fit_transform(traintest)
#
#train_X = traintest[range(train_X.shape[0])]
#test_X = traintest[range(train_X.shape[0], traintest.shape[0])]
#print(train_X.shape)
#print(test_X.shape)  

def nn_model(size):
    model = Sequential()
    
    model.add(Dense(65, input_dim = size, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(35, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Dense(15, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.1))
    model.add(Dense(1, init = 'he_normal', activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics=['accuracy'])
    return(model)
    
callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=1)]
#callbacks = [
#            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
#           ]
        
nbags = 1
folds = 5
epochs = 4
batchsize = 128
verboselog = 2

x_test = testingSet[features_to_use].apply(pd.to_numeric)

def nnet(i):
    print('Fold ', i , ' Processing')
    X_build = trainingSet[trainingSet['CVindices'] != i] # 636112
    #X_build = X_build.groupby('id').first().reset_index() # 331085
    X_val   = trainingSet[trainingSet['CVindices'] == i]
    
    print(X_build.shape) # (404290, 6)
    print(X_val.shape)  # (2345796, 3)

    X_train = X_build[features_to_use]
    X_valid = X_val[features_to_use]
    
    X_train = X_train.apply(pd.to_numeric)
    X_valid = X_valid.apply(pd.to_numeric)

    X_trainy = X_build['is_duplicate']
    X_validy = X_val['is_duplicate']
    
    X_trainy = X_trainy.apply(pd.to_numeric).values
    X_validy = X_validy.apply(pd.to_numeric).values
    
    pred_cv = np.zeros(X_validy.shape[0])
    pred_test = np.zeros(x_test.shape[0])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros(X_validy.shape[0])
        model = nn_model(X_train.shape[1])
        
        model.fit(X_train.values, X_trainy, nb_epoch=epochs, batch_size=batchsize,  verbose=verboselog,validation_data=(X_valid.values,X_validy))#,callbacks=callbacks
        #model.load_weights('C:/Users/SriPrav/Documents/R/23Quora/nn50_question_pairs_weights.h5')
        bag_cv   = model.predict(X_valid.values)[:,0]
        pred_cv += model.predict(X_valid.values)[:,0]
        pred_test += model.predict(x_test.values)[:,0]
        bag_score = log_loss(X_validy, bag_cv)
        print('bag ', j, '- logloss:', bag_score)
#        os.remove('C:/Users/SriPrav/Documents/R/23Quora/nn50_question_pairs_weights.h5')
    pred_cv /= nbags
    pred_test/= nbags
    fold_score = log_loss(X_validy, pred_cv)
    print('Fold ', i, '- logloss:', fold_score)
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = ["is_duplicate"]
    pred_cv["id"] = X_val.id.values
    pred_cv = pred_cv[['id','is_duplicate']]
    sub_valfile = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.nn51.fold' + str(i) + '.csv'    
    pred_cv.to_csv(sub_valfile, index=False)
    
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["is_duplicate"]
    pred_test["test_id"] = testingSet.test_id.values
    pred_test = pred_test[['test_id','is_duplicate']]
    sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.nn51.fold' + str(i) + '-test' + '.csv'
    pred_test.to_csv(sub_file, index=False)
    del pred_cv
    del pred_test
    

##########################################################################################
# Full model training
########################################################################################## 
best_epoch = 4
full_epochs = int(best_epoch * 1.5) # Get Best epoch
#
def full_train_nn(i):
    X_train = trainingSet[features_to_use]
    X_train = X_train.apply(pd.to_numeric)
    X_trainy = trainingSet['is_duplicate'] 
    X_trainy = X_trainy.apply(pd.to_numeric).values
    
    model = nn_model(X_train.shape[1])
    
#    callbacks = [
#            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
#        ]
#    early_stop = EarlyStopping(monitor='val_loss', patience=75, verbose=0)
#    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)
    model.fit(X_train.values, X_trainy, nb_epoch=full_epochs, batch_size=batchsize,  verbose=verboselog)
    #model.load_weights('C:/Users/SriPrav/Documents/R/23Quora/nn1_question_pairs_weights.h5')
    pred_test = model.predict(x_test.values)[:,0]
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["is_duplicate"]
    pred_test["test_id"] = testingSet.test_id.values
    pred_test = pred_test[['test_id','is_duplicate']]
    sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.nn51.full' + '.csv'
    pred_test.to_csv(sub_file, index=False)   

    del pred_test
    
folds = 5
nbags = 1

if __name__ == '__main__':
    for i in range(1, folds+1):
        nnet(i)
    full_train_nn(nbags)
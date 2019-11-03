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

#cv_file = inDir + "/CVSchema/Prav_CVindices_5folds.csv"
#CV_Schema = pd.read_csv(cv_file)
features_to_use = cols = [col for col in train_df.columns if col not in ['id','qid1','qid2','question1', 'question2','is_duplicate','CVindices','test_id']] 
train_df = train_df.fillna(0)   
test_df = test_df.fillna(0)

x_train = pd.DataFrame()
x_test = pd.DataFrame()
x_train = train_df[features_to_use]
x_test  = test_df[features_to_use]

y_train = train_df['is_duplicate'].values

pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]

# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = ((len(pos_train)*1.0 / (len(pos_train) + len(neg_train))*1.0) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train)*1.0 / (len(pos_train) + len(neg_train))*1.0)

x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train

from sklearn.cross_validation import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

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
num_rounds = 3300
plst = list(param.items())


d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

#bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
model = xgb.train(plst, 
                      d_train, 
                      num_rounds, 
                      watchlist, 
                      verbose_eval = 100 ,
                      early_stopping_rounds=20
                      )
 
d_test = xgb.DMatrix(x_test)
p_test = model.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = test_df['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('./submissions/Prav_xgb02.csv', index=False)
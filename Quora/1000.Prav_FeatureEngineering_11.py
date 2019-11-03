
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

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train_df.shape) # (404290, 6)
print(test_df.shape)  # (2345796, 3)

train_features_50 = inDir + "/input/train_features_50.csv"
test_features_50 = inDir + "/input/test_features_50.csv"
train_features_50 = pd.read_csv(train_features_50)
test_features_50 = pd.read_csv(test_features_50)
print(train_features_50.shape) # (404290, 36)
print(test_features_50.shape)  # (2345796, 34)
features_50 = ['wmd','norm_wmd']

train_features_50 = train_features_50[features_50]
test_features_50 = test_features_50[features_50]

train_df = pd.concat([train_df, train_features_50], axis = 1)
test_df  = pd.concat([test_df, test_features_50], axis = 1)

trainfeatures = ['id','wmd','norm_wmd']
testfeatures = ['test_id','wmd','norm_wmd']

sub_valfile = 'C:/Users/SriPrav/Documents/R/23Quora/input/train_features_52.csv'    
train_df[trainfeatures].to_csv(sub_valfile, index=False)

sub_valfile = 'C:/Users/SriPrav/Documents/R/23Quora/input/test_features_52.csv'    
test_df[testfeatures].to_csv(sub_valfile, index=False)


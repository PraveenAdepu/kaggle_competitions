# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:30:22 2017

@author: PAdepu
"""

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

stops = set(stopwords.words("english"))

inDir = 'C:/Users/padepu/Documents/R/23Quora'

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train_df.shape) # (404290, 6)
print(test_df.shape)  # (2345796, 3)




train_qs = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist()).astype(str)
test_qs = pd.Series(test_df['question1'].tolist() + test_df['question2'].tolist()).astype(str)

dist_train = train_qs.apply(lambda x: len(x.split(' ')))
dist_test = test_qs.apply(lambda x: len(x.split(' ')))

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

print('Most common words and weights: \n')
print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
print('\nLeast common words and weights: ')
(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])
           
def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/((len(q1words) + len(q2words)))
    return R


def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = (np.sum(shared_weights)) / (np.sum(total_weights) )
    return R
train_word_match = train_df.apply(word_match_share, axis=1, raw=True)
tfidf_train_word_match = train_df.apply(tfidf_word_match_share, axis=1, raw=True)


train_df['word_match'] = train_word_match
train_df['tfidf_word_match'] = tfidf_train_word_match
test_df['word_match'] = test_df.apply(word_match_share, axis=1, raw=True)
test_df['tfidf_word_match'] = test_df.apply(tfidf_word_match_share, axis=1, raw=True)

from sklearn.metrics import roc_auc_score
print('Original AUC:', roc_auc_score(train_df['is_duplicate'], train_word_match))
print('   TFIDF AUC:', roc_auc_score(train_df['is_duplicate'], tfidf_train_word_match.fillna(0)))
print('Original AUC:', roc_auc_score(test_df['is_duplicate'], test_df['word_match']))
print('   TFIDF AUC:', roc_auc_score(test_df['is_duplicate'], test_df['tfidf_word_match'].fillna(0)))
 


features_to_use = cols = [col for col in train_df.columns if col not in ['qid1','qid2','question1', 'question2','is_duplicate','CVindices','test_id']] 
sub_file = 'C:/Users/padepu/Documents/R/23Quora/input/train_features_03.csv'
train_df[features_to_use].to_csv(sub_file, index=False)

features_to_use = cols = [col for col in test_df.columns if col not in ['qid1','qid2','question1', 'question2','is_duplicate','CVindices']] 
sub_file = 'C:/Users/padepu/Documents/R/23Quora/input/test_features_03.csv'
test_df[features_to_use].to_csv(sub_file, index=False)

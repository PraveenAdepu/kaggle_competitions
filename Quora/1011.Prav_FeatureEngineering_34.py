# -*- coding: utf-8 -*-
"""
Created on Mon May 22 08:17:39 2017

@author: PAdepu
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
import networkx as nx
from collections import defaultdict

stops = set(stopwords.words("english"))

inDir = 'C:/Users/SriPrav/Documents/R/23Quora'

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train_df.shape) # (404290, 6)
print(test_df.shape)  # (2345796, 3)


ques = pd.concat([train_df[['question1', 'question2']], \
        test_df[['question1', 'question2']]], axis=0).reset_index(drop='index')

ques.shape

stops = set(stopwords.words("english"))
def word_match_share(q1, q2, stops=None):
    q1 = str(q1).lower().split()
    q2 = str(q2).lower().split()
    q1words = {}
    q2words = {}
    for word in q1:
        if word not in stops:
            q1words[word] = 1
    for word in q2:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0.
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R
	
q_dict = defaultdict(dict)
for i in range(ques.shape[0]):
        wm = word_match_share(ques.question1[i], ques.question2[i], stops=stops)
        q_dict[ques.question1[i]][ques.question2[i]] = wm
        q_dict[ques.question2[i]][ques.question1[i]] = wm

def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))
def q1_q2_wm_ratio(row):
    q1 = q_dict[row['question1']]
    q2 = q_dict[row['question2']]
    inter_keys = set(q1.keys()).intersection(set(q2.keys()))
    if(len(inter_keys) == 0): return 0.
    inter_wm = 0.
    total_wm = 0.
    for q,wm in q1.items():
        if q in inter_keys:
            inter_wm += wm
        total_wm += wm
    for q,wm in q2.items():
        if q in inter_keys:
            inter_wm += wm
        total_wm += wm
    if(total_wm == 0.): return 0.
    return inter_wm/total_wm

train_df['q1_q2_wm_ratio'] = train_df.apply(q1_q2_wm_ratio, axis=1, raw=True)
test_df['q1_q2_wm_ratio'] = test_df.apply(q1_q2_intersect, axis=1, raw=True)

train_df['q1_q2_intersect'] = train_df.apply(q1_q2_intersect, axis=1, raw=True)
test_df['q1_q2_intersect'] = test_df.apply(q1_q2_intersect, axis=1, raw=True)

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
temp = train_df.q1_q2_intersect.value_counts()
sns.barplot(temp.index[:20], temp.values[:20])
plt.subplot(1,2,2)
train_df['q1_q2_wm_ratio'].plot.hist()

train_df.plot.scatter(x='q1_q2_intersect', y='q1_q2_wm_ratio', figsize=(12,6))
print(train_df[['q1_q2_intersect', 'q1_q2_wm_ratio']].corr())

train_feat = train_df[['id', 'q1_q2_wm_ratio']]
test_feat = test_df[['test_id', 'q1_q2_wm_ratio']]

train_feat.head(15)
sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/train_features_34.csv' 
train_feat.to_csv(sub_file, index=False)  

sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/test_features_34.csv' 
test_feat.to_csv(sub_file, index=False)  

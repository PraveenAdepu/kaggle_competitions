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

inDir = 'C:/Users/padepu/Documents/R/23Quora'

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train_df.shape) # (404290, 6)
print(test_df.shape)  # (2345796, 3)


ques = pd.concat([train_df[['question1', 'question2']], \
        test_df[['question1', 'question2']]], axis=0).reset_index(drop='index')

ques.shape

q_dict = defaultdict(set)
for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

train_df['q1_q2_intersect'] = train_df.apply(q1_q2_intersect, axis=1, raw=True)
test_df['q1_q2_intersect'] = test_df.apply(q1_q2_intersect, axis=1, raw=True)

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


temp = train_df.q1_q2_intersect.value_counts()
sns.barplot(temp.index[:20], temp.values[:20])

train_feat = train_df[['id','q1_q2_intersect']]
test_feat = test_df[['test_id','q1_q2_intersect']]

train_feat.head(15)
sub_file = 'C:/Users/padepu/Documents/R/23Quora/input/train_feature_30.csv' 
train_feat.to_csv(sub_file, index=False)  

sub_file = 'C:/Users/padepu/Documents/R/23Quora/input/test_feature_30.csv' 
test_feat.to_csv(sub_file, index=False)  

#g = nx.Graph()
#g.add_nodes_from(df.question1)
#g.add_nodes_from(df.question2)
#edges = list(df[['question1', 'question2']].to_records(index=False))
#g.add_edges_from(edges)
#
#
#def get_intersection_count(row):
#    return(len(set(g.neighbors(row.question1)).intersection(set(g.neighbors(row.question2)))))
#
#train_ic = pd.DataFrame()
#test_ic = pd.DataFrame()
#
#
#train_df['intersection_count'] = train_df.apply(lambda row: get_intersection_count(row), axis=1)
#test_df['intersection_count'] = test_df.apply(lambda row: get_intersection_count(row), axis=1)
#train_ic['intersection_count'] = train_df['intersection_count']
#test_ic['intersection_count'] = test_df['intersection_count']
#
#train_ic.to_csv("train_ic.csv", index=False)
#test_ic.to_csv("test_ic.csv", index=False)


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

locations = inDir + "/input/cities.csv"
locations = pd.read_csv(locations)
# There's lots of room to add more locations, but start with just countries
countries = set(locations['Country'].dropna(inplace=False).values.tolist())
all_places = countries
# Turn it into a Regex
regex = "|".join(sorted(set(all_places)))


from tqdm import tqdm
import re
from subprocess import check_output

results = []
print("processing:", train_df.shape)
for index, row in tqdm(train_df.iterrows()):
    q1 = str(row['question1'])
    q2 = str(row['question2'])

    rr = {}

    q1_matches = []
    q2_matches = []

    if (len(q1) > 0):
        q1_matches = [i.lower() for i in re.findall(regex, q1, flags=re.IGNORECASE)]

    if (len(q2) > 0):
        q2_matches = [i.lower() for i in re.findall(regex, q2, flags=re.IGNORECASE)]

    rr['z_q1_place_num'] = len(q1_matches)
    rr['z_q1_has_place'] =len(q1_matches) > 0

    rr['z_q2_place_num'] = len(q2_matches) 
    rr['z_q2_has_place'] = len(q2_matches) > 0

    rr['z_place_match_num'] = len(set(q1_matches).intersection(set(q2_matches)))
    rr['z_place_match'] = rr['z_place_match_num'] > 0

    rr['z_place_mismatch_num'] = len(set(q1_matches).difference(set(q2_matches)))
    rr['z_place_mismatch'] = rr['z_place_mismatch_num'] > 0

    results.append(rr) 
    
out_df = pd.DataFrame.from_dict(results)

out_df['id'] = train_df['id'] 

sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/train_features_31.csv' 
out_df.to_csv(sub_file, index=False)



results = []
print("processing:", test_df.shape)
for index, row in tqdm(test_df.iterrows()):
    q1 = str(row['question1'])
    q2 = str(row['question2'])

    rr = {}

    q1_matches = []
    q2_matches = []

    if (len(q1) > 0):
        q1_matches = [i.lower() for i in re.findall(regex, q1, flags=re.IGNORECASE)]

    if (len(q2) > 0):
        q2_matches = [i.lower() for i in re.findall(regex, q2, flags=re.IGNORECASE)]

    rr['z_q1_place_num'] = len(q1_matches)
    rr['z_q1_has_place'] =len(q1_matches) > 0

    rr['z_q2_place_num'] = len(q2_matches) 
    rr['z_q2_has_place'] = len(q2_matches) > 0

    rr['z_place_match_num'] = len(set(q1_matches).intersection(set(q2_matches)))
    rr['z_place_match'] = rr['z_place_match_num'] > 0

    rr['z_place_mismatch_num'] = len(set(q1_matches).difference(set(q2_matches)))
    rr['z_place_mismatch'] = rr['z_place_mismatch_num'] > 0

    results.append(rr) 
    
out_df1 = pd.DataFrame.from_dict(results)

out_df1['test_id'] = test_df['test_id'] 

sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/test_features_31.csv' 
out_df1.to_csv(sub_file, index=False)




import argparse
import functools
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split

from xgboost import XGBClassifier

stops = set(stopwords.words("english"))

inDir = 'C:/Users/padepu/Documents/R/23Quora'

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
train_df = pd.read_csv(train_file)
train_df = train_df.fillna(' ')
test_df = pd.read_csv(test_file)
test_df = test_df.fillna(' ')

print(train_df.shape) # (404290, 6)
print(test_df.shape)  # (2345796, 3)

def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))

def total_unq_words_stop(row, stops):
    return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])

def wc_diff_unique_stop(row, stops):
    return abs(len([x for x in set(row['question1']) if x not in stops]) - len([x for x in set(row['question2']) if x not in stops]))

def char_diff_unique_stop(row, stops):
    return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(''.join([x for x in set(row['question2']) if x not in stops])))


train_df['q_unique_words'] = train_df.apply(total_unique_words, axis=1, raw=True)
f = functools.partial(total_unq_words_stop, stops=stops)
train_df['q_unique_words_excluding_stop'] = train_df.apply(f, axis=1, raw=True)
f = functools.partial(wc_diff_unique_stop, stops=stops)
train_df['q_words_diff_unique_excluding_stop'] = train_df.apply(f, axis=1, raw=True)
f = functools.partial(char_diff_unique_stop, stops=stops)
train_df['q_char_diff_unique_excluding_stop'] = train_df.apply(f, axis=1, raw=True)

test_df['q_unique_words'] = test_df.apply(total_unique_words, axis=1, raw=True)
f = functools.partial(total_unq_words_stop, stops=stops)
test_df['q_unique_words_excluding_stop'] = test_df.apply(f, axis=1, raw=True)
f = functools.partial(wc_diff_unique_stop, stops=stops)
test_df['q_words_diff_unique_excluding_stop'] = test_df.apply(f, axis=1, raw=True)
f = functools.partial(char_diff_unique_stop, stops=stops)
test_df['q_char_diff_unique_excluding_stop'] = test_df.apply(f, axis=1, raw=True)

train_feat = train_df[['id','q_unique_words','q_unique_words_excluding_stop','q_words_diff_unique_excluding_stop','q_char_diff_unique_excluding_stop']]
test_feat = test_df[['test_id','q_unique_words','q_unique_words_excluding_stop','q_words_diff_unique_excluding_stop','q_char_diff_unique_excluding_stop']]

train_feat.head(15)
sub_file = 'C:/Users/padepu/Documents/R/23Quora/input/train_features_36.csv' 
train_feat.to_csv(sub_file, index=False)  

sub_file = 'C:/Users/padepu/Documents/R/23Quora/input/test_features_36.csv' 
test_feat.to_csv(sub_file, index=False)  

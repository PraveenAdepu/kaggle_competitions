# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:07:18 2017

@author: SriPrav
"""

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
from sklearn.ensemble import ExtraTreesRegressor


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

train_features_04 = inDir + "/input/train_features_04.csv"
test_features_04 = inDir + "/input/test_features_04.csv"
train_features_04 = pd.read_csv(train_features_04)
test_features_04 = pd.read_csv(test_features_04)
print(train_features_04.shape) # (404290, 36)
print(test_features_04.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_04, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_04, how = 'left', on = 'test_id')

train_features_06 = inDir + "/input/train_features_06.csv"
test_features_06 = inDir + "/input/test_features_06.csv"
train_features_06 = pd.read_csv(train_features_06)
test_features_06 = pd.read_csv(test_features_06)
print(train_features_06.shape) # (404290, 36)
print(test_features_06.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_06, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_06, how = 'left', on = 'test_id')


from sklearn.metrics import roc_auc_score, log_loss
print('Original AUC:', roc_auc_score(train_df['is_duplicate'], train_df['zbigrams_common_count'].fillna(0)))
print('   TFIDF AUC:', roc_auc_score(train_df['is_duplicate'], train_df['zbigrams_common_ratio'].fillna(0)))
#cv_file = inDir + "/CVSchema/Prav_CVindices_5folds.csv"
#CV_Schema = pd.read_csv(cv_file)
features_to_use = cols = [col for col in train_df.columns if col not in ['id','qid1','qid2','question1', 'question2','is_duplicate','CVindices','test_id'
                                                                       'q1_2pairwords_matched_q2_ratio']] 

train_df = train_df.replace(np.inf, np.nan) 
test_df = test_df.replace(np.inf, np.nan)
train_df = train_df.fillna(0)   
test_df = test_df.fillna(0)

##################################################################################################################################
##################################################################################################################################
for column in features_to_use:
    print(column)
    print(' AUC:', roc_auc_score(train_df['is_duplicate'], train_df[column]))
##################################################################################################################################
##################################################################################################################################
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
x_build, x_valid, y_build, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)


etr = ExtraTreesRegressor(random_state=2017,bootstrap=False,oob_score=False,
                          n_jobs=20,
                          verbose=1,
                          max_features=63,
                          min_samples_split=2,
                          n_estimators=10000,
                          max_depth=6,
                          min_samples_leaf=1)

etr.fit(x_build,y_build)
#print('OOB score: {:6f}'.format(etr.oob_score_))

error = log_loss(y_valid, etr.predict(x_valid))
print('log_lss: {:.6f}'.format(error))


#################################################################################################################################
#################################################################################################################################

##########################################################################################
# Full model training
########################################################################################## 

#fullnum_rounds = int(num_rounds * 1.2)

etr = ExtraTreesRegressor(random_state=2017,bootstrap=False,oob_score=False,
                          n_jobs=20,
                          verbose=1,
                          max_features=63,
                          min_samples_split=2,
                          n_estimators=12000,
                          max_depth=6,
                          min_samples_leaf=1)

def fulltrain_xgboost(bags):
    etr.fit(x_train,y_train)  
    predfull_test = etr.predict(x_test)
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["is_duplicate"]
    predfull_test["test_id"] = test_df.test_id.values
    predfull_test = predfull_test[['test_id','is_duplicate']]
    sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.et01.full' + '.csv'
    predfull_test.to_csv(sub_file, index=False)
    
folds = 5
if __name__ == '__main__':
    #for i in range(1, folds+1):
        #train_xgboost(i)
    fulltrain_xgboost(folds)
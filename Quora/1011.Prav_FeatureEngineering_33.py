import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
import itertools as it
import pickle
import glob
import os
import string
from scipy import sparse
import nltk
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, make_scorer
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import minimize
import xgboost as xgb

inDir = 'C:/Users/SriPrav/Documents/R/23Quora'

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)
print(df_train.shape) # (404290, 6)
print(df_test.shape)  # (2345796, 3)

df_train['test_id'] = -1

df_test['id'] = -1
df_test['qid1'] = -1
df_test['qid2'] = -1
df_test['is_duplicate'] = -1

df = pd.concat([df_train, df_test])
df['question1'] = df['question1'].fillna('')
df['question2'] = df['question2'].fillna('')
df['uid'] = np.arange(df.shape[0])
df = df.set_index(['uid'])
print(df.dtypes)
del(df_train, df_test)

ix_train = np.where(df['id'] >= 0)[0]
ix_test = np.where(df['id'] == -1)[0]
ix_is_dup = np.where(df['is_duplicate'] == 1)[0]
ix_not_dup = np.where(df['is_duplicate'] == 0)[0]


cv_char = CountVectorizer(ngram_range=(1, 3), analyzer='char')
ch_freq = np.array(cv_char.fit_transform(df['question1'].tolist() + df['question2'].tolist()).sum(axis=0))[0, :]

m_q1 = cv_char.transform(df['question1'].values)
m_q2 = cv_char.transform(df['question2'].values)

tft = TfidfTransformer(
    norm='l2', 
    use_idf=False, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])



#...................................
####   m_q1_q2_tf_svd0
#...................................

svd = TruncatedSVD(n_components=100)
m_svd = svd.fit_transform(sparse.csc_matrix(sparse.vstack((m_q1_tf, m_q2_tf))))    


m_svd_0 =  m_svd[:, 0]
m_svd_0_q1 = m_svd_0[:2750086]
m_svd_0_q2 = m_svd_0[2750086:]

df['m_q1_q2_tf_svd0_q1'] = m_svd_0_q1
df['m_q1_q2_tf_svd0_q2'] = m_svd_0_q2

train = df[df['is_duplicate'] != -1]
test  = df[df['is_duplicate'] == -1]


train_feat = train[['id','m_q1_q2_tf_svd0_q1','m_q1_q2_tf_svd0_q2']]
test_feat = test[['test_id','m_q1_q2_tf_svd0_q1','m_q1_q2_tf_svd0_q2']]

train_feat.head(15)
sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/train_feature_33.csv' 
train_feat.to_csv(sub_file, index=False)  

sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/test_feature_33.csv' 
test_feat.to_csv(sub_file, index=False)  



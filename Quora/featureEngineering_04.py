# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 21:33:50 2017

@author: PAdepu
"""

"""
Detecting duplicate quora questions
feature engineering
@author: Abhishek Thakur
"""

import cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
stop_words = stopwords.words('english')




inDir = 'C:/Users/padepu/Documents/R/23Quora'

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train_df.shape) # (404290, 6)
print(test_df.shape)  # (2345796, 3)

def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())




model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/padepu/Documents/R/23Quora/preModels/GoogleNews-vectors-negative300.bin.gz', binary=True)
#train_df['wmd'] = train_df.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
#
#norm_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
#norm_model.init_sims(replace=True)
#train_df['norm_wmd'] = train_df.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

question1_vectors = np.zeros((train_df.shape[0], 300))
error_count = 0

for i, q in tqdm(enumerate(train_df.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((train_df.shape[0], 300))
for i, q in tqdm(enumerate(train_df.question2.values)):
    question2_vectors[i, :] = sent2vec(q)

train_df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
train_df['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
train_df['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
train_df['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

cPickle.dump(question1_vectors, open('C:/Users/padepu/Documents/R/23Quora/preModels/train_q1_w2v.pkl', 'wb'), -1)
cPickle.dump(question2_vectors, open('C:/Users/padepu/Documents/R/23Quora/preModels/train_q2_w2v.pkl', 'wb'), -1)


features_to_use = cols = [col for col in train_df.columns if col not in ['qid1','qid2','question1', 'question2','is_duplicate','CVindices','test_id']] 

sub_file = 'C:/Users/padepu/Documents/R/23Quora/input/train_features_04.csv'
train_df[features_to_use].to_csv(sub_file, index=False)
############################################################################################################################


test_question1_vectors = np.zeros((test_df.shape[0], 300))
error_count = 0

for i, q in tqdm(enumerate(test_df.question1.values)):
    test_question1_vectors[i, :] = sent2vec(q)

test_question2_vectors  = np.zeros((test_df.shape[0], 300))
for i, q in tqdm(enumerate(test_df.question2.values)):
    test_question2_vectors[i, :] = sent2vec(q)

test_df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                          np.nan_to_num(test_question2_vectors))]

test_df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                          np.nan_to_num(test_question2_vectors))]

test_df['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                          np.nan_to_num(test_question2_vectors))]

test_df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                          np.nan_to_num(test_question2_vectors))]

test_df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                          np.nan_to_num(test_question2_vectors))]

test_df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                          np.nan_to_num(test_question2_vectors))]

test_df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(test_question1_vectors),
                                                          np.nan_to_num(test_question2_vectors))]

test_df['skew_q1vec'] = [skew(x) for x in np.nan_to_num(test_question1_vectors)]
test_df['skew_q2vec'] = [skew(x) for x in np.nan_to_num(test_question2_vectors)]
test_df['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(test_question1_vectors)]
test_df['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(test_question2_vectors)]

cPickle.dump(test_question1_vectors, open('C:/Users/padepu/Documents/R/23Quora/preModels/test_q1_w2v.pkl', 'wb'), -1)
cPickle.dump(test_question2_vectors, open('C:/Users/padepu/Documents/R/23Quora/preModels/test_q2_w2v.pkl', 'wb'), -1)


features_to_use = cols = [col for col in test_df.columns if col not in ['qid1','qid2','question1', 'question2','is_duplicate','CVindices']] 

sub_file = 'C:/Users/padepu/Documents/R/23Quora/input/test_features_04.csv'
test_df[features_to_use].to_csv(sub_file, index=False)

test_df[features_to_use].head()

from sklearn.metrics import roc_auc_score
print('Original AUC:', roc_auc_score(train_df['is_duplicate'], train_df['cosine_distance'].fillna(0)))
print('   TFIDF AUC:', roc_auc_score(train_df['is_duplicate'], train_df['cityblock_distance'].fillna(0)))
print('Original AUC:', roc_auc_score(train_df['is_duplicate'], train_df['jaccard_distance'].fillna(0)))
print('Original AUC:', roc_auc_score(train_df['is_duplicate'], train_df['canberra_distance'].fillna(0)))
print('Original AUC:', roc_auc_score(train_df['is_duplicate'], train_df['euclidean_distance'].fillna(0)))
print('Original AUC:', roc_auc_score(train_df['is_duplicate'], train_df['minkowski_distance'].fillna(0)))
print('Original AUC:', roc_auc_score(train_df['is_duplicate'], train_df['braycurtis_distance'].fillna(0)))
print('Original AUC:', roc_auc_score(train_df['is_duplicate'], train_df['skew_q1vec'].fillna(0)))
print('Original AUC:', roc_auc_score(train_df['is_duplicate'], train_df['skew_q2vec'].fillna(0)))
print('Original AUC:', roc_auc_score(train_df['is_duplicate'], train_df['kur_q1vec'].fillna(0)))
print('Original AUC:', roc_auc_score(train_df['is_duplicate'], train_df['kur_q2vec'].fillna(0)))


print('Original AUC:', roc_auc_score(test_df['is_duplicate'], test_df['word_match']))
print('   TFIDF AUC:', roc_auc_score(test_df['is_duplicate'], test_df['tfidf_word_match'].fillna(0)))
 
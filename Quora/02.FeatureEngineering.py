# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 12:05:37 2017

@author: SriPrav
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

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    #stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    #stop_words = stopwords.words('english')
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

inDir = 'C:/Users/SriPrav/Documents/R/23Quora'

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train_df.shape) # (404290, 6)
print(test_df.shape)  # (2345796, 3)


train_df['len_q1'] = train_df.question1.apply(lambda x: len(str(x)))
train_df['len_q2'] = train_df.question2.apply(lambda x: len(str(x)))
train_df['diff_len'] = train_df.len_q1 - train_df.len_q2
train_df['len_char_q1'] = train_df.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
train_df['len_char_q2'] = train_df.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
train_df['len_word_q1'] = train_df.question1.apply(lambda x: len(str(x).split()))
train_df['len_word_q2'] = train_df.question2.apply(lambda x: len(str(x).split()))
train_df['common_words'] = train_df.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
train_df['fuzz_qratio'] = train_df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
train_df['fuzz_WRatio'] = train_df.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
train_df['fuzz_partial_ratio'] = train_df.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
train_df['fuzz_partial_token_set_ratio'] = train_df.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
train_df['fuzz_partial_token_sort_ratio'] = train_df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
train_df['fuzz_token_set_ratio'] = train_df.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
train_df['fuzz_token_sort_ratio'] = train_df.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


test_df['len_q1'] = test_df.question1.apply(lambda x: len(str(x)))
test_df['len_q2'] = test_df.question2.apply(lambda x: len(str(x)))
test_df['diff_len'] = test_df.len_q1 - test_df.len_q2
test_df['len_char_q1'] = test_df.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
test_df['len_char_q2'] = test_df.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
test_df['len_word_q1'] = test_df.question1.apply(lambda x: len(str(x).split()))
test_df['len_word_q2'] = test_df.question2.apply(lambda x: len(str(x).split()))
test_df['common_words'] = test_df.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
test_df['fuzz_qratio'] = test_df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
test_df['fuzz_WRatio'] = test_df.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
test_df['fuzz_partial_ratio'] = test_df.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
test_df['fuzz_partial_token_set_ratio'] = test_df.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
test_df['fuzz_partial_token_sort_ratio'] = test_df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
test_df['fuzz_token_set_ratio'] = test_df.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
test_df['fuzz_token_sort_ratio'] = test_df.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

features_to_use = cols = [col for col in train_df.columns if col not in ['qid1','qid2','question1', 'question2','is_duplicate']]
sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/train_features_02.csv' 
train_df[features_to_use].to_csv(sub_file, index=False)

features_to_use = cols = [col for col in test_df.columns if col not in ['qid1','qid2','question1', 'question2','is_duplicate']]
sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/test_features_02.csv' 
test_df[features_to_use].to_csv(sub_file, index=False)
#######################################################################################################################################



model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/SriPrav/Documents/R/23Quora/preModels/GoogleNews-vectors-negative300.bin.gz', binary=True)
train_df['wmd'] = train_df.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)


norm_model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/SriPrav/Documents/R/23Quora/preModels/GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)
train_df['norm_wmd'] = train_df.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

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

cPickle.dump(question1_vectors, open('train_df/q1_w2v.pkl', 'wb'), -1)
cPickle.dump(question2_vectors, open('train_df/q2_w2v.pkl', 'wb'), -1)

train_df.to_csv('train_df/quora_features.csv', index=False)

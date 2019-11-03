# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:27:34 2017

@author: PAdepu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:28:14 2017

@author: PAdepu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:18:52 2017

@author: PAdepu
"""



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
import pandas as pd
import numpy as np
import nltk
from collections import Counter
from sklearn.metrics import log_loss
from scipy.optimize import minimize


import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

from nltk.stem.porter import *
stemmer = PorterStemmer()

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

###########################################################################################################
###########################################################################################################

inDir = 'C:/Users/padepu/Documents/R/23Quora'

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
train = pd.read_csv(train_file, encoding="utf-8")
test = pd.read_csv(test_file, encoding="utf-8")
print(train.shape) # (404290, 6)
print(test.shape)  # (2345796, 3)

###########################################################################################################
###########################################################################################################
train.isnull().values.any()
train.isnull().sum()

train['question1'].isnull().values.any()
train['question2'].isnull().values.any()

train['question2'] = train['question2'].fillna("no")

train.isnull().sum()

###########################################################################################################
###########################################################################################################

test.isnull().values.any()
test.isnull().sum()

test['question1'] = test['question1'].fillna("no")
test['question2'] = test['question2'].fillna("no")

test.isnull().sum()
###########################################################################################################
###########################################################################################################

#def str_stem(s): 
#    if isinstance(s, str):
#        s = s.lower()
#        s = s.replace("  "," ")
#        #s = (" ").join([stemmer.stem(z) for z in s.lower().split(" ")])
#        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
#        return s.lower()
#    else:
#        return "null"
#
#def str_lemmatizer(s): 
#    if isinstance(s, str):
#        s = s.lower()
#        s = s.replace("  "," ")
#        #s = (" ").join([stemmer.stem(z) for z in s.lower().split(" ")])
#        s = (" ").join([wordnet_lemmatizer.lemmatize(z) for z in s.split(" ")])
#        return s.lower()
#    else:
#        return "null"
        
###########################################################################################################
###########################################################################################################
        
train['question1'].tail()

train['s_question1'] = train['question1'].map(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
train['s_question1'].tail()


#train['Porterstem_question1'] = train['s_question1'].astype('str').map(lambda x:str_stem(x))
#train['lemmatizer_question1'] = train['s_question1'].astype('str').map(lambda x:str_lemmatizer(x))

train['question2'].head()

train['s_question2'] = train['question2'].map(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
train['s_question2'].head()


#train['Porterstem_question2'] = train['s_question2'].astype('str').map(lambda x:str_stem(x))
#train['lemmatizer_question2'] = train['s_question2'].astype('str').map(lambda x:str_lemmatizer(x))
#
#train['Porterstem_question2'].head()
#train['lemmatizer_question2'].head()
###########################################################################################################
###########################################################################################################

test['question1'].head()

test['s_question1'] = test['question1'].map(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
test['s_question1'].head()


#test['Porterstem_question1'] = test['s_question1'].astype('str').map(lambda x:str_stem(x))
#test['lemmatizer_question1'] = test['s_question1'].astype('str').map(lambda x:str_lemmatizer(x))

test['question2'].head()

test['s_question2'] = test['question2'].map(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
test['s_question2'].head()


#test['Porterstem_question2'] = test['s_question2'].astype('str').map(lambda x:str_stem(x))
#test['lemmatizer_question2'] = test['s_question2'].astype('str').map(lambda x:str_lemmatizer(x))
#
#test['Porterstem_question2'].head()
#test['lemmatizer_question2'].head()

###########################################################################################################
###########################################################################################################
import difflib
stops = set(stopwords.words("english"))

def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.ratio()

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['s_question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['s_question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

def get_weight(count, eps=500, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

train_qs = pd.Series(train['s_question1'].tolist() + train['s_question2'].tolist()).astype(str)
test_qs = pd.Series(test['s_question1'].tolist() + test['s_question2'].tolist()).astype(str)
words = (" ".join(train_qs)).lower().split() + (" ".join(test_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['s_question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['s_question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R
    
def get_unigrams(que):
    return [word for word in nltk.word_tokenize(que.lower()) if word not in stops]

def get_common_unigrams(row):
    return len( set(row["unigrams_ques1"]).intersection(set(row["unigrams_ques2"])) )

def get_common_unigram_ratio(row):
    return float(row["zunigrams_common_count"]) / max(len( set(row["unigrams_ques1"]).union(set(row["unigrams_ques2"])) ),1)

def get_bigrams(que):
    return [i for i in nltk.ngrams(que, 2)]

def get_common_bigrams(row):
    return len( set(row["bigrams_ques1"]).intersection(set(row["bigrams_ques2"])) )

def get_common_bigram_ratio(row):
    return float(row["zbigrams_common_count"]) / max(len( set(row["bigrams_ques1"]).union(set(row["bigrams_ques2"])) ),1)

###########################################################################################################
###########################################################################################################

train['question1_nouns'] = train.s_question1.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
train['question2_nouns'] = train.s_question2.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])


train['z_noun_match'] = train.apply(lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)

train['z_match_ratio'] = train.apply(lambda r: diff_ratios(r.s_question1, r.s_question2), axis=1)

train['z_word_match'] = train.apply(word_match_share, axis=1, raw=True)
train['z_tfidf_word_match'] = train.apply(tfidf_word_match_share, axis=1, raw=True)


#-------------------------------------------------------------------------------------------------

test['question1_nouns'] = test.s_question1.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
test['question2_nouns'] = test.s_question2.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])


test['z_noun_match'] = test.apply(lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)

test['z_match_ratio'] = test.apply(lambda r: diff_ratios(r.s_question1, r.s_question2), axis=1)

test['z_word_match'] = test.apply(word_match_share, axis=1, raw=True)
test['z_tfidf_word_match'] = test.apply(tfidf_word_match_share, axis=1, raw=True)



###########################################################################################################################################
###########################################################################################################################################

features_to_use =  ['id','z_noun_match','z_match_ratio','z_word_match','z_tfidf_word_match']
sub_file = 'C:/Users/padepu/Documents/R/23Quora/input/train_feature_07.csv' 
train[features_to_use].to_csv(sub_file, index=False)

features_to_use =  ['test_id','z_noun_match','z_match_ratio','z_word_match','z_tfidf_word_match']
sub_file = 'C:/Users/padepu/Documents/R/23Quora/input/test_feature_07.csv' 
test[features_to_use].to_csv(sub_file, index=False)

###########################################################################################################################################
###########################################################################################################################################

from sklearn.metrics import roc_auc_score
print('Original AUC:', roc_auc_score(train['is_duplicate'], train['z_noun_match'].fillna(0)))
print('   TFIDF AUC:', roc_auc_score(train['is_duplicate'], train['z_match_ratio'].fillna(0)))
print('Original AUC:', roc_auc_score(train['is_duplicate'], train['z_word_match'].fillna(0)))
print('   TFIDF AUC:', roc_auc_score(train['is_duplicate'], train['z_tfidf_word_match'].fillna(0)))


train[features_to_use].head()
test[features_to_use].head()
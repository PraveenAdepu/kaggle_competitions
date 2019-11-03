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

import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.engine.topology import Merge
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text



inDir = 'C:/Users/SriPrav/Documents/R/23Quora'

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train_df.shape) # (404290, 6)
print(test_df.shape)  # (2345796, 3)

x_train = train_df
              
######################################################################################################################



model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/SriPrav/Documents/R/23Quora/preModels/GoogleNews-vectors-negative300.bin.gz', binary=True)
#train_df['wmd'] = train_df.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
#
#norm_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
#norm_model.init_sims(replace=True)
#train_df['norm_wmd'] = train_df.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

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
    
question1_vectors = np.zeros((x_train.shape[0], 300))
error_count = 0

for i, q in tqdm(enumerate(x_train.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((x_train.shape[0], 300))

for i, q in tqdm(enumerate(x_train.question2.values)):
    question2_vectors[i, :] = sent2vec(q)


############################################################################################################################


test_question1_vectors = np.zeros((test_df.shape[0], 300))
error_count = 0

for i, q in tqdm(enumerate(test_df.question1.values)):
    test_question1_vectors[i, :] = sent2vec(q)

test_question2_vectors  = np.zeros((test_df.shape[0], 300))
for i, q in tqdm(enumerate(test_df.question2.values)):
    test_question2_vectors[i, :] = sent2vec(q)

train_q1 = pd.DataFrame(question1_vectors)
train_q1.shape
train_q2 = pd.DataFrame(question2_vectors)
train_q2.shape

list(train_q1.columns.values)

test_q1 = pd.DataFrame(test_question1_vectors)
test_q2 = pd.DataFrame(test_question2_vectors)

train_q1.columns = [str(col) + '_q1' for col in train_q1.columns]
train_q2.columns = [str(col) + '_q2' for col in train_q2.columns]

test_q1.columns = [str(col) + '_q1' for col in test_q1.columns]
test_q2.columns = [str(col) + '_q2' for col in test_q2.columns]
              
###############################################################################################################################

train_q1 = train_q1.apply(pd.to_numeric)
train_q2 = train_q2.apply(pd.to_numeric)

test_q1 = test_q1.apply(pd.to_numeric)
test_q2 = test_q2.apply(pd.to_numeric)

y_train = x_train['is_duplicate'].apply(pd.to_numeric).values


model = Sequential()
print('Build model...')

model5 = Sequential()
model5.add(Embedding(120000 + 1, 300, input_length=300, dropout=0.2))
model5.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

model6 = Sequential()
model6.add(Embedding(120000 + 1, 300, input_length=300, dropout=0.2))
model6.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

merged_model = Sequential()
merged_model.add(Merge([model5, model6], mode='concat'))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

#merged_model.add(Dense(300))
#merged_model.add(PReLU())
#merged_model.add(Dropout(0.2))
#merged_model.add(BatchNormalization())
#
#merged_model.add(Dense(300))
#merged_model.add(PReLU())
#merged_model.add(Dropout(0.2))
#merged_model.add(BatchNormalization())
#
#merged_model.add(Dense(300))
#merged_model.add(PReLU())
#merged_model.add(Dropout(0.2))
#merged_model.add(BatchNormalization())
#
#merged_model.add(Dense(300))
#merged_model.add(PReLU())
#merged_model.add(Dropout(0.2))
#merged_model.add(BatchNormalization())

merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid'))

merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint('dn6_question_pairs_weights.h5', monitor='val_loss', save_best_only=True, verbose=2)

merged_model.fit([ train_q1, train_q2], y=y_train, batch_size=100, nb_epoch=10,
                 verbose=1, validation_split=0.2, shuffle=True, callbacks=[checkpoint])

merged_model.load_weights('C:/Users/SriPrav/Documents/R/23Quora/dn6_question_pairs_weights.h5')

#loss, accuracy = model.evaluate([Q1_test, Q2_test], y_test)
#print('')
#print('loss      = {0:.4f}'.format(loss))
#print('accuracy  = {0:.4f}'.format(accuracy))
#print('precision = {0:.4f}'.format(precision))
#print('recall    = {0:.4f}'.format(recall))
#print('F         = {0:.4f}'.format(fbeta_score))

preds = merged_model.predict([test_q1, test_q2])
sub = pd.DataFrame()
test_pd = pd.read_csv(test_file)
sub['test_id'] = test_pd['test_id']
sub['is_duplicate'] = preds

sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.deepnet03.csv'
sub.to_csv(sub_file, index=False)
































###############################################################################################################################

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, merge, BatchNormalization, Activation, Input, Merge
import argparse
import sys
from gensim.utils import tokenize
import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


###############################################################################
# Merginf TF-IDF scores with Word2Vec
###############################################################################
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in list(tokenize(words , deacc=True)) if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
###############################################################################
# Stats on WordVec vectors
##############################################################################

#def compare_two_pairs(pair_num, data):
#    a = data[pair_num,0,:]
#    b = data[pair_num,1,:]
#    return euclidean(a,b)
#
#def compute_feats_discrimination(data, labels):
#    unique_labels = np.unique(labels)
#    avg_dist_dict = {}
#    for label in unique_labels:
#        sub_data = data[labels==label]
#        sub_res_data = data[labels!=label]
#        dists = []
#        for i in tqdm(range(sub_data.shape[0])):
#            # compute pari distance
#            pair_dist = compare_two_pairs(i, sub_data)
#            
#            # compute mean non pair distance
#            idxs = np.random.permutation(sub_res_data.shape[0])[0:10]
#            non_pair_dist = cdist(sub_res_data[idxs,0,:], sub_data[i,1,:][None,:], metric='euclidean').mean()
#            
#            # append to results
#            dists.append([dist, non_pair_dist])
#        avg_dist = np.mean(dists, axis=1)
#        avg_dist_dict[label] = avg_dist
#    return avg_dist_dict
#
################################################################################
## Evaluation functions for Stanford GLOVE vectors
###############################################################################
#def generate_glove(vocab_file, vectors_file):
#    with open(vocab_file, 'r') as f:
#        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
#    with open(vectors_file, 'r') as f:
#        vectors = {}
#        for line in f:
#            vals = line.rstrip().split(' ')
#            vectors[vals[0]] = [float(x) for x in vals[1:]]
#
#    vocab_size = len(words)
#    vocab = {w: idx for idx, w in enumerate(words)}
#    ivocab = {idx: w for idx, w in enumerate(words)}
#
#    vector_dim = len(vectors[ivocab[0]])
#    W = np.zeros((vocab_size, vector_dim))
#    for word, v in vectors.items():
#        if word == '<unk>':
#            continue
#        W[vocab[word], :] = v
#
#    # normalize each word vector to unit variance
#    W_norm = np.zeros(W.shape)
#    d = (np.sum(W ** 2, 1) ** (0.5))
#    W_norm = (W.T / d).T
#    return (W_norm, vocab, ivocab)
#
#
#def distance_glove(W, vocab, ivocab, input_term):
#    for idx, term in enumerate(input_term.split(' ')):
#        if term in vocab:
#            print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
#            if idx == 0:
#                vec_result = np.copy(W[vocab[term], :])
#            else:
#                vec_result += W[vocab[term], :] 
#        else:
#            print('Word: %s  Out of dictionary!\n' % term)
#            return
#    
#    vec_norm = np.zeros(vec_result.shape)
#    d = (np.sum(vec_result ** 2,) ** (0.5))
#    vec_norm = (vec_result.T / d).T
#
#    dist = np.dot(W, vec_norm.T)
#
#    for term in input_term.split(' '):
#        index = vocab[term]
#        dist[index] = -np.Inf
#
#    a = np.argsort(-dist)[:N]
#
#    print("\n                               Word       Cosine distance\n")
#    print("---------------------------------------------------------\n")
#    for x in a:
#        print("%35s\t\t%f\n" % (ivocab[x], dist[x]))


###############################################################################
# Evaluation functions for Stanford GLOVE vectors
###############################################################################


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

#def contrastive_loss(y_true, y_pred):
#    '''Contrastive loss from Hadsell-et-al.'06
#    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#    '''
#    margin = 1
#    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
#    
def create_base_network(input_dim):
    '''
    Base network for feature extraction.
    '''
    input = Input(shape=(input_dim, ))
    dense1 = Dense(128)(input)
    bn1 = BatchNormalization(mode=2)(dense1)
    relu1 = Activation('relu')(bn1)

    dense2 = Dense(128)(relu1)
    bn2 = BatchNormalization(mode=2)(dense2)
    res2 = merge([relu1, bn2], mode='sum')
    relu2 = Activation('relu')(res2)    

    dense3 = Dense(128)(relu2)
    bn3 = BatchNormalization(mode=2)(dense3)
    res3 = Merge(mode='sum')([relu2, bn3])
    relu3 = Activation('relu')(res3)   
    
    feats = merge([relu3, relu2, relu1], mode='concat')
    bn4 = BatchNormalization(mode=2)(feats)

    model = Model(input=input, output=bn4)

    return model

#
#def compute_accuracy(predictions, labels):
#    '''
#    Compute classification accuracy with a fixed threshold on distances.
#    '''
#    return labels[predictions.ravel() < 0.5].mean()

def create_network(input_dim):
    # network definition
    base_network = create_base_network(input_dim)
    
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))
    
    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    
    model = Model(input=[input_a, input_b], output=distance)
    return model

# avoid decoding problems
import sys
import os  
import pandas as pd
import numpy as np
from tqdm import tqdm
##############################################################################
# LOAD DATA
##############################################################################

inDir = 'C:/Users/SriPrav/Documents/R/23Quora'

train_file = inDir + "/input/train.csv"
df = pd.read_csv(train_file)
#df = pd.read_csv("/media/eightbit/8bit_5tb/NLP_data/Quora/DuplicateQuestion/quora_duplicate_questions.tsv",delimiter='\t')

# encode questions to unicode
df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))

##############################################################################
# TRAIN GLOVE
##############################################################################
import gensim


questions = list(df['question1']) + list(df['question2'])

# tokenize
c = 0
for question in tqdm(questions):
    questions[c] = list(gensim.utils.tokenize(question, deacc=True, lower=True))
    c += 1

# train model
model = gensim.models.Word2Vec(questions, size=300, workers=16, iter=10, negative=20)

# trim memory
model.init_sims(replace=True)

# creta a dict 
w2v = dict(zip(model.wv.index2word, model.wv.syn0))
print "Number of tokens in Word2Vec:", len(w2v.keys())

# save model
#model.save('data/3_word2vec.mdl')
#model.save_word2vec_format('data/3_word2vec.bin', binary=True)
#del questions
    
##############################################################################
# EXTRACT FEATURES
##############################################################################


# gather all questions
questions = list(df['question1']) + list(df['question2'])

# tokenize questions
c = 0
for question in tqdm(questions):
    questions[c] = list(gensim.utils.tokenize(question, deacc=True))
    c += 1

#    me = MeanEmbeddingVectorizer(w2v)
me = TfidfEmbeddingVectorizer(w2v)
me.fit(questions)
# exctract word2vec vectors
vecs1 = me.transform(df['question1'])
df['q1_feats'] = list(vecs1)

vecs2 = me.transform(df['question2'])
df['q2_feats'] = list(vecs2)

# save features
#pd.to_pickle(df, 'data/3_df.pkl')

##############################################################################
# CREATE TRAIN DATA
##############################################################################
# shuffle df
df = df.reindex(np.random.permutation(df.index))

# set number of train and test instances
num_train = int(df.shape[0] * 0.9)
num_test = df.shape[0] - num_train                 
print("Number of training pairs: %i"%(num_train))
print("Number of testing pairs: %i"%(num_test))

# init data data arrays
X_train = np.zeros([num_train, 2, 300])
X_test  = np.zeros([num_test, 2, 300])
Y_train = np.zeros([num_train]) 
Y_test = np.zeros([num_test])

# format data 
b = [a[None,:] for a in list(df['q1_feats'].values)]
q1_feats = np.concatenate(b, axis=0)

b = [a[None,:] for a in list(df['q2_feats'].values)]
q2_feats = np.concatenate(b, axis=0)

# fill data arrays with features
X_train[:,0,:] = q1_feats[:num_train]
X_train[:,1,:] = q2_feats[:num_train]
Y_train = df[:num_train]['is_duplicate'].values
            
X_test[:,0,:] = q1_feats[num_train:]
X_test[:,1,:] = q2_feats[num_train:]
Y_test = df[num_train:]['is_duplicate'].values

del b
del q1_feats
del q2_feats

# preprocess data, mean center unit std
#from sklearn.preprocessing import normalize
#X_train_norm = np.zeros_like(X_train)
#X_train_norm[:,0,:] = normalize(X_train[:,0,:], axis=0)
#X_train_norm[:,1,:] = normalize(X_train[:,1,:], axis=0)
#X_test_norm = np.zeros_like(X_test)
#X_test_norm[:,0,:] = normalize(X_test[:,0,:], axis=0)
#X_test_norm[:,1,:] = normalize(X_test[:,1,:], axis=0)

##############################################################################
# TRAIN MODEL
# 3 layers resnet (before relu) + adam + layer concat : 0.68
# 3 layers resnet (before relu) + adam + layer concat + 20 negative sampling: ?
##############################################################################           
# create model

train_q1 = np.array(train_q1)
train_q2 = np.array(train_q2)

from keras.optimizers import RMSprop, SGD, Adam
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint
net = create_network(300)
MODEL_WEIGHTS_FILE = 'dn6_question_pairs_weights.h5'
# train
#optimizer = SGD(lr=0.01, momentum=0.8, nesterov=True, decay=0.004)
optimizer = Adam(lr=0.001)
#net.compile(loss=contrastive_loss, optimizer=optimizer)
net.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True)]

net.fit([train_q1, train_q2], y=y_train, batch_size=384, nb_epoch=10,
           verbose=1, validation_split=0.1, shuffle=True, callbacks=callbacks)
             
#for epoch in range(50):
net.fit([X_train[:,0,:], X_train[:,1,:]], Y_train,
      validation_data=([X_test[:,0,:], X_test[:,1,:]], Y_test),
      batch_size=128, nb_epoch=20, shuffle=True,verbose=2,callbacks=callbacks)
    
    # compute final accuracy on training and test sets
    pred = net.predict([X_test[:,0,:], X_test[:,1,:]])
    te_acc = compute_accuracy(pred, Y_test)
    
#    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
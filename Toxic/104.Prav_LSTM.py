# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:06:10 2017

@author: SriPrav
"""



import numpy as np

random_state = 201802
np.random.seed(201802)
np.random.RandomState(random_state)

import pandas as pd
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint
from keras.optimizers import SGD, Adam
from gensim.models import KeyedVectors
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
import re
       
import os


import csv
import codecs


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys

########################################
## set directories and parameters
########################################



from keras import backend as K
from keras.engine.topology import Layer
#from keras import initializations
from keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim
    
    
inDir = 'C:/Users/SriPrav/Documents/R/42Toxic'

train = pd.read_csv(inDir+'/input/train.csv')
test  = pd.read_csv(inDir+'/input/test.csv')
Prav_5folds_CVindices = pd.read_csv(inDir+'/input/Prav_5folds_CVindices.csv')

train.shape #(95851, 8)
test.shape  #(226998, 2)
Prav_5folds_CVindices.shape #(95851, 2)

trainingSet = pd.merge(train, Prav_5folds_CVindices, on='id', how='left')

EMBEDDING_FILE = inDir + '/preModels/glove.42B.300d.txt'
#EMBEDDING_FILE = inDir + '/preModels/GoogleNews-vectors-negative300.bin'

########################################
## process texts in datasets
########################################
print('Processing text dataset')

#Regex to remove all Non-Alpha Numeric and space
special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)

#regex to replace all numerics
replace_numbers=re.compile(r'\d+',re.IGNORECASE)

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)
    
    #Remove Special Characters
    text=special_character_removal.sub('',text)
    
    #Replace Numbers
    text=replace_numbers.sub('n',text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)
    
#texts_1 = train['question1'].apply(lambda x: text_to_wordlist(str(x)))
#texts_2 = test['question2'].apply(lambda x: text_to_wordlist(str(x)))



MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 323760
EMBEDDING_DIM = 300


list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("CVxTz").values


comments = []
for text in list_sentences_train:
    comments.append(text_to_wordlist(text))
    
test_comments=[]
for text in list_sentences_test:
    test_comments.append(text_to_wordlist(text))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(comments + test_comments) #

sequences = tokenizer.texts_to_sequences(comments)
test_sequences = tokenizer.texts_to_sequences(test_comments)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

X_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X_train.shape)
print('Shape of label tensor:', y.shape)

X_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of test_data tensor:', X_test.shape)

#tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS)
#tokenizer.fit_on_texts(list(list_sentences_train)+list(list_sentences_test)) #+list(list_sentences_test)
#list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
#list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
#X_train = sequence.pad_sequences(list_tokenized_train, maxlen=MAX_SEQUENCE_LENGTH)
#X_test  = sequence.pad_sequences(list_tokenized_test, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

########################################
## index word vectors
########################################
#print('Indexing word vectors')
#
#word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
#        binary=True)
#print('Found %s word vectors of word2vec' % len(word2vec.vocab))
#
#print('Indexing word vectors')

#Glove Vectors
embeddings_index = {}
f = open(EMBEDDING_FILE, encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
#word = ' '.join(values[:-300])
#coefs = np.asarray(values[-300:], dtype='float32')
print('Total %s word vectors.' % len(embeddings_index))

########################################
## prepare embeddings
########################################
#print('Preparing embedding matrix')
#
#nb_words = min(MAX_NB_WORDS, len(word_index))+1
#
#embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
#for word, i in word_index.items():
#    if word in word2vec.vocab:
#        embedding_matrix[i] = word2vec.word_vec(word)
#print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0)) #Null word embeddings: 96546

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))



########################################
## sample train/validation data
########################################

num_lstm = 300
num_dense = 256
rate_drop_lstm = 0.1
rate_drop_dense = 0.1

def lstm_model():
    
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,trainable=False))
    model.add(Bidirectional(LSTM(num_lstm, return_sequences=True)))
    
    model.add(GlobalMaxPool1D())
    
    model.add(Dropout(rate_drop_dense))
    model.add(BatchNormalization())
    
    model.add(Dense(num_dense, activation="relu"))
    model.add(Dropout(rate_drop_dense))
        
    model.add(Dense(6, activation="sigmoid"))
    
    return model
    
def get_model():
   
    inp = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    x = Embedding(MAX_NB_WORDS, EMBEDDING_DIM)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)    

    return model

def get_lstm_attention():
    embedding_layer = Embedding(nb_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False)

    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm,return_sequences=True)

    comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences= embedding_layer(comment_input)
    x = lstm_layer(embedded_sequences)
    x = Dropout(rate_drop_dense)(x)
    merged = Attention(MAX_SEQUENCE_LENGTH)(x)
    merged = Dense(num_dense, activation='relu')(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(6, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[comment_input], \
            outputs=preds)
    return model


MODEL_WEIGHTS_FILE = inDir + '/104_Prav_LSTM.h5'

nb_epoch = 10
VERBOSEFLAG = 1
batch_size  = 128
patience = 2
optim_type = 'Adam'
learning_rate = 1e-3
nbags = 1


ModelName = '104_Prav_LSTM'

def lstmnet(i):

    print('Fold ', i , ' Processing')
    X_build = trainingSet[trainingSet['CVindices'] != i] # 636112
    X_val   = trainingSet[trainingSet['CVindices'] == i]
    
    trainindex = trainingSet[trainingSet['CVindices'] != i].index.tolist()
    valindex   = trainingSet[trainingSet['CVindices'] == i].index.tolist()

    X_build_data, y_build = X_train[trainindex], y[trainindex]
    X_valid_data, y_valid = X_train[valindex], y[valindex]   
       
    pred_cv = np.zeros([X_val.shape[0],6])    
    pred_test = np.zeros([test.shape[0],6])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros([X_val.shape[0],6])
        
        model = get_lstm_attention()
        callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=VERBOSEFLAG),
        ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=VERBOSEFLAG),
                ]
        #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        if optim_type == 'SGD':
            optim = SGD(lr=learning_rate, momentum=0.9)
        else:
            optim = Adam(lr=learning_rate)
        model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
        
        model.fit(X_build_data, y_build, 
                  validation_data=(X_valid_data, y_valid), 
                  batch_size=batch_size, 
                  nb_epoch=nb_epoch,
                  callbacks=callbacks,
                  verbose=VERBOSEFLAG

                 )
        model.load_weights(MODEL_WEIGHTS_FILE)
        
        bag_cv  += model.predict(X_valid_data, batch_size=batch_size, verbose=VERBOSEFLAG)        
        pred_cv += model.predict(X_valid_data, batch_size=batch_size, verbose=VERBOSEFLAG)        
        pred_test += model.predict(X_test, batch_size=batch_size, verbose=VERBOSEFLAG)
#        bag_score = log_loss(labels_val[:X_val.shape[0]], bag_cv_final)
#        print('bag ', j, '- logloss:', bag_score)
#        os.remove('C:/Users/SriPrav/Documents/R/23Quora/dn50_question_pairs_weights.h5')
    pred_cv /= nbags
    

    pred_test/= nbags
#    fold_score = log_loss(labels_val[:X_val.shape[0]], pred_cv_final)
#    print('Fold ', i, '- logloss:', fold_score)
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = list_classes
    pred_cv["id"] = X_val.id.values
    pred_cv = pred_cv[["id",'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    sub_valfile = inDir+'/submissions/Prav.'+ModelName+'.fold' + str(i) + '.csv'    
    pred_cv.to_csv(sub_valfile, index=False)
    
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = list_classes
    pred_test["id"] = test.id.values
    pred_test = pred_test[["id",'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    sub_file = inDir+'/submissions/Prav.'+ModelName+'.fold' + str(i) + '-test' + '.csv'
    pred_test.to_csv(sub_file, index=False)
    del pred_cv
    del pred_test
    os.remove(MODEL_WEIGHTS_FILE)

#
#fullepochs = 22
#def full_lstmnet(i):
#    print('Full Processing')
#   
#
#    data_1_train = np.vstack((data_1, data_2))
#    data_2_train = np.vstack((data_2, data_1))
#    labels_train = np.concatenate((labels, labels))
#    
#    pred_test = np.zeros(test_df.shape[0])
#    
#    for j in range(1,nbags+1):
#        print('bag ', j , ' Processing')
#        model = lstm_model()
##        Enable this and model remove after getting best epoch value
##        callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=1)]
##        model.fit(X_train.values, X_trainy, nb_epoch=epochs, batch_size=batchsize,  verbose=verboselog,validation_data=(X_valid.values,X_validy),callbacks=callbacks)
##        model.load_weights('C:/Users/SriPrav/Documents/R/23Quora/nn2_question_pairs_weights.h5')
#        model.fit([data_1_train, data_2_train], labels_train, 
#                  batch_size=batchsize, 
#                  nb_epoch=fullepochs,
#                  verbose=verboselog,
#                  class_weight=class_weight
##                 callbacks=[early_stopping, model_checkpoint]
#                 )
#        
#        pred_test += model.predict([test_data_1, test_data_2], batch_size=batchsize, verbose=1)[:,0]
#        pred_test += model.predict([test_data_2, test_data_1], batch_size=batchsize, verbose=1)[:,0]       
#    pred_test/= 2*nbags
#    pred_test = pd.DataFrame(pred_test)
#    pred_test.columns = ["is_duplicate"]
#    pred_test["test_id"] = test_df.test_id.values
#    pred_test = pred_test[['test_id','is_duplicate']]
#    sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.dn51.full.csv'
#    pred_test.to_csv(sub_file, index=False)
    
nbags = 1
folds = 5
if __name__ == '__main__':
    for i in range(1, folds+1):
        lstmnet(i)
#    full_lstmnet(nbags)
    
    





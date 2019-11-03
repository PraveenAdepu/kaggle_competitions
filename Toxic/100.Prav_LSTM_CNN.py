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
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, Conv1D,GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint
from keras.optimizers import SGD, Adam
from gensim.models import KeyedVectors
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential
       
import os

inDir = 'C:/Users/SriPrav/Documents/R/42Toxic'

train = pd.read_csv(inDir+'/input/train.csv')
test  = pd.read_csv(inDir+'/input/test.csv')
Prav_5folds_CVindices = pd.read_csv(inDir+'/input/Prav_5folds_CVindices.csv')

train.shape #(95851, 8)
test.shape  #(226998, 2)
Prav_5folds_CVindices.shape #(95851, 2)

trainingSet = pd.merge(train, Prav_5folds_CVindices, on='id', how='left')

EMBEDDING_FILE = inDir + '/preModels/GoogleNews-vectors-negative300.bin'


MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300


list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("CVxTz").values

tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(list_sentences_train)) #+list(list_sentences_test)
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_train = sequence.pad_sequences(list_tokenized_train, maxlen=MAX_SEQUENCE_LENGTH)
X_test  = sequence.pad_sequences(list_tokenized_test, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

########################################
## index word vectors
########################################
print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0)) #Null word embeddings: 96546

########################################
## sample train/validation data
########################################

num_lstm = 128
num_dense = 50
rate_drop_lstm = 0.1
rate_drop_dense = 0.1

filter_length = 5
nb_filter = 64

def lstm_cnn_model():
    
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,trainable=False))
    model.add(Bidirectional(LSTM(num_lstm, return_sequences=True)))
    
    model.add(Conv1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    model.add(Dropout(0.1))
    
    model.add(Conv1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    model.add(GlobalMaxPool1D())
    
    model.add(Dropout(rate_drop_dense))
    model.add(BatchNormalization())
    
    model.add(Dense(num_dense, activation="relu"))
    model.add(Dropout(rate_drop_dense))
        
    model.add(Dense(6, activation="sigmoid"))
    
    return model
    

MODEL_WEIGHTS_FILE = inDir + '/100_Prav_LSTM_CNN.h5'

nb_epoch = 10
VERBOSEFLAG = 1
batch_size  = 128
patience = 2
optim_type = 'Adam'
learning_rate = 1e-3
nbags = 1


ModelName = '100_Prav_LSTM_CNN'
model.summary()
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
        
        model = lstm_cnn_model()
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


    
nbags = 1
folds = 5
if __name__ == '__main__':
    for i in range(1, folds+1):
        lstmnet(i)
#    full_lstmnet(nbags)
    
    





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
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 128


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

EMBEDDING_FILE = inDir + '/preModels/GoogleNews-vectors-negative300.bin'

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
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## sample train/validation data
########################################

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


MODEL_WEIGHTS_FILE = inDir + '/101_Prav_LSTM.h5'

nb_epoch = 10
VERBOSEFLAG = 1
batch_size  = 128
patience = 2
optim_type = 'Adam'
learning_rate = 1e-3
nbags = 1


ModelName = '102_Prav_LSTM'

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
        
        model = get_model()
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
    
    





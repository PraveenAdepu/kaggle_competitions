# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:06:10 2017

@author: SriPrav
"""


import pandas as pd
import numpy as np

random_state = 201802
np.random.seed(201802)
np.random.RandomState(random_state)

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint
from keras.optimizers import SGD, Adam
from gensim.models import KeyedVectors
from keras.layers.merge import Concatenate


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
MAX_NB_WORDS = 20000
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

nb_epoch = 2
VERBOSEFLAG = 1
batch_size  = 32
patience = 5
optim_type = 'Adam'
learning_rate = 1e-3

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
model.summary()

model.fit(X_train, y, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.1, callbacks=callbacks,verbose = VERBOSEFLAG)

model.load_weights(MODEL_WEIGHTS_FILE)

y_test = model.predict(X_test, verbose=VERBOSEFLAG, batch_size=200)

pred_test = pd.DataFrame(y_test)
pred_test.columns = list_classes
pred_test["id"] = test.id.values
pred_test = pred_test[["id",'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    
pred_test.to_csv(inDir+"/submissions/baseline.csv", index=False)




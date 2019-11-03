
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 22:14:12 2017

@author: SriPrav
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.engine.topology import Merge
from keras.layers import TimeDistributed, Lambda , Input, merge
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
from keras.optimizers import RMSprop, SGD, Adam

inDir = 'C:/Users/SriPrav/Documents/R/23Quora'

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train_df.shape) # (404290, 6)
print(test_df.shape)  # (2345796, 3)

data = train_df
y = train_df['is_duplicate']
#pos_train = train_df[train_df['is_duplicate'] == 1]
#neg_train = train_df[train_df['is_duplicate'] == 0]
#
## Now we oversample the negative class
## There is likely a much more elegant way to do this...
#p = 0.165
#scale = ((len(pos_train)*1.0 / (len(pos_train) + len(neg_train))*1.0) / p) - 1
#while scale > 1:
#    neg_train = pd.concat([neg_train, neg_train])
#    scale -=1
#neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
#print(len(pos_train)*1.0 / (len(pos_train) + len(neg_train))*1.0)
#
#data = pd.concat([pos_train, neg_train]) # (780486,6)
#y = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
#del pos_train, neg_train
#
#import gc
#gc.collect()
#data = pd.read_csv(train_file)
#y = data.is_duplicate.values

tk = text.Tokenizer(nb_words=200000)

max_len = 25 #40
tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)) + list(test_df.question1.values.astype(str)) + list(test_df.question2.values.astype(str)) )
x1 = tk.texts_to_sequences(data.question1.values)
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

x11 = tk.texts_to_sequences(test_df.question1.values.astype(str))
x11 = sequence.pad_sequences(x11, maxlen=max_len)

x22 = tk.texts_to_sequences(test_df.question2.values.astype(str))
x22 = sequence.pad_sequences(x22, maxlen=max_len)

word_index = tk.word_index

ytrain_enc = np_utils.to_categorical(y)

embeddings_index = {}
f = open('C:/Users/SriPrav/Documents/R/23Quora/preModels/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

max_features = 200000
filter_length = 5
nb_filter = 64
pool_length = 4

model = Sequential()
print('Build model...')

model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))

model1.add(TimeDistributed(Dense(300, activation='relu')))
model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

model2 = Sequential()
model2.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))

model2.add(TimeDistributed(Dense(300, activation='relu')))
model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

model3 = Sequential()
model3.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model3.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model3.add(Dropout(0.2))

model3.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model3.add(GlobalMaxPooling1D())
model3.add(Dropout(0.2))

model3.add(Dense(200))
model3.add(Dropout(0.2))
model3.add(BatchNormalization())

model4 = Sequential()
model4.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model4.add(Dropout(0.2))

model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model4.add(GlobalMaxPooling1D())
model4.add(Dropout(0.2))

model4.add(Dense(200))
model4.add(Dropout(0.2))
model4.add(BatchNormalization())
model5 = Sequential()
model5.add(Embedding(len(word_index) + 1, 300, input_length=max_len, dropout=0.2))
model5.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

model6 = Sequential()
model6.add(Embedding(len(word_index) + 1, 300, input_length=max_len, dropout=0.2))
model6.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

merged_model = Sequential()
merged_model.add(Merge([model5, model6], mode='concat')) # model1, model2,, model5, model6
merged_model.add(BatchNormalization())

merged_model.add(Dense(200))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(200))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

#merged_model.add(Dense(200))
#merged_model.add(PReLU())
#merged_model.add(Dropout(0.2))
#merged_model.add(BatchNormalization())

#merged_model.add(Dense(200))
#merged_model.add(PReLU())
#merged_model.add(Dropout(0.2))
#merged_model.add(BatchNormalization())

#merged_model.add(Dense(200))
#merged_model.add(PReLU())
#merged_model.add(Dropout(0.2))
#merged_model.add(BatchNormalization())

merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid'))

merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



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

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
    
def compute_accuracy(predictions, labels):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

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

net = create_network(25)

optimizer = Adam(lr=0.001)
#net.compile(loss=contrastive_loss, optimizer=optimizer)
net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


   
checkpoint = ModelCheckpoint('dn6_question_pairs_weights.h5', monitor='val_loss', save_best_only=True, verbose=1)

net.fit([x1, x2], y=y, batch_size=384, nb_epoch=10,
                 verbose=1, validation_split=0.2, shuffle=True, callbacks=[checkpoint])

net.load_weights('C:/Users/SriPrav/Documents/R/23Quora/dn6_question_pairs_weights.h5')

preds = net.predict([x11, x22])#,x11, x22
print(preds.shape)

out_df = pd.DataFrame({"test_id":test_df['test_id'], "is_duplicate":preds.ravel()})
#sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.deepnet06.csv'
output_file = out_df[['test_id','is_duplicate']]
output_file.head()
output_file.to_csv(sub_file, index=False)


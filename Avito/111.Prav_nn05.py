# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:56:17 2018

@author: SriPrav
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 23 19:16:44 2018

@author: SriPrav
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:35:40 2018

@author: SriPrav
"""

'''
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!
'''

########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Merge
#from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from sklearn.metrics import roc_auc_score, log_loss


########################################
## set directories and parameters
########################################

inDir = 'C:/Users/SriPrav/Documents/R/48Avito'
EMBEDDING_FILE = inDir+'/input/fasttest-common-crawl-russian/cc.ru.300.vec'
TRAIN_DATA_FILE = inDir + '/input/train.csv'
TEST_DATA_FILE = inDir + '/input/test.csv'
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300

training = pd.read_csv(inDir+'/input/train.csv', parse_dates = ["activation_date"])
Prav_5folds_CVIndices = pd.read_csv(inDir+'./input/Prav_5folds_CVindices_weekdayStratified.csv')

train_df = pd.merge(training, Prav_5folds_CVIndices, how = 'inner', on = 'item_id')

train_df = train_df.reset_index(drop=True)

traindex = train_df.index

test_df = pd.read_csv(inDir+'/input/test.csv', parse_dates = ["activation_date"])
testdex = test_df.index

train_df.columns
test_df.columns

test_df['deal_probability'] = 0
test_df['CVindices'] = 0


Prav_train_FE01 = pd.read_csv(inDir+'/input/Prav_train_FE_01.csv')
Prav_test_FE01 = pd.read_csv(inDir+'/input/Prav_test_FE_01.csv')

train_df = pd.merge(train_df, Prav_train_FE01, how = 'left', on = 'item_id')
test_df  = pd.merge(test_df, Prav_test_FE01, how = 'left', on = 'item_id')

y = training.deal_probability.copy()
#training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*train_df.shape))
print('Test shape: {} Rows, {} Columns'.format(*test_df.shape))


Prav_train_FE03 = pd.read_csv(inDir+'/input/Prav_train_FE_03.csv')
Prav_test_FE03 = pd.read_csv(inDir+'/input/Prav_test_FE_03.csv')

train_df = pd.merge(train_df, Prav_train_FE03, how = 'left', on = 'item_id')
test_df  = pd.merge(test_df, Prav_test_FE03, how = 'left', on = 'item_id')

traintest_FE02 = pd.read_csv(inDir+"/input/traintest_FE_02.csv")

train_df = pd.merge(train_df, traintest_FE02, how = 'left', on = 'user_id')
test_df  = pd.merge(test_df, traintest_FE02, how = 'left', on = 'user_id')

#########################################################################################################
num_lstm = 110
num_dense = 60
rate_drop_lstm = 0.15
rate_drop_dense = 0.20

print(train_df.shape) # (404290, 6)
print(test_df.shape)  # (2345796, 3)


act = 'relu'
re_weight = False # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'Prav_nn05_weights.h5'

########################################
## index word vectors
########################################
print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, encoding="utf8")
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

########################################
## process texts in datasets
########################################
print('Processing text dataset')

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("ru"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('ru')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)
train_df['description']= (train_df['title']+" "+train_df['description']).astype(str)
train_df['param_2']= (train_df['param_1']+'_'+train_df['param_2']+'_'+train_df['param_3']).astype(str)
test_df['description']= (test_df['title']+" "+test_df['description']).astype(str)
test_df['param_2']= (test_df['param_1']+'_'+test_df['param_2']+'_'+test_df['param_3']).astype(str)

    
texts_1 = train_df['description'].apply(lambda x: text_to_wordlist(str(x)))
texts_2 = train_df['title'].apply(lambda x: text_to_wordlist(str(x)))

test_texts_1 = test_df['description'].apply(lambda x: text_to_wordlist(str(x)))
test_texts_2 = test_df['title'].apply(lambda x: text_to_wordlist(str(x)))

#############################################################################################################################

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(texts_1.values) + list(texts_2.values) + list(test_texts_1.values) + list(test_texts_2.values))

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(train_df.deal_probability.values)
train_ids = np.array(train_df.item_id.values)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_df.item_id.values)

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS: continue
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## prepare embeddings
########################################
#############################################################################################################################
#############################################################################################################################

train_test = pd.concat([train_df,test_df],axis=0)

train_test = train_test.replace(np.nan,-1,regex=True) #nan and other missing values are mapped to -1


##================Create the Tokenizers
region_tk = {x:i+1 for i, x in enumerate(train_test.region.unique())}#+1 because we want to reserve 0 for new but not missing values
city_tk =  {x:i+1 for i, x in enumerate(train_test.city.unique())}
cat1_tk =  {x:i+1 for i, x in enumerate(train_test.parent_category_name.unique())}
cat2_tk =  {x:i+1 for i, x in enumerate(train_test.category_name.unique())}
param1_tk =  {x:i+1 for i, x in enumerate(train_test.param_1.unique())}
param2_tk =  {x:i+1 for i, x in enumerate(train_test.param_2.unique())}
param3_tk =  {x:i+1 for i, x in enumerate(train_test.param_3.unique())}
#seqnum_tk =  {x:i+1 for i, x in enumerate(train_test.item_seq_number.unique())}
usertype_tk = {x:i+1 for i, x in enumerate(train_test.user_type.unique())}
imgtype_tk = {x:i+1 for i, x in enumerate(train_test.image_top_1.unique())}
tokenizers = [region_tk, city_tk, cat1_tk, cat2_tk, param1_tk, param2_tk, param3_tk, usertype_tk, imgtype_tk]#, seqnum_tk

##================These functions are going to get repeated on train, val, and test data
def tokenize_data(data, tokenizers):
    region_tk, city_tk, cat1_tk, cat2_tk, param1_tk, param2_tk, param3_tk, usertype_tk, imgtype_tk = tokenizers
    x_reg = np.asarray([region_tk.get(key, 0) for key in data.region], dtype=int)
    x_city   = np.asarray([city_tk.get(key, 0) for key in data.city], dtype=int)
    x_cat1   = np.asarray([cat1_tk.get(key, 0) for key in data.parent_category_name], dtype=int)
    x_cat2   = np.asarray([cat2_tk.get(key, 0) for key in data.category_name], dtype=int)
    x_prm1 = np.asarray([param1_tk.get(key, 0) for key in data.param_1], dtype=int)
    x_prm2 = np.asarray([param2_tk.get(key, 0) for key in data.param_2], dtype=int)
    x_prm3 = np.asarray([param3_tk.get(key, 0) for key in data.param_3], dtype=int)
#    x_sqnm = np.asarray([seqnum_tk.get(key, 0) for key in data.item_seq_number], dtype=int)
    x_usr = np.asarray([usertype_tk.get(key, 0) for key in data.user_type], dtype=int)
    x_itype = np.asarray([imgtype_tk.get(key, 0) for key in data.image_top_1], dtype=int)
    return [x_reg, x_city, x_cat1, x_cat2, x_prm1, x_prm2, x_prm3, x_usr, x_itype]#, x_sqnm

def log_prices(data):
    prices = data.price.as_matrix()
    prices = np.log1p(prices)
    prices[prices==-np.inf] = -1
    return prices

##================Final Processing on x, y train, val, test data
    
x_train_test = tokenize_data(train_test, tokenizers)

train_data_reg = x_train_test[0][:train_df.shape[0]]
test_data_reg  = x_train_test[0][train_df.shape[0]:]

train_data_city = x_train_test[1][:train_df.shape[0]]
test_data_city  = x_train_test[1][train_df.shape[0]:]

train_data_cat1 = x_train_test[2][:train_df.shape[0]]
test_data_cat1  = x_train_test[2][train_df.shape[0]:]

train_data_cat2 = x_train_test[3][:train_df.shape[0]]
test_data_cat2  = x_train_test[3][train_df.shape[0]:]

train_data_prm1 = x_train_test[4][:train_df.shape[0]]
test_data_prm1  = x_train_test[4][train_df.shape[0]:]

train_data_prm2 = x_train_test[5][:train_df.shape[0]]
test_data_prm2  = x_train_test[5][train_df.shape[0]:]

train_data_prm3 = x_train_test[6][:train_df.shape[0]]
test_data_prm3 = x_train_test[6][train_df.shape[0]:]

#train_data_sqnm = x_train_test[7][:train_df.shape[0]]
#test_data_sqnm = x_train_test[7][train_df.shape[0]:]

train_data_usr = x_train_test[7][:train_df.shape[0]]
test_data_usr = x_train_test[7][train_df.shape[0]:]

train_data_itype = x_train_test[8][:train_df.shape[0]]
test_data_itype = x_train_test[8][train_df.shape[0]:]

#############################################################################################################################
#############################################################################################################################

nn_features = [c for c in train_test if c not in ['item_id', 'user_id', 'region', 'city', 'parent_category_name',
       'category_name', 'param_1', 'param_2', 'param_3', 'title',
       'description', 'activation_date',
       'user_type', 'image', 'image_top_1', 'deal_probability', 'CVindices']]
train_test[nn_features].head()

train_test['price'].head()
train_test['price'].describe()

#train_test['price'] = np.log1p(train_test['price'])
#train_test['price'][train_test['price']==-np.inf] = -1

train_test['price'] = np.log1p(train_test['price'])
train_test['item_seq_number'] = np.log(train_test['item_seq_number'])

train_test['price'][train_test['price']==-np.inf] = 0 
train_test['price'].fillna(0, inplace=True)
train_test['price'].describe()
train_test['item_seq_number'].describe()

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_test[nn_features])
train_test[nn_features] = ss.transform(train_test[nn_features])


train_user_features = train_test[nn_features][:train_df.shape[0]]

train_user_features = train_user_features.values
test_user_features = train_test[nn_features][train_df.shape[0]:]

test_user_features = test_user_features.values

train_user_features.shape

#############################################################################################################################
#############################################################################################################################
#np.random.seed(1234)
import keras.backend as K
from sklearn import metrics

from keras.layers import Dense, Input, Embedding, Dropout, Flatten, Reshape,GRU
from keras.layers.noise import AlphaDropout, GaussianNoise
from keras.layers.merge import concatenate, dot, multiply, add

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def lstm_model():  
    
   
    model5 = Sequential()
    model5.add(Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,trainable=False))
#    model5.add(LSTM(num_lstm, dropout_W=rate_drop_lstm, dropout_U=rate_drop_lstm))
    model5.add(GRU(num_lstm))
    
#    model6 = Sequential()
#    model6.add(Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,trainable=False))
##    model6.add(LSTM(num_lstm, dropout_W=rate_drop_lstm, dropout_U=rate_drop_lstm))
#    model6.add(GRU(num_lstm))    
    
    emb_size = 10
        
    model_inp_reg = Sequential()
    model_inp_reg.add(Embedding(len(region_tk)+1, emb_size,input_length=1,trainable=False))
    model_inp_reg.add(Flatten())
    
    model_inp_city = Sequential()
    model_inp_city.add(Embedding(len(city_tk)+1, emb_size,input_length=1,trainable=False))
    model_inp_city.add(Flatten())
    
    model_inp_cat1 = Sequential()
    model_inp_cat1.add(Embedding(len(cat1_tk)+1, emb_size,input_length=1,trainable=False))
    model_inp_cat1.add(Flatten())
    
    model_inp_cat2 = Sequential()
    model_inp_cat2.add(Embedding(len(cat2_tk)+1, emb_size,input_length=1,trainable=False))
    model_inp_cat2.add(Flatten())
    
    model_inp_prm1 = Sequential()
    model_inp_prm1.add(Embedding(len(param1_tk)+1, emb_size,input_length=1,trainable=False))
    model_inp_prm1.add(Flatten())
    
    model_inp_prm2 = Sequential()
    model_inp_prm2.add(Embedding(len(param2_tk)+1, emb_size,input_length=1,trainable=False))
    model_inp_prm2.add(Flatten())
    
#    model_inp_prm3 = Sequential()
#    model_inp_prm3.add(Embedding(len(param3_tk)+1, emb_size,input_length=1,trainable=False))
#    model_inp_prm3.add(Flatten())
    
#    model_inp_sqnm = Sequential()
#    model_inp_sqnm.add(Embedding(len(seqnum_tk)+1, emb_size,input_length=1,trainable=False))
#    model_inp_sqnm.add(Flatten())
    
    model_inp_usr = Sequential()
    model_inp_usr.add(Embedding(len(usertype_tk)+1, emb_size,input_length=1,trainable=False))
    model_inp_usr.add(Flatten())
    
    model_inp_itype = Sequential()
    model_inp_itype.add(Embedding(len(imgtype_tk)+1, emb_size,input_length=1,trainable=False))
    model_inp_itype.add(Flatten())
    
    model7 = Sequential()
#    model7.add(input_dim = leaks.shape[1])
#    model7.add(shape=leaks.shape[1])
#    leaks_input = Input(shape=(leaks.shape[1],))
#    model7.add(Dense(num_dense/2, activation=act))
    model7.add(Dense(num_dense, input_dim = train_user_features.shape[1]))
    model7.add(PReLU())
    
    merged_model = Sequential()

    merged_model.add(Merge([model5, model_inp_reg,model_inp_city,model_inp_cat1,model_inp_cat2,model_inp_prm1,model_inp_prm2,model_inp_usr,model_inp_itype,model7], mode='concat'))
    
    merged_model.add(Dropout(rate_drop_dense))
    merged_model.add(BatchNormalization())
    
   
    merged_model.add(Dense(num_dense))
    merged_model.add(PReLU()) # act   
    
    
    merged_model.add(Dropout(rate_drop_dense))
    merged_model.add(BatchNormalization())
    
    merged_model.add(Dense(1))
    merged_model.add(Activation('sigmoid'))
    merged_model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics =[root_mean_squared_error])
    return(merged_model)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = 'dn51_question_pairs_weights.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

########################################
## add class weight
########################################

########################################
## add class weight
########################################
nbags = 1
folds = 1
epochs = 4
batchsize = 256
verboselog = 1  
fullepochs = 5

def lstmnet(i):

    print('Fold ', i , ' Processing')
    X_build = train_df[train_df['CVindices'] != i] # 636112
    #X_build = X_build.groupby('id').first().reset_index() # 331085
    X_val   = train_df[train_df['CVindices'] == i]
    
    trainindex = train_df[train_df['CVindices'] != i].index.tolist()
    valindex   = train_df[train_df['CVindices'] == i].index.tolist()

    data_1_train = data_1[trainindex]
#    data_2_train = data_2[trainindex]
    train_data_reg_train  = train_data_reg[trainindex]
    train_data_city_train  = train_data_city[trainindex]
    train_data_cat1_train  = train_data_cat1[trainindex]
    train_data_cat2_train  = train_data_cat2[trainindex]
    train_data_prm1_train  = train_data_prm1[trainindex]
    train_data_prm2_train  = train_data_prm2[trainindex]
#    train_data_prm3_train  = train_data_prm3[trainindex]
#    train_data_sqnm_train  = train_data_sqnm[trainindex]
    train_data_usr_train  = train_data_usr[trainindex]
    train_data_itype_train  = train_data_itype[trainindex]  
    
    train_user_features_train = train_user_features[trainindex]

    
    labels_train = labels[trainindex]
    
    data_1_val = data_1[valindex]
#    data_2_val = data_2[valindex]
    
    train_data_reg_val  = train_data_reg[valindex]
    train_data_city_val  = train_data_city[valindex]
    train_data_cat1_val  = train_data_cat1[valindex]
    train_data_cat2_val  = train_data_cat2[valindex]
    train_data_prm1_val  = train_data_prm1[valindex]
    train_data_prm2_val  = train_data_prm2[valindex]
#    train_data_prm3_val  = train_data_prm3[valindex]
#    train_data_sqnm_val  = train_data_sqnm[valindex]
    train_data_usr_val  = train_data_usr[valindex]
    train_data_itype_val  = train_data_itype[valindex]
    train_user_features_val = train_user_features[valindex]
    
    labels_val = labels[valindex]    
          
    pred_cv = np.zeros(X_val.shape[0])
    
    pred_test = np.zeros(test_df.shape[0])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros(X_val.shape[0])
       
        model = lstm_model()
        model.summary()

        model.fit([data_1_train, train_data_reg_train,train_data_city_train,train_data_cat1_train,train_data_cat2_train,train_data_prm1_train,train_data_prm2_train,train_data_usr_train,train_data_itype_train,train_user_features_train], labels_train, 
                  validation_data=([data_1_val, train_data_reg_val,train_data_city_val,train_data_cat1_val,train_data_cat2_val,train_data_prm1_val,train_data_prm2_val,train_data_usr_val,train_data_itype_val,train_user_features_val], labels_val), 
                  batch_size=batchsize, 
                  nb_epoch=epochs,
                  verbose=verboselog

                 )
        bag_cv   = model.predict([data_1_val, train_data_reg_val,train_data_city_val,train_data_cat1_val,train_data_cat2_val,train_data_prm1_val,train_data_prm2_val,train_data_usr_val,train_data_itype_val,train_user_features_val], batch_size=batchsize, verbose=1)[:,0]
        pred_cv += model.predict([data_1_val, train_data_reg_val,train_data_city_val,train_data_cat1_val,train_data_cat2_val,train_data_prm1_val,train_data_prm2_val,train_data_usr_val,train_data_itype_val,train_user_features_val], batch_size=batchsize, verbose=1)[:,0]
        
        pred_test += model.predict([test_data_1, test_data_reg,test_data_city,test_data_cat1,test_data_cat2,test_data_prm1,test_data_prm2,test_data_usr,test_data_itype,test_user_features], batch_size=batchsize, verbose=1)[:,0]
        
        bag_score = np.sqrt(metrics.mean_squared_error(labels_val, bag_cv))
        print('bag ', j, '- rmse:', bag_score)

    pred_cv /= nbags
    

    pred_test/= nbags
    fold_score = np.sqrt(metrics.mean_squared_error(labels_val, pred_cv))
    print('Fold ', i, '- rmse:', fold_score)
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = ["deal_probability"]
    pred_cv["item_id"] = X_val.item_id.values
    pred_cv = pred_cv[['item_id','deal_probability']]
    pred_cv['deal_probability'] = pred_cv['deal_probability'].clip(0.0, 1.0)
    sub_valfile = inDir+'/submissions/Prav.nn05.fold' + str(i) + '.csv'    
    pred_cv.to_csv(sub_valfile, index=False)
    
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["deal_probability"]
    pred_test["item_id"] = test_df.item_id.values
    pred_test = pred_test[['item_id','deal_probability']]
    pred_test['deal_probability'] = pred_test['deal_probability'].clip(0.0, 1.0)
    sub_file = inDir+'/submissions/Prav.nn05.fold' + str(i) + '-test' + '.csv'
    pred_test.to_csv(sub_file, index=False)
    del pred_cv
    del pred_test



def full_lstmnet(i):
    print('Full Processing')
   

    data_1_train = data_1
#    data_2_train = data_2

    train_data_reg_train  = train_data_reg
    train_data_city_train  = train_data_city
    train_data_cat1_train  = train_data_cat1
    train_data_cat2_train  = train_data_cat2
    train_data_prm1_train  = train_data_prm1
    train_data_prm2_train  = train_data_prm2
#    train_data_prm3_train  = train_data_prm3
#    train_data_sqnm_train  = train_data_sqnm
    train_data_usr_train  = train_data_usr
    train_data_itype_train  = train_data_itype
    train_user_features_train = train_user_features
    
    labels_train = labels
    
    pred_test = np.zeros(test_df.shape[0])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        model = lstm_model()
#        Enable this and model remove after getting best epoch value
#        callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=1)]
#        model.fit(X_train.values, X_trainy, nb_epoch=epochs, batch_size=batchsize,  verbose=verboselog,validation_data=(X_valid.values,X_validy),callbacks=callbacks)
#        model.load_weights('C:/Users/SriPrav/Documents/R/23Quora/nn2_question_pairs_weights.h5')
        model.fit([data_1_train, train_data_reg_train,train_data_city_train,train_data_cat1_train,train_data_cat2_train,train_data_prm1_train,train_data_prm2_train,train_data_usr_train,train_data_itype_train,train_user_features_train], labels_train, 
                  batch_size=batchsize, 
                  nb_epoch=fullepochs,
                  verbose=verboselog
                 
                 )
        
        pred_test += model.predict([test_data_1, test_data_reg,test_data_city,test_data_cat1,test_data_cat2,test_data_prm1,test_data_prm2,test_data_usr,test_data_itype,test_user_features], batch_size=batchsize, verbose=1)[:,0]
              
    pred_test/= nbags
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["deal_probability"]
    pred_test["item_id"] = test_df.item_id.values
    pred_test = pred_test[['item_id','deal_probability']]
    pred_test['deal_probability'] = pred_test['deal_probability'].clip(0.0, 1.0)
    sub_file = inDir+'/submissions/Prav.nn05.full' + '.csv'
    pred_test.to_csv(sub_file, index=False)
    
model_run = 'fast_test'

if __name__ == '__main__':
    if model_run == 'fast_test':
        i = 1
        lstmnet(i)
        full_lstmnet(nbags)
    else:
        for i in range(1, folds+1):
            lstmnet(i)
        full_lstmnet(nbags)
    

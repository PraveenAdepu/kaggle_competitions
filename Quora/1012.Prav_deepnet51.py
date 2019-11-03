'''
Single model may achieve LB scores at around 0.29+ ~ 0.30+
Average ensembles can easily get 0.28+ or less
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

from collections import defaultdict

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

from sklearn.preprocessing import StandardScaler


########################################
## set directories and parameters
########################################

inDir = 'C:/Users/SriPrav/Documents/R/23Quora'
EMBEDDING_FILE = inDir + '/preModels/GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE =inDir + '/input/train.csv'
TEST_DATA_FILE = inDir + '/input/test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train_df.shape) # (404290, 6)
print(test_df.shape)  # (2345796, 3)

cv_file = inDir + "/CVSchema/Prav_CVindices_5folds.csv"
CV_Schema = pd.read_csv(cv_file)

train_df = pd.merge(train_df, CV_Schema, how = 'left', on = ['id','qid1','qid2'])


########################################
## index word vectors
########################################
print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

#print('Indexing word vectors')
#
#embeddings_index = {}
#f = open(EMBEDDING_FILE)
#count = 0
#for line in f:
#    if count == 0:
#        count = 1
#        continue
#    values = line.split()
#    word = values[0]
#    coefs = np.asarray(values[1:], dtype='float32')
#    embeddings_index[word] = coefs
#f.close()
#
#print('Found %d word vectors of glove.' % len(embeddings_index))


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
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)
    
texts_1 = train_df['question1'].apply(lambda x: text_to_wordlist(str(x)))
texts_2 = train_df['question2'].apply(lambda x: text_to_wordlist(str(x)))

test_texts_1 = test_df['question1'].apply(lambda x: text_to_wordlist(str(x)))
test_texts_2 = test_df['question2'].apply(lambda x: text_to_wordlist(str(x)))

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
labels = np.array(train_df.is_duplicate.values)
train_ids = np.array(train_df.id.values)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_df.test_id.values)

########################################
## generate leaky features
########################################



train_features_06 = inDir + "/input/train_features_06.csv"
test_features_06 = inDir + "/input/test_features_06.csv"
train_features_06 = pd.read_csv(train_features_06)
test_features_06 = pd.read_csv(test_features_06)
print(train_features_06.shape) # (404290, 36)
print(test_features_06.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_06, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_06, how = 'left', on = 'test_id')

train_features_07 = inDir + "/input/train_features_07.csv"
test_features_07 = inDir + "/input/test_features_07.csv"
train_features_07 = pd.read_csv(train_features_07)
test_features_07 = pd.read_csv(test_features_07)
print(train_features_07.shape) # (404290, 36)
print(test_features_07.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_07, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_07, how = 'left', on = 'test_id')

train_features_30 = inDir + "/input/train_features_30.csv"
test_features_30 = inDir + "/input/test_features_30.csv"
train_features_30 = pd.read_csv(train_features_30)
test_features_30 = pd.read_csv(test_features_30)
print(train_features_30.shape) # (404290, 36)
print(test_features_30.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_30, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_30, how = 'left', on = 'test_id')

train_features_32 = inDir + "/input/train_features_32.csv"
test_features_32 = inDir + "/input/test_features_32.csv"
train_features_32 = pd.read_csv(train_features_32)
test_features_32 = pd.read_csv(test_features_32)
print(train_features_32.shape) # (404290, 36)
print(test_features_32.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_32, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_32, how = 'left', on = 'test_id')


train_features_11 = inDir + "/input/train_question_freq_features.csv"
test_features_11 = inDir + "/input/test_question_freq_features.csv"
train_features_11 = pd.read_csv(train_features_11)
test_features_11 = pd.read_csv(test_features_11)
print(train_features_11.shape) # (404290, 36)
print(test_features_11.shape)  # (2345796, 34)

del train_features_11['is_duplicate']
test_features_11.rename(columns={'id': 'test_id'}, inplace=True)

train_df = pd.merge(train_df, train_features_11, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_11, how = 'left', on = 'test_id')

train_df.columns

train_df = train_df.replace(np.inf, np.nan) 
test_df = test_df.replace(np.inf, np.nan)
train_df = train_df.fillna(0)   
test_df = test_df.fillna(0)                    

leaks = train_df[['zbigrams_common_count',
                     'zbigrams_common_ratio',
                     'z_noun_match',
                     'z_match_ratio',
                     'z_word_match',
                     'z_tfidf_word_match',
                     'q1_freq',
                     'q2_freq',
                     'q1_q2_intersect',
                     'max_kcore']]
test_leaks = test_df[['zbigrams_common_count',
                     'zbigrams_common_ratio',
                     'z_noun_match',
                     'z_match_ratio',
                     'z_word_match',
                     'z_tfidf_word_match',
                     'q1_freq',
                     'q2_freq',
                     'q1_q2_intersect',
                     'max_kcore']]

ss = StandardScaler()
ss.fit(np.vstack((leaks, test_leaks)))
leaks = ss.transform(leaks)
test_leaks = ss.transform(test_leaks)

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

########################################
## sample train/validation data
########################################
#np.random.seed(1234)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
leaks_train = np.vstack((leaks[idx_train], leaks[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
leaks_val = np.vstack((leaks[idx_val], leaks[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344
    
########################################
## define the model structure
########################################
#embedding_layer = Embedding(nb_words,
#        EMBEDDING_DIM,
#        weights=[embedding_matrix],
#        input_length=MAX_SEQUENCE_LENGTH,
#        trainable=False)
#lstm_layer = LSTM(num_lstm, dropout_W=rate_drop_lstm, dropout_U=rate_drop_lstm)
#
#sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#embedded_sequences_1 = embedding_layer(sequence_1_input)
#x1 = lstm_layer(embedded_sequences_1)
#
#sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#embedded_sequences_2 = embedding_layer(sequence_2_input)
#y1 = lstm_layer(embedded_sequences_2)
#
#leaks_input = Input(shape=(leaks.shape[1],))
#leaks_dense = Dense(num_dense/2, activation=act)(leaks_input)
#
#
#merged = Merge([x1, y1, leaks_dense], mode='concat')
#merged = Dropout(rate_drop_dense)(merged)
#merged = BatchNormalization()(merged)
#
#merged = Dense(num_dense, activation=act)(merged)
#merged = Dropout(rate_drop_dense)(merged)
#merged = BatchNormalization()(merged)
#
#preds = Dense(1, activation='sigmoid')(merged)   

    
#np.random.seed(1234)


def lstm_model():
    
    model5 = Sequential()
    model5.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,trainable=False))
    model5.add(LSTM(num_lstm, dropout_W=rate_drop_lstm, dropout_U=rate_drop_lstm))
    
    model6 = Sequential()
    model6.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,trainable=False))
    model6.add(LSTM(num_lstm, dropout_W=rate_drop_lstm, dropout_U=rate_drop_lstm))
    
    model7 = Sequential()
#    model7.add(input_dim = leaks.shape[1])
#    model7.add(shape=leaks.shape[1])
#    leaks_input = Input(shape=(leaks.shape[1],))
#    model7.add(Dense(num_dense/2, activation=act))
    model7.add(Dense(num_dense/2, input_dim = leaks.shape[1]))
    model7.add(PReLU())
    
    merged_model = Sequential()
    merged_model.add(Merge([model5, model6,model7], mode='concat'))
    
    merged_model.add(Dropout(rate_drop_dense))
    merged_model.add(BatchNormalization())
    
    merged_model.add(Dense(num_dense))
    merged_model.add(PReLU()) # act
    merged_model.add(Dropout(rate_drop_dense))
    merged_model.add(BatchNormalization())
    
    merged_model.add(Dense(1))
    merged_model.add(Activation('sigmoid'))
    merged_model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return(merged_model)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None
########################################
## add class weight
########################################
########################################
## train the model
########################################
model = lstm_model()

hist = model.fit([data_1_train, data_2_train, leaks_train], labels_train, \
        validation_data=([data_1_val, data_2_val, leaks_val], labels_val, weight_val), \
        nb_epoch=200, verbose=2, batch_size=2048, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

########################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

preds = model.predict([test_data_1, test_data_2, test_leaks], batch_size=8192, verbose=1)
preds += model.predict([test_data_2, test_data_1, test_leaks], batch_size=8192, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)











nbags = 1
folds = 5
epochs = 20
batchsize = 3800
verboselog = 2    

def lstmnet(i):

    print('Fold ', i , ' Processing')
    X_build = train_df[train_df['CVindices'] != i] # 636112
    #X_build = X_build.groupby('id').first().reset_index() # 331085
    X_val   = train_df[train_df['CVindices'] == i]
    
    trainindex = train_df[train_df['CVindices'] != i].index.tolist()
    valindex   = train_df[train_df['CVindices'] == i].index.tolist()

    data_1_train = np.vstack((data_1[trainindex], data_2[trainindex]))
    data_2_train = np.vstack((data_2[trainindex], data_1[trainindex]))
    labels_train = np.concatenate((labels[trainindex], labels[trainindex]))
    
    data_1_val = np.vstack((data_1[valindex], data_2[valindex]))
    data_2_val = np.vstack((data_2[valindex], data_1[valindex]))
    labels_val = np.concatenate((labels[valindex], labels[valindex]))
    
    weight_val = np.ones(len(labels_val))
    if re_weight:
        weight_val *= 0.472001959
        weight_val[labels_val==0] = 1.309028344
       
    pred_cv = np.zeros(X_val.shape[0]*2)
    pred_cv_final = np.zeros(X_val.shape[0])   
    pred_test = np.zeros(test_df.shape[0])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros(X_val.shape[0]*2)
        bag_cv_final = np.zeros(X_val.shape[0])
        model = lstm_model()
#        Enable this and model remove after getting best epoch value
#        callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=1)]
#        model.fit(X_train.values, X_trainy, nb_epoch=epochs, batch_size=batchsize,  verbose=verboselog,validation_data=(X_valid.values,X_validy),callbacks=callbacks)
#        model.load_weights('C:/Users/SriPrav/Documents/R/23Quora/nn2_question_pairs_weights.h5')
        model.fit([data_1_train, data_2_train], labels_train, 
                  validation_data=([data_1_val, data_2_val], labels_val, weight_val), 
                  batch_size=batchsize, 
                  nb_epoch=epochs,
                  verbose=verboselog,
                  class_weight=class_weight
#                 callbacks=[early_stopping, model_checkpoint]
                 )
        bag_cv  += model.predict([data_1_val, data_2_val], batch_size=batchsize, verbose=1)[:,0]
        bag_cv_final += (bag_cv[:X_val.shape[0]] + bag_cv[X_val.shape[0]:] )/2
        pred_cv += model.predict([data_1_val, data_2_val], batch_size=batchsize, verbose=1)[:,0]
        
        pred_test += model.predict([test_data_1, test_data_2], batch_size=batchsize, verbose=1)[:,0]
        pred_test += model.predict([test_data_2, test_data_1], batch_size=batchsize, verbose=1)[:,0]
        bag_score = log_loss(labels_val[:X_val.shape[0]], bag_cv_final)
        print('bag ', j, '- logloss:', bag_score)
#        os.remove('C:/Users/SriPrav/Documents/R/23Quora/dn50_question_pairs_weights.h5')
    pred_cv /= nbags
    pred_cv_final+=(pred_cv[:X_val.shape[0]] + pred_cv[X_val.shape[0]:])/2 

    pred_test/= 2*nbags
    fold_score = log_loss(labels_val[:X_val.shape[0]], pred_cv_final)
    print('Fold ', i, '- logloss:', fold_score)
    pred_cv_final = pd.DataFrame(pred_cv_final)
    pred_cv_final.columns = ["is_duplicate"]
    pred_cv_final["id"] = X_val.id.values
    pred_cv_final = pred_cv_final[['id','is_duplicate']]
    sub_valfile = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.dn51.fold' + str(i) + '.csv'    
    pred_cv_final.to_csv(sub_valfile, index=False)
    
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["is_duplicate"]
    pred_test["test_id"] = test_df.test_id.values
    pred_test = pred_test[['test_id','is_duplicate']]
    sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.dn51.fold' + str(i) + '-test' + '.csv'
    pred_test.to_csv(sub_file, index=False)
    del pred_cv
    del pred_cv_final
    del pred_test
    del weight_val

fullepochs = 22
def full_lstmnet(i):
    print('Full Processing')
   

    data_1_train = np.vstack((data_1, data_2))
    data_2_train = np.vstack((data_2, data_1))
    labels_train = np.concatenate((labels, labels))
    
    pred_test = np.zeros(test_df.shape[0])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        model = lstm_model()
#        Enable this and model remove after getting best epoch value
#        callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=1)]
#        model.fit(X_train.values, X_trainy, nb_epoch=epochs, batch_size=batchsize,  verbose=verboselog,validation_data=(X_valid.values,X_validy),callbacks=callbacks)
#        model.load_weights('C:/Users/SriPrav/Documents/R/23Quora/nn2_question_pairs_weights.h5')
        model.fit([data_1_train, data_2_train], labels_train, 
                  batch_size=batchsize, 
                  nb_epoch=fullepochs,
                  verbose=verboselog,
                  class_weight=class_weight
#                 callbacks=[early_stopping, model_checkpoint]
                 )
        
        pred_test += model.predict([test_data_1, test_data_2], batch_size=batchsize, verbose=1)[:,0]
        pred_test += model.predict([test_data_2, test_data_1], batch_size=batchsize, verbose=1)[:,0]       
    pred_test/= 2*nbags
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["is_duplicate"]
    pred_test["test_id"] = test_df.test_id.values
    pred_test = pred_test[['test_id','is_duplicate']]
    sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.dn51.full.csv'
    pred_test.to_csv(sub_file, index=False)
    
nbags = 1
if __name__ == '__main__':
    for i in range(1, folds+1):
        lstmnet(i)
    full_lstmnet(nbags)
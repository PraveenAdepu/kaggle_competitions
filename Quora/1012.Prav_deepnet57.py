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

train_file = inDir + "/input/TrainingSet01.csv"
test_file = inDir + "/input/TestingSet01.csv"
train_features_01 = pd.read_csv(train_file)
train_features_01 = train_features_01.groupby('id').first().reset_index()

test_features_01 = pd.read_csv(test_file)
print(train_features_01.shape) # (404290, 6)
print(test_features_01.shape)  # (2345796, 3)

train_df = pd.merge(train_df, train_features_01, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_01, how = 'left', on = 'test_id')



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


train_features_30 = inDir + "/input/train_features_35.csv"
test_features_30 = inDir + "/input/test_features_35.csv"
train_features_30 = pd.read_csv(train_features_30)
test_features_30 = pd.read_csv(test_features_30)
print(train_features_30.shape) # (404290, 36)
print(test_features_30.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_30, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_30, how = 'left', on = 'test_id')

train_features_31 = inDir + "/input/train_features_31.csv"
test_features_31 = inDir + "/input/test_features_31.csv"
train_features_31 = pd.read_csv(train_features_31)
test_features_31 = pd.read_csv(test_features_31)
print(train_features_31.shape) # (404290, 36)
print(test_features_31.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_31, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_31, how = 'left', on = 'test_id')

train_features_32 = inDir + "/input/train_features_32.csv"
test_features_32 = inDir + "/input/test_features_32.csv"
train_features_32 = pd.read_csv(train_features_32)
test_features_32 = pd.read_csv(test_features_32)
print(train_features_32.shape) # (404290, 36)
print(test_features_32.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_32, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_32, how = 'left', on = 'test_id')


train_features_33 = inDir + "/input/train_features_33.csv"
test_features_33 = inDir + "/input/test_features_33.csv"
train_features_33 = pd.read_csv(train_features_33)
test_features_33 = pd.read_csv(test_features_33)
print(train_features_33.shape) # (404290, 36)
print(test_features_33.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_33, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_33, how = 'left', on = 'test_id')

#train_features_34 = inDir + "/input/train_features_34.csv"
#test_features_34 = inDir + "/input/test_features_34.csv"
#train_features_34 = pd.read_csv(train_features_34)
#test_features_34 = pd.read_csv(test_features_34)
#print(train_features_34.shape) # (404290, 36)
#print(test_features_34.shape)  # (2345796, 34)
#
#train_df = pd.merge(train_df, train_features_34, how = 'left', on = 'id')
#test_df = pd.merge(test_df, test_features_34, how = 'left', on = 'test_id')



train_features_52 = inDir + "/input/train_features_52.csv"
test_features_52 = inDir + "/input/test_features_52.csv"
train_features_52 = pd.read_csv(train_features_52)
test_features_52 = pd.read_csv(test_features_52)
print(train_features_52.shape) # (404290, 36)
print(test_features_52.shape)  # (2345796, 34)

train_df = pd.merge(train_df, train_features_52, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_52, how = 'left', on = 'test_id')

train_df.columns

train_df = train_df.replace(np.inf, np.nan) 
test_df = test_df.replace(np.inf, np.nan)
train_df = train_df.fillna(0)   
test_df = test_df.fillna(0)                    

leaks = train_df[['q_dist_soundex'
                        ,'q_dist_jarowinkler'
                        ,'q_dist_lcs'
                        ,'q1_nchar'
                        ,'q2_nchar'
                        ,'q1_EndsWith_q2'
                        ,'q1_EndsWith_Sound_q2'
                        ,'q_nchar_ratios_pmax'
                        ,'q_nchar_pmin'
                        ,'q_nchar_pmax'
                        ,'q1_StartsWith_Sound_q2'
                        ,'q1_StartsWith_q2'
                        ,'q1_nwords'
                        ,'q1_nwords_matched_q2'
                        ,'q1_MatchedWords_ratio_to_q2'
                        ,'q2_nwords'
                        ,'q2_nwords_matched_q1'
                        ,'q2_MatchedWords_ratio_to_q1'
                        ,'q_MatchedWords_ratio_to_q_ratios_pmax'
                        ,'q_MatchedWords_ratio_to_q_pmin'
                        ,'q_MatchedWords_ratio_to_q_pmax'
                        ,'q_2gram_jaccard'
                        ,'q_3gram_jaccard'
                        ,'q_4gram_jaccard'
                        ,'q_5gram_jaccard'
                        ,'q_dist_lv'
                        ,'q_dist_cosine'
                        ,'q1_PunctCount'
                        ,'q2_PunctCount'
                        ,'q_PunctCount_ratios_pmax'
                        ,'q_PunctCount_pmin'
                        ,'q_PunctCount_pmax'
                        ,'len_q1'
                        ,'len_q2'
                        ,'diff_len'
                        ,'len_char_q1'
                        ,'len_char_q2'
                        ,'len_word_q1'
                        ,'len_word_q2'
                        ,'common_words'
                        ,'fuzz_qratio'
                        ,'fuzz_WRatio'
                        ,'fuzz_partial_ratio'
                        ,'fuzz_partial_token_set_ratio'
                        ,'fuzz_partial_token_sort_ratio'
                        ,'fuzz_token_set_ratio'
                        ,'fuzz_token_sort_ratio'
                        ,'word_match'
                        ,'tfidf_word_match'
                        ,'cosine_distance'
                        ,'cityblock_distance'
                        ,'jaccard_distance'
                        ,'canberra_distance'
                        ,'euclidean_distance'
                        ,'minkowski_distance'
                        ,'braycurtis_distance'
                        ,'skew_q1vec'
                        ,'skew_q2vec'
                        ,'kur_q1vec'
                        ,'kur_q2vec',
                     'zbigrams_common_count',
                     'zbigrams_common_ratio',
                     'z_noun_match',
                     'z_match_ratio',
                     'z_word_match',                    
                     'q1_freq',
                     'q2_freq',"q1_q2_intersect","q1_q2_wm_ratio" ,"z_place_match_num","z_place_mismatch_num","qid1_max_kcore"
                     ,"qid2_max_kcore","max_kcore","m_q1_q2_tf_svd0_q1","m_q1_q2_tf_svd0_q2", "wmd" ,"norm_wmd"
                     ]]
                     
test_leaks = test_df[['q_dist_soundex'
                        ,'q_dist_jarowinkler'
                        ,'q_dist_lcs'
                        ,'q1_nchar'
                        ,'q2_nchar'
                        ,'q1_EndsWith_q2'
                        ,'q1_EndsWith_Sound_q2'
                        ,'q_nchar_ratios_pmax'
                        ,'q_nchar_pmin'
                        ,'q_nchar_pmax'
                        ,'q1_StartsWith_Sound_q2'
                        ,'q1_StartsWith_q2'
                        ,'q1_nwords'
                        ,'q1_nwords_matched_q2'
                        ,'q1_MatchedWords_ratio_to_q2'
                        ,'q2_nwords'
                        ,'q2_nwords_matched_q1'
                        ,'q2_MatchedWords_ratio_to_q1'
                        ,'q_MatchedWords_ratio_to_q_ratios_pmax'
                        ,'q_MatchedWords_ratio_to_q_pmin'
                        ,'q_MatchedWords_ratio_to_q_pmax'
                        ,'q_2gram_jaccard'
                        ,'q_3gram_jaccard'
                        ,'q_4gram_jaccard'
                        ,'q_5gram_jaccard'
                        ,'q_dist_lv'
                        ,'q_dist_cosine'
                        ,'q1_PunctCount'
                        ,'q2_PunctCount'
                        ,'q_PunctCount_ratios_pmax'
                        ,'q_PunctCount_pmin'
                        ,'q_PunctCount_pmax'
                        ,'len_q1'
                        ,'len_q2'
                        ,'diff_len'
                        ,'len_char_q1'
                        ,'len_char_q2'
                        ,'len_word_q1'
                        ,'len_word_q2'
                        ,'common_words'
                        ,'fuzz_qratio'
                        ,'fuzz_WRatio'
                        ,'fuzz_partial_ratio'
                        ,'fuzz_partial_token_set_ratio'
                        ,'fuzz_partial_token_sort_ratio'
                        ,'fuzz_token_set_ratio'
                        ,'fuzz_token_sort_ratio'
                        ,'word_match'
                        ,'tfidf_word_match'
                        ,'cosine_distance'
                        ,'cityblock_distance'
                        ,'jaccard_distance'
                        ,'canberra_distance'
                        ,'euclidean_distance'
                        ,'minkowski_distance'
                        ,'braycurtis_distance'
                        ,'skew_q1vec'
                        ,'skew_q2vec'
                        ,'kur_q1vec'
                        ,'kur_q2vec',
                     'zbigrams_common_count',
                     'zbigrams_common_ratio',
                     'z_noun_match',
                     'z_match_ratio',
                     'z_word_match',                    
                     'q1_freq',
                     'q2_freq',"q1_q2_intersect","q1_q2_wm_ratio" ,"z_place_match_num","z_place_mismatch_num","qid1_max_kcore"
                     ,"qid2_max_kcore","max_kcore","m_q1_q2_tf_svd0_q1","m_q1_q2_tf_svd0_q2", "wmd" ,"norm_wmd"
                    ]]

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









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


pos_train = train_df[train_df['is_duplicate'] == 1]
neg_train = train_df[train_df['is_duplicate'] == 0]

# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = ((len(pos_train)*1.0 / (len(pos_train) + len(neg_train))*1.0) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train)*1.0 / (len(pos_train) + len(neg_train))*1.0)

x_train = pd.concat([pos_train, neg_train]) # (780486,6)
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train

tk = text.Tokenizer(nb_words=200000)

max_len = 40
tk.fit_on_texts(list(x_train.question1.values) + list(x_train.question2.values.astype(str)) + list(test_df.question1.values.astype(str)) + list(test_df.question2.values.astype(str)) )
x1 = tk.texts_to_sequences(x_train.question1.values)
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tk.texts_to_sequences(x_train.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

x11 = tk.texts_to_sequences(test_df.question1.values.astype(str))
x11 = sequence.pad_sequences(x11, maxlen=max_len)

x22 = tk.texts_to_sequences(test_df.question2.values.astype(str))
x22 = sequence.pad_sequences(x22, maxlen=max_len)

train_q1 = pd.DataFrame(x1)
train_q1.shape
train_q2 = pd.DataFrame(x2)
train_q2.shape

list(train_q1.columns.values)

test_q1 = pd.DataFrame(x11)
test_q2 = pd.DataFrame(x22)

train_q1.columns = [str(col) + '_q1' for col in train_q1.columns]
train_q2.columns = [str(col) + '_q2' for col in train_q2.columns]

test_q1.columns = [str(col) + '_q1' for col in test_q1.columns]
test_q2.columns = [str(col) + '_q2' for col in test_q2.columns]
              
x_train = data                 
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

train_features_01 = inDir + "/input/train_features_01.csv"
test_features_01 = inDir + "/input/test_features_01.csv"
train_features_01 = pd.read_csv(train_features_01)
test_features_01 = pd.read_csv(test_features_01)
print(train_features_01.shape) # (404290, 36)
print(test_features_01.shape)  # (2345796, 34)

x_train = pd.merge(x_train, train_features_01, how = 'left', on = ['id','qid1','qid2'])

test_features_01.rename(columns={'id': 'test_id'}, inplace=True)

test_df = pd.merge(test_df, test_features_01, how = 'left', on = 'test_id')

train_features_02 = inDir + "/input/train_features_02.csv"
test_features_02 = inDir + "/input/test_features_02.csv"
train_features_02 = pd.read_csv(train_features_02)
test_features_02 = pd.read_csv(test_features_02)
print(train_features_02.shape) # (404290, 36)
print(test_features_02.shape)  # (2345796, 34)

x_train = pd.merge(x_train, train_features_02, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_02, how = 'left', on = 'test_id')

train_features_03 = inDir + "/input/train_features_03.csv"
test_features_03 = inDir + "/input/test_features_03.csv"
train_features_03 = pd.read_csv(train_features_03)
test_features_03 = pd.read_csv(test_features_03)
print(train_features_03.shape) # (404290, 36)
print(test_features_03.shape)  # (2345796, 34)

x_train = pd.merge(x_train, train_features_03, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_03, how = 'left', on = 'test_id')

train_features_04 = inDir + "/input/train_features_04.csv"
test_features_04 = inDir + "/input/test_features_04.csv"
train_features_04 = pd.read_csv(train_features_04)
test_features_04 = pd.read_csv(test_features_04)
print(train_features_04.shape) # (404290, 36)
print(test_features_04.shape)  # (2345796, 34)

x_train = pd.merge(x_train, train_features_04, how = 'left', on = 'id')
test_df = pd.merge(test_df, test_features_04, how = 'left', on = 'test_id')

x_train = x_train.fillna(0)   
test_df = test_df.fillna(0)

#############################################################################################################################

X_training = pd.concat([x_train, train_q1,train_q2], axis=1)
X_testing = pd.concat([test_df, test_q1,test_q2], axis=1) 

X_training.shape
X_testing.shape

sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/TrainingSet01.csv'
X_training.to_csv(sub_file, index=False)

sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/TestingSet01.csv'
X_testing.to_csv(sub_file, index=False)

y_train_labels = pd.DataFrame(y_train)

y_train_labels.columns = [str(col) + '_isDuplicate' for col in y_train_labels.columns]

sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/TrainingSetLabel.csv'
y_train_labels.to_csv(sub_file, index=False)

############################################################################################################################

features_to_use = cols = [col for col in X_training.columns if col not in ['id','qid1','qid2','question1', 'question2','is_duplicate','CVindices','test_id']] 

############################################################################################################################


X_process = X_training[features_to_use]

X_process = X_process.fillna(0)

from sklearn.cross_validation import train_test_split
x_build, x_valid, y_build, y_valid = train_test_split(X_process, y, test_size=0.2, random_state=4242)


sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/TrainingSetBuild01.csv'
x_build.to_csv(sub_file, index=False)

sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/TestingSetValid01.csv'
x_valid.to_csv(sub_file, index=False)

y_build_labels = pd.DataFrame(y_build)
y_build_labels.columns = [str(col) + '_isDuplicate' for col in y_build_labels.columns]

sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/TrainingSetBuildLabel.csv'
y_build_labels.to_csv(sub_file, index=False)

y_valid_labels = pd.DataFrame(y_valid)
y_valid_labels.columns = [str(col) + '_isDuplicate' for col in y_valid_labels.columns]

sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/TrainingSetValidLabel.csv'
y_valid_labels.to_csv(sub_file, index=False)

# 624388 , 156098
import xgboost as xgb

#params = {}
#params['objective'] = 'binary:logistic'
#params['eval_metric'] = 'logloss'
#params['eta'] = 0.02
#params['max_depth'] = 4

param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.02
param['max_depth'] = 4
param['silent'] = 1
param['eval_metric'] = "logloss"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 25
param['seed'] = 2017
param['print_every_n'] = 50
num_rounds = 10000
plst = list(param.items())


d_train = xgb.DMatrix(x_build, label=y_build)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

#bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
model = xgb.train(plst, 
                      d_train, 
                      num_rounds, 
                      watchlist, 
                      verbose_eval = 100 ,
                      early_stopping_rounds=20
                      )

#d_test = xgb.DMatrix(x_test)
#p_test = model.predict(d_test)
#
#sub = pd.DataFrame()
#sub['test_id'] = test_df['test_id']
#sub['is_duplicate'] = p_test
#sub.to_csv('./submissions/Prav_xgb04.csv', index=False)
##########################################################################################
# Full model training
########################################################################################## 

fullnum_rounds = int(num_rounds * 1.2)

def fulltrain_xgboost(bags):
    xgbtrain = xgb.DMatrix( x_train, label=y_train)
    watchlist = [ (xgbtrain,'train') ]
    fullmodel = xgb.train(plst, 
                              xgbtrain, 
                              fullnum_rounds, 
                              watchlist,
                              verbose_eval = 100,
                              )
    xgtest = xgb.DMatrix(x_test)
    predfull_test = fullmodel.predict(xgtest)
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["is_duplicate"]
    predfull_test["test_id"] = test_df.test_id.values
    predfull_test = predfull_test[['test_id','is_duplicate']]
    sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.xgb04.full' + '.csv'
    predfull_test.to_csv(sub_file, index=False)
    
folds = 5
if __name__ == '__main__':
    #for i in range(1, folds+1):
        #train_xgboost(i)
    fulltrain_xgboost(folds)
    
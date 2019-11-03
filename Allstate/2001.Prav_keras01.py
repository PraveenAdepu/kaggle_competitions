# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 13:00:55 2016

@author: PAdepu
"""

## import libraries
import numpy as np
np.random.seed(123)

import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU

## Batch generators ##################################################################################################################################

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

########################################################################################################################################################

## read data
train            = pd.read_csv('C:/Users/padepu/Documents/R/14Allstate/input/train.csv')
test             = pd.read_csv('C:/Users/padepu/Documents/R/14Allstate/input/test.csv')
CVindices_5folds = pd.read_csv( 'C:/Users/padepu/Documents/R/14Allstate/CVSchema/Prav_CVindices_5folds.csv' )

train = pd.merge( train, CVindices_5folds, on=['id'] , how='left'  )
## set test loss to NaN
test['loss']      = -100
test['CVindices'] = 0

train_test = pd.concat((train, test), axis = 0)

featureReference = [col for col in train_test.columns if col  in ['id', 'CVindices','loss']]

train_test_referenceCols = train_test[featureReference]

featureCols = [col for col in train_test.columns if col  not in ['id', 'CVindices','loss']]

train_test_1 = train_test[featureCols]

list(train_test.columns.values)


sparse_data = []

f_cat = [f for f in train_test_1.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(train_test_1[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in train_test_1.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(train_test_1[f_num]))
sparse_data.append(tmp)

xtr_te = hstack(sparse_data, format = 'csr')

train_test_ref = csr_matrix(train_test_referenceCols)

train_test_ref1 = hstack(train_test_ref, sparse_data)
train_test_ref = train_test_ref.append(sparse_data)

train_test_spa
del(tr_te, train, test)

f_cat = [f for f in train_test.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(train_test[f].astype('category'))
    
cols = [c for c in train_test.columns if  'cat' not in c]


train_testOriginal = train_test[cols]

train_testOhe = pd.concat((train_testOriginal, dummy),axis = 1)

f_num = [f for f in train_test.columns if 'cont' in f]
scaler = StandardScaler()
ContScaler = pd.DataFrame(scaler.fit_transform(train_test[f_num]), columns=f_num)

cols = [c for c in train_testOhe.columns if  'cont' not in c]
train_testOhe = train_testOhe[cols]

# list(train_testOheAll.columns.values)
train_testOheAll =  train_testOhe.join(ContScaler)

trainingSet = train_testOheAll[train_testOheAll['CVindices'] != 0]
testingSet  = train_testOheAll[train_testOheAll['CVindices'] == 0]


## neural net
def nn_model():
    model = Sequential()
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(200, init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)

featurenames = [col for col in trainingSet.columns if col not in ['id', 'CVindices','loss']]
#df2 = df[cols]

nepochs = 10 
# Fold 1    
i = 5    
X_build = trainingSet[trainingSet['CVindices'] != i]  
X_val   = trainingSet[trainingSet['CVindices'] == i]  
#X_build[featurenames], X_build['loss']
#xtrain.shape
xtrain = csr_matrix( X_build[featurenames] )
y = csr_matrix(X_build['loss'])
model = nn_model()
fit = model.fit(xtrain, y, nb_epoch=10, batch_size=10,  verbose=1)
fit = model.fit_generator(generator = batch_generator(xtrain, y, 10, True),
                          nb_epoch = nepochs,
                          samples_per_epoch = xtrain.shape[0],
                          verbose = 1)
pred = model.predict_generator(generator = batch_generatorp(X_val[featurenames], 800, False), val_samples = X_val[featurenames].shape[0])[:,0]

## response and IDs
y = train['loss'].values
id_train = train['id'].values
id_test = test['id'].values

## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)

## Preprocessing and transforming to sparse data
sparse_data = []

f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data, tmp)

## neural net
def nn_model():
    model = Sequential()
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(200, init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)

## cv-folds
nfolds = 5
folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)

## train models
i = 0
nbags = 5
nepochs = 75
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        model = nn_model()
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0],
                                  verbose = 0)
        pred += model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0]
        pred_test += model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0]
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(yte, pred)
    i += 1
    print('Fold ', i, '- MAE:', score)

print('Total - MAE:', mean_absolute_error(y, pred_oob))

## train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
df.to_csv('C:/Users/padepu/Documents/R/14Allstate/submissions/preds_oob_keras.csv', index = False)

## test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv('C:/Users/padepu/Documents/R/14Allstate/submissions/submission_keras.csv', index = False)

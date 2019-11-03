# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:57:13 2016

@author: PAdepu
"""

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
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler # Extra
import itertools

train            = pd.read_csv('C:/Users/SriPrav/Documents/R/14Allstate/input/prav.L2train.csv')
test             = pd.read_csv('C:/Users/SriPrav/Documents/R/14Allstate/input/prav.L2test.csv')

featureCols = [col for col in train.columns if col  not in ['id', 'CVindices','loss','et20loss']]
X_trainFull  = train[featureCols]
X_trainFully = np.log(train['loss'].values+200)
X_test = test[featureCols]

def nn_model():
    model = Sequential()
    
    model.add(Dense(9, input_dim = X_train.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization()) 
    model.add(Dropout(0.2))
    
#    model.add(Dense(7, init = 'he_normal'))
#    model.add(PReLU())
#    model.add(BatchNormalization()) 
#    model.add(Dropout(0.1))
    
    model.add(Dense(4, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
#    model.add(Dropout(0.1))
    
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)
    
nbags = 1
folds = 5
epochs = 40
batchsize = 128
verboselog = 0

for i in range(1, folds+1):
    print('Fold ', i , ' Processing')
    X_build = train[train['CVindices'] != i]
    X_val   = train[train['CVindices'] == i]
    
    X_train = X_build[featureCols]
    X_valid = X_val[featureCols]
    
    X_trainy = np.log(X_build['loss']+200)
    X_validy = np.log(X_val['loss']+200)
        
    pred_cv = np.zeros(X_validy.shape[0])
    pred_test = np.zeros(X_test.shape[0])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros(X_validy.shape[0])
        model = nn_model()
        fit      = model.fit(X_train.values, X_trainy, nb_epoch=epochs, batch_size=batchsize,  verbose=verboselog,validation_data=(X_valid.values,X_validy))
        bag_cv   = np.exp(model.predict(X_valid.values)[:,0])-200
        pred_cv += np.exp(model.predict(X_valid.values)[:,0])-200
        pred_test += np.exp(model.predict(X_test.values)[:,0])-200
        bag_score = mean_absolute_error(np.exp(X_validy)-200, bag_cv)
        print('bag ', j, '- MAE:', bag_score)
    pred_cv /= nbags
    pred_test/= nbags
    fold_score = mean_absolute_error(np.exp(X_validy)-200, pred_cv)
    print('Fold ', i, '- MAE:', fold_score)
    
    pred_cv_df = pd.DataFrame({'id': X_val['id'], 'loss': pred_cv})
    pred_test_df = pd.DataFrame({'id': test['id'], 'loss': pred_test})
    
#    if i == 1:  
#        pred_cv_df.to_csv('C:/Users/SriPrav/Documents/R/14Allstate/submissions/prav.keras20.fold1.csv', index = False)   
#        pred_test_df.to_csv('C:/Users/SriPrav/Documents/R/14Allstate/submissions/prav.keras20.fold1-test.csv', index = False)
#    if i == 2:  
#        pred_cv_df.to_csv('C:/Users/SriPrav/Documents/R/14Allstate/submissions/prav.keras20.fold2.csv', index = False)   
#        pred_test_df.to_csv('C:/Users/SriPrav/Documents/R/14Allstate/submissions/prav.keras20.fold2-test.csv', index = False)
#    if i == 3:  
#        pred_cv_df.to_csv('C:/Users/SriPrav/Documents/R/14Allstate/submissions/prav.keras20.fold3.csv', index = False)   
#        pred_test_df.to_csv('C:/Users/SriPrav/Documents/R/14Allstate/submissions/prav.keras20.fold3-test.csv', index = False)
#    if i == 4:  
#        pred_cv_df.to_csv('C:/Users/SriPrav/Documents/R/14Allstate/submissions/prav.keras20.fold4.csv', index = False)   
#        pred_test_df.to_csv('C:/Users/SriPrav/Documents/R/14Allstate/submissions/prav.keras20.fold4-test.csv', index = False)
#    if i == 5:  
#        pred_cv_df.to_csv('C:/Users/SriPrav/Documents/R/14Allstate/submissions/prav.keras20.fold5.csv', index = False)   
#        pred_test_df.to_csv('C:/Users/SriPrav/Documents/R/14Allstate/submissions/prav.keras20.fold5-test.csv', index = False)




#def nn_model():
#    model = Sequential()
#    
#    model.add(Dense(8, input_dim = X_trainFull.shape[1], init = 'he_normal'))
#    model.add(PReLU())
#    model.add(BatchNormalization()) 
#    model.add(Dropout(0.1))
#    
#    model.add(Dense(6, init = 'he_normal'))
#    model.add(PReLU())
#    model.add(BatchNormalization()) 
#    model.add(Dropout(0.05))
#    
#    model.add(Dense(4, init = 'he_normal'))
#    model.add(PReLU())
#    model.add(BatchNormalization())    
#    model.add(Dropout(0.05))
#    
#    model.add(Dense(1, init = 'he_normal'))
#    model.compile(loss = 'mae', optimizer = 'adadelta')
#    return(model)

def nn_model():
    model = Sequential()
    
    model.add(Dense(8, input_dim = X_trainFull.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization()) 
    model.add(Dropout(0.1))
    
#    model.add(Dense(7, init = 'he_normal'))
#    model.add(PReLU())
#    model.add(BatchNormalization()) 
#    model.add(Dropout(0.1))
    
    model.add(Dense(4, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.1))
    
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)
    
pred_fulltest = np.zeros(X_test.shape[0])
print('Full model training')
for j in range(1,nbags+1):
    print('bag ', j , ' Processing')        
    model = nn_model()
    fit      = model.fit(X_trainFull.values, X_trainFully, nb_epoch=epochs, batch_size=batchsize,  verbose=verboselog)
    pred_fulltest += np.exp(model.predict(X_test.values)[:,0])-200
pred_fulltest/= nbags
pred_fulltest_df = pd.DataFrame({'id': test['id'], 'loss': pred_fulltest})    
pred_fulltest_df.to_csv('C:/Users/SriPrav/Documents/R/14Allstate/submissions/prav.L2.keras30_9.csv', index = False)


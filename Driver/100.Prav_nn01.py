# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 21:46:08 2017

@author: SriPrav
"""

import numpy as np
np.random.seed(2017)
import pandas as pd
from datetime import datetime
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras.wrappers.scikit_learn import KerasClassifier

import os
import sys
import operator

class roc_auc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_proba(self.x, verbose=0)
        roc = roc_auc_score(self.y, y_pred)
        logs['roc_auc'] = roc_auc_score(self.y, y_pred)
        logs['norm_gini'] = ( roc_auc_score(self.y, y_pred) * 2 ) - 1

        y_pred_val = self.model.predict_proba(self.x_val, verbose=0)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        logs['roc_auc_val'] = roc_auc_score(self.y_val, y_pred_val)
        logs['norm_gini_val'] = ( roc_auc_score(self.y_val, y_pred_val) * 2 ) - 1

        print('\rroc_auc: %s - roc_auc_val: %s - norm_gini: %s - norm_gini_val: %s' % (str(round(roc,5)),str(round(roc_val,5)),str(round((roc*2-1),5)),str(round((roc_val*2-1),5))), end=10*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))

def scale_data(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

inDir = 'C:\\Users\\SriPrav\\Documents\\R\\36Driver'

MODEL_WEIGHTS_FILE = inDir + '/nn01_weights.h5'

# train and test data path
DATA_TRAIN_PATH = inDir +'/input/train.csv'
DATA_TEST_PATH = inDir+'/input/test.csv'

def load_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH):
    train_loader = pd.read_csv(path_train, dtype={'target': np.int8, 'id': np.int32})
    train = train_loader.drop(['target', 'id'], axis=1)
    train_labels = train_loader['target'].values
    train_ids = train_loader['id'].values
    train_id_df = train_loader['id']
    print('\n Shape of raw train data:', train.shape)

    test_loader = pd.read_csv(path_test, dtype={'id': np.int32})
    test = test_loader.drop(['id'], axis=1)
    test_ids = test_loader['id'].values
    print(' Shape of raw test data:', test.shape)

    return train, train_labels, test, train_ids, test_ids,train_id_df

folds = 4
runs = 2

cv_LL = 0
cv_AUC = 0
cv_gini = 0
fpred = []
avpred = []
avreal = []
avids = []

# Load data set and target values
train, target, test, tr_ids, te_ids,train_id_df = load_data()
n_train = train.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True)
col_to_drop = train.columns[train.columns.str.endswith('_cat')]
col_to_dummify = train.columns[train.columns.str.endswith('_cat')].astype(str).tolist()

for col in col_to_dummify:
    dummy = pd.get_dummies(train_test[col].astype('category'))
    columns = dummy.columns.astype(str).tolist()
    columns = [col + '_' + w for w in columns]
    dummy.columns = columns
    train_test = pd.concat((train_test, dummy), axis=1)

train_test.drop(col_to_dummify, axis=1, inplace=True)
train_test_scaled, scaler = scale_data(train_test)
train = train_test_scaled[:n_train, :]
test = train_test_scaled[n_train:, :]
print('\n Shape of processed train data:', train.shape)
print(' Shape of processed test data:', test.shape)

train_id_df = pd.DataFrame(train_id_df)

trainfoldSource = pd.read_csv(inDir+'/input/Prav_5folds_CVindices.csv')
train_CVindices = pd.merge(train_id_df, trainfoldSource, how='left',on="id")

patience    = 10
batch_size  = 8912
nb_epoch    = 100
VERBOSEFLAG = 1
full_nb_epoch = 28

def nn_model(X_train):
        model = Sequential()
        model.add(
            Dense(
                200,
                input_dim=X_train.shape[1],
                kernel_initializer='glorot_normal',
                ))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(100, kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Dense(50, kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.15))
        model.add(Dense(25, kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
       # model.compile(optimizer='adam', metrics = ['accuracy'], loss='binary_crossentropy')

        return model
 
def train_nn(i):
    trainindex = train_CVindices[train_CVindices['CVindices'] != i].index.tolist()
    valindex   = train_CVindices[train_CVindices['CVindices'] == i].index.tolist()
    
    X_val_df = train_CVindices.iloc[valindex,:]
    
#    X_build , X_valid = train_X[trainindex,:], train_X[valindex,:]
#    y_build , y_valid = train_y[trainindex,:], train_y[valindex,:]
    
    X_build, X_valid = train[trainindex], train[valindex]
    y_build, y_valid = target[trainindex], target[valindex]
    #train_ids, val_ids = tr_ids[trainindex], tr_ids[valindex]
    
    y_build = np.array(y_build)
    y_valid = np.array(y_valid)
    model = nn_model(X_build)
#    callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=1)]
    model.compile(optimizer='adam', metrics = ['accuracy'], loss='binary_crossentropy')
#    callbacks = [
#            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
#        ]
    callbacks = [
            roc_auc_callback(training_data=(X_build, y_build),validation_data=(X_valid, y_valid)),  # call this before EarlyStopping
            EarlyStopping(monitor='norm_gini_val', patience=patience, mode='max', verbose=1),
            CSVLogger(inDir+'/keras-5fold-run-01-v1-epochs.log', separator=',', append=False),
            ModelCheckpoint(
                    MODEL_WEIGHTS_FILE,
                    monitor='norm_gini_val', mode='max', # mode must be set to max or Keras will be confused
                    save_best_only=True,
                    verbose=1)
        ]
#    nnet = KerasClassifier(
#            build_fn = model,
## Epoch needs to be set to a very large number ; early stopping will prevent it from reaching
##            epochs=5000,
#            epochs=1,
#            batch_size=batchsize,
#            validation_data=(X_valid, y_valid),
#            verbose=2,
#            shuffle=True,
#            callbacks=callbacks)
#
#    nnet.fit(X_build, y_build)
    model.fit(X_build, y_build, batch_size=batch_size, nb_epoch=nb_epoch,
          shuffle=True, verbose=VERBOSEFLAG, validation_data=(X_valid, y_valid),
          callbacks=callbacks
          )
    
    model.load_weights(MODEL_WEIGHTS_FILE)
    
#    nnet = load_model(inDir+'/keras-5fold-run-01-v1-fold-' + str(i) +'.check')
    
    pred_cv = model.predict_proba(X_valid, verbose=VERBOSEFLAG)
    LL_run = log_loss(y_valid, pred_cv)
    print('\n Fold %d Log-loss: %.5f' % ((i), LL_run))
    AUC_run = roc_auc_score(y_valid, pred_cv)
    print(' Fold %d AUC: %.5f' % ((i), AUC_run))
    print(' Fold %d normalized gini: %.5f' % ((i), AUC_run*2-1))
    
    
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.head()
    pred_cv.columns = ["target"]
    pred_cv["id"] = X_val_df.id.values
    
    sub_valfile = inDir+'/submissions/Prav.nn01.fold' + str(i) + '.csv'
    pred_cv = pred_cv[["id","target"]]
    pred_cv.to_csv(sub_valfile, index=False)
    
    pred_test = model.predict_proba(test,verbose=VERBOSEFLAG)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["target"]
    pred_test["id"] = te_ids
    pred_test = pred_test[["id","target"]]
    sub_file = inDir+'/submissions/Prav.nn01.fold' + str(i) + '-test' + '.csv'
    pred_test.to_csv(sub_file, index=False)   
   
    os.remove(MODEL_WEIGHTS_FILE)
    
    del pred_cv
    del pred_test
    del model


def train_full_nn():   
    
   
    model = nn_model(train)
#    callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=1)]
    model.compile(optimizer='adam', metrics = ['accuracy'], loss='binary_crossentropy')
#    callbacks = [
#            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
#        ]
    callbacks = [
            roc_auc_callback(training_data=(train, target),validation_data=(train, target)),  # call this before EarlyStopping
            EarlyStopping(monitor='norm_gini_val', patience=patience, mode='max', verbose=VERBOSEFLAG),
            CSVLogger(inDir+'/keras-5fold-run-01-v1-epochs.log', separator=',', append=False),
            ModelCheckpoint(
                    MODEL_WEIGHTS_FILE,
                    monitor='norm_gini_val', mode='max', # mode must be set to max or Keras will be confused
                    save_best_only=True,
                    verbose=1)
        ]

    model.fit(train, target, batch_size=batch_size, nb_epoch=full_nb_epoch,
          shuffle=True, verbose=VERBOSEFLAG, validation_data=(train, target),
          callbacks=callbacks
          )
    
    model.load_weights(MODEL_WEIGHTS_FILE)
    
#    nnet = load_model(inDir+'/keras-5fold-run-01-v1-fold-' + str(i) +'.check')
    
       
    pred_test = model.predict_proba(test,verbose=VERBOSEFLAG)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["target"]
    pred_test["id"] = te_ids
    pred_test = pred_test[["id","target"]]
    sub_file = inDir+'/submissions/Prav.nn01.full.csv'
    pred_test.to_csv(sub_file, index=False)   
   
    os.remove(MODEL_WEIGHTS_FILE)
    
    
    del pred_test
    del model




folds = 5
if __name__ == '__main__':
    for i in range(1, folds+1):
        train_nn(i)
    train_full_nn()
        
    


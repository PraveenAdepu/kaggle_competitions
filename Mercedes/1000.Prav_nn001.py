
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 22:03:59 2017

@author: SriPrav
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

pal = sns.color_palette()


import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, hstack
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import roc_auc_score, log_loss

user = 'SriPrav'
inDir = 'C:/Users/'+user+'/Documents/R/26Mercedes'
train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
trainingSet = pd.read_csv(train_file)
testingSet = pd.read_csv(test_file)
print(trainingSet.shape) # (4209, 378)
print(testingSet.shape)  # (4209, 377)

#for c in trainingSet.columns:
#    if trainingSet[c].dtype == 'object':
#        print('object column ', c , ' Processing') 
#        lbl = LabelEncoder() 
#        lbl.fit(list(trainingSet[c].values) + list(testingSet[c].values)) 
#        trainingSet[c] = lbl.transform(list(trainingSet[c].values))
#        testingSet[c] = lbl.transform(list(testingSet[c].values))

y_mean = np.mean(trainingSet["y"])

# shape        
print('Shape train: {}\nShape test: {}'.format(trainingSet.shape, testingSet.shape))

trainingSet['IDx'] = trainingSet['ID']
testingSet['IDx'] = trainingSet['ID']

# shape        
print('Shape train: {}\nShape test: {}'.format(trainingSet.shape, testingSet.shape))


##Add decomposed components: PCA / ICA etc.
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
n_comp = 10

# tSVD
#tsvd = TruncatedSVD(n_components=n_comp, random_state=42)
#tsvd_results_train = tsvd.fit_transform(trainingSet.drop(["y"], axis=1))
#tsvd_results_test = tsvd.transform(testingSet)

# PCA
#pca = PCA(n_components=n_comp, random_state=42)
#pca2_results_train = pca.fit_transform(trainingSet.drop(["y"], axis=1))
#pca2_results_test = pca.transform(testingSet)
#
## ICA
#ica = FastICA(n_components=n_comp, random_state=42)
#ica2_results_train = ica.fit_transform(trainingSet.drop(["y"], axis=1))
#ica2_results_test = ica.transform(testingSet)

## GRP
#grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
#grp_results_train = grp.fit_transform(trainingSet.drop(["y"], axis=1))
#grp_results_test = grp.transform(testingSet)
#
## SRP
#srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
#srp_results_train = srp.fit_transform(trainingSet.drop(["y"], axis=1))
#srp_results_test = srp.transform(testingSet)

# Append decomposition components to datasets
#for i in range(1, n_comp+1):
#    trainingSet['pca_' + str(i)] = pca2_results_train[:,i-1]
#    testingSet['pca_' + str(i)] = pca2_results_test[:, i-1]
#    
#    trainingSet['ica_' + str(i)] = ica2_results_train[:,i-1]
#    testingSet['ica_' + str(i)] = ica2_results_test[:, i-1]
    
#    trainingSet['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
#    testingSet['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]
#
#    trainingSet['grp_' + str(i)] = grp_results_train[:, i - 1]
#    testingSet['grp_' + str(i)] = grp_results_test[:, i - 1]
#
#    trainingSet['srp_' + str(i)] = srp_results_train[:, i - 1]
#    testingSet['srp_' + str(i)] = srp_results_test[:, i - 1]
# shape        
print('Shape train: {}\nShape test: {}'.format(trainingSet.shape, testingSet.shape))
   
#    train['tsvd_' + str(i)] = tsvd_results_train[:,i-1]
#    test['tsvd_' + str(i)] = tsvd_results_test[:, i-1]


#train_features_12 = inDir + "/input/train_question_freq_features_from_porter_02.csv"
#test_features_12 = inDir + "/input/test_question_freq_features_from_porter_02.csv"
#train_features_12 = pd.read_csv(train_features_12)
#test_features_12 = pd.read_csv(test_features_12)
#print(train_features_12.shape) # (404290, 36)
#print(test_features_12.shape)  # (2345796, 34)
#
#del train_features_12['is_duplicate']
#test_features_12.rename(columns={'id': 'test_id'}, inplace=True)
#
#trainingSet = pd.merge(trainingSet, train_features_12, how = 'left', on = 'id')
#testingSet = pd.merge(testingSet, test_features_12, how = 'left', on = 'test_id')

cv_file = inDir + "/CVSchema/Prav_CVindices_10folds.csv"
CV_Schema = pd.read_csv(cv_file)

trainingSet = pd.merge(trainingSet, CV_Schema, how = 'left', on = ['ID'])

feature_names  = [col for col in trainingSet.columns if col not in ['y','CVindices']] #'ID',


Normalize_features = [col for col in trainingSet.columns if col not in ['ID','y','CVindices']] #
                     
train_Normalize = trainingSet[Normalize_features]
test_Normalize = testingSet[Normalize_features]

# Scale train_X and test_X together
traintest_Normalize = pd.concat([train_Normalize, test_Normalize], axis=0)
print(traintest_Normalize.shape)

# One-hot encoding of categorical/strings
#categories = ['X0', 'X1', 'X2', 'X3', 'X4','X5', 'X6', 'X8']

traintest_Normalize = pd.get_dummies(traintest_Normalize, prefix=['X0', 'X1', 'X2', 'X3', 'X4','X5', 'X6', 'X8'], columns=['X0', 'X1', 'X2', 'X3', 'X4','X5', 'X6', 'X8'])

Normalize_dummy_features = traintest_Normalize.columns
traintest_Normalize = preprocessing.minmax_scale(traintest_Normalize)


train_Normalize_complete = traintest_Normalize[range(train_Normalize.shape[0])]
test_Normalize_complete = traintest_Normalize[range(train_Normalize.shape[0], traintest_Normalize.shape[0])]
print(train_Normalize_complete.shape)
print(test_Normalize_complete.shape) 

train_Normalize_complete = pd.DataFrame(train_Normalize_complete, columns = Normalize_dummy_features)
test_Normalize_complete = pd.DataFrame(test_Normalize_complete, columns = Normalize_dummy_features)

train_Normalize.head() 
train_Normalize_complete.head()

for col in Normalize_features:
    del trainingSet[col]

for col in Normalize_features:
    del testingSet[col]
       
trainingSet = pd.concat([trainingSet,train_Normalize_complete], axis=1)
testingSet = pd.concat([testingSet,test_Normalize_complete], axis=1)

del train_Normalize
del test_Normalize
del traintest_Normalize
del train_Normalize_complete
del test_Normalize_complete

gc.collect()




##################################################################################################################################
##################################################################################################################################
#for column in features_to_use:
#    print(column)
#    print(' AUC:', roc_auc_score(trainingSet['is_duplicate'], trainingSet[column]))
##################################################################################################################################
##################################################################################################################################
features_to_use =  [col for col in trainingSet.columns if col not in ['ID','y','CVindices']]



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
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import MinMaxScaler # Extra
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
MODEL_WEIGHTS_FILE = inDir+'/nn001_weights.h5'

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    
    
def nn_model(size):
    model = Sequential()
    
    model.add(Dense(200, input_dim = size, init = 'he_normal', activation='linear'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
#    model.add(Dense(200, init = 'he_normal', activation='linear'))
#    model.add(PReLU())
#    model.add(BatchNormalization())
#    model.add(Dropout(0.2))
    
    model.add(Dense(25, init = 'he_normal', activation='linear'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'he_normal', activation='linear'))
    
    return(model)
    

#callbacks = [
#            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
#           ]
        
nbags = 1
folds = 10
epochs = 300
batchsize = 128
verboselog = 2
ModelName = 'nn001'
x_test = testingSet[features_to_use].apply(pd.to_numeric)

def nnet(i):
    print('Fold ', i , ' Processing')
    X_build = trainingSet[trainingSet['CVindices'] != i] # 636112
    #X_build = X_build.groupby('id').first().reset_index() # 331085
    X_val   = trainingSet[trainingSet['CVindices'] == i]
    
    print(X_build.shape) # (404290, 6)
    print(X_val.shape)  # (2345796, 3)

    X_train = X_build[features_to_use]
    X_valid = X_val[features_to_use]
    
    X_train = X_train.apply(pd.to_numeric)
    X_valid = X_valid.apply(pd.to_numeric)

    X_trainy = X_build['y']
    X_validy = X_val['y']
    
    X_trainy = X_trainy.apply(pd.to_numeric).values
    X_validy = X_validy.apply(pd.to_numeric).values
    
    pred_cv = np.zeros(X_validy.shape[0])
    pred_test = np.zeros(x_test.shape[0])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros(X_validy.shape[0])
        model = nn_model(X_train.shape[1])
        callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=1)]
        model.compile(loss = 'mse', optimizer = 'adadelta',metrics=[r2_keras])        
        model.fit(X_train.values, X_trainy, nb_epoch=epochs, batch_size=batchsize,  verbose=verboselog,validation_data=(X_valid.values,X_validy),callbacks=callbacks)#
        model.load_weights(inDir +'/nn001_weights.h5')
        bag_cv   = model.predict(X_valid.values)[:,0]
        pred_cv += model.predict(X_valid.values)[:,0]
        pred_test += model.predict(x_test.values)[:,0]
        bag_score = r2_score(X_validy, bag_cv)
        print('bag ', j, '- r2_score:', bag_score)
        os.remove(inDir +'/nn001_weights.h5')
    pred_cv /= nbags
    pred_test/= nbags
    fold_score = r2_score(X_validy, pred_cv)
    print('Fold ', i, '- r2_score:', fold_score)
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = ["y"]
    pred_cv["ID"] = X_val.ID.values
    pred_cv = pred_cv[['ID','y']]
    sub_valfile = inDir +'/submissions/Prav.'+ModelName+'.fold' + str(i) + '.csv'    
    pred_cv.to_csv(sub_valfile, index=False)
    
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["y"]
    pred_test["ID"] = testingSet.ID.values
    pred_test = pred_test[['ID','y']]
    sub_file = inDir +'/submissions/Prav.'+ModelName+'.fold' + str(i) + '-test' + '.csv'
    pred_test.to_csv(sub_file, index=False)
    del pred_cv
    del pred_test
    

##########################################################################################
# Full model training
########################################################################################## 
best_epoch = 70
full_epochs = int(best_epoch * 1.5) # Get Best epoch
#
def full_train_nn(i):
    X_train = trainingSet[features_to_use]
    X_train = X_train.apply(pd.to_numeric)
    X_trainy = trainingSet['y'] 
    X_trainy = X_trainy.apply(pd.to_numeric).values
    
    model = nn_model(X_train.shape[1])
    
#    callbacks = [
#            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
#        ]
#    early_stop = EarlyStopping(monitor='val_loss', patience=75, verbose=0)
#    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)
    model.fit(X_train.values, X_trainy, nb_epoch=full_epochs, batch_size=batchsize,  verbose=verboselog)
    #model.load_weights('C:/Users/SriPrav/Documents/R/23Quora/nn1_question_pairs_weights.h5')
    pred_test = model.predict(x_test.values)[:,0]
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["y"]
    pred_test["ID"] = testingSet.ID.values
    pred_test = pred_test[['ID','y']]
    sub_file = inDir +'/submissions/Prav.'+ModelName+'.full' + '.csv'
    pred_test.to_csv(sub_file, index=False)   

    del pred_test
    
folds = 10
nbags = 1

if __name__ == '__main__':
    for i in range(1, folds+1):
        nnet(i)
    full_train_nn(nbags)
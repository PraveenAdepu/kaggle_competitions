# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 12:19:07 2017

@author: SriPrav
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
from sklearn import model_selection, preprocessing, ensemble
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback


inDir = 'C:/Users/SriPrav/Documents/R/30Caesars'

X_build = pd.read_csv(inDir + "/input/X_build.csv")
X_val = pd.read_csv(inDir + "/input/X_val.csv")

featureCols = [col for col in X_build.columns if col  not in ["customer_id", "date" ,"target","id","f_19","f_29"
                                                                 ,"customer_target_median1"
                                                                 ,"customer_target_lag_target_diff_flag"
                                                                 ,"market_target_median1","roll"
                                                                 ,"f0_target_lag1"               
                                                                 ,"f0_target_lead1"
                                                                 ,"f2_target_lag1"
                                                                 ,"f2_target_lead1"
                                                                 ,"marketf33_target_lag1"
                                                                 ,"marketf23_target_lag1"
                                                                 ,"f33_f23_target_lag1"]]

ohe_columns = ["market","f_1","f_9","f_16","f_19","f_20","f_23","f_24","f_29","f_31","f_33"]
Normalize_features = ['f_3','f_4','f_6','f_8','f_10','f_11','f_12','f_13','f_14','f_15','f_17','f_18','f_21','f_22','f_25'
                             ,'f_26','f_27','f_28','f_30','f_32','f_34','f_35','f_36','f_37','f_38','f_39','f_40','f_41' ]

featureCols = ohe_columns + Normalize_features  

def normalise_contineous_features(X_build, X_val ,Normalize_features):                   
    X_build_Normalize = X_build[Normalize_features]
    X_val_Normalize = X_val[Normalize_features]

    # Scale train_X and test_X together
    traintest_Normalize = np.vstack((X_build_Normalize, X_val_Normalize))
    print(traintest_Normalize.shape)
    traintest_Normalize = preprocessing.StandardScaler().fit_transform(traintest_Normalize)

    X_build_Normalize_complete = traintest_Normalize[range(X_build_Normalize.shape[0])]
    X_val_Normalize_complete = traintest_Normalize[range(X_build_Normalize.shape[0], traintest_Normalize.shape[0])]
    print(X_build_Normalize_complete.shape)
    print(X_val_Normalize_complete.shape) 


    X_build_Normalize_complete = pd.DataFrame(X_build_Normalize_complete, columns = Normalize_features)
    X_val_Normalize_complete = pd.DataFrame(X_val_Normalize_complete, columns = Normalize_features)

    X_build_Normalize.head() 
    X_build_Normalize_complete.head()

    X_build.drop(Normalize_features, axis=1, inplace=True)

    #for col in Normalize_features:
    #    del X_build[col]

    X_val.drop(Normalize_features, axis=1, inplace=True)
    #for col in Normalize_features:
    #    del X_val[col]
       
    X_build = pd.concat([X_build,X_build_Normalize_complete], axis=1)
    X_val = pd.concat([X_val,X_val_Normalize_complete], axis=1)

    del X_build_Normalize
    del X_val_Normalize
    del traintest_Normalize
    del X_build_Normalize_complete
    del X_val_Normalize_complete
    return X_build, X_val

X_build, X_val = normalise_contineous_features(X_build, X_val ,Normalize_features)

def ohe_category_features(X_build, X_val ,ohe_columns):
    X_build_ohe = X_build[ohe_columns]
    X_val_ohe = X_val[ohe_columns]

    traintest_ohe = np.vstack((X_build_ohe, X_val_ohe))
    traintest_ohe = pd.DataFrame(traintest_ohe, columns = ohe_columns)
    traintest_ohe = pd.get_dummies(traintest_ohe, prefix=ohe_columns, columns=ohe_columns)


    X_build_ohe_complete = traintest_ohe.iloc[range(X_build_ohe.shape[0])]
    X_val_ohe_complete = traintest_ohe.iloc[range(X_build_ohe.shape[0], traintest_ohe.shape[0])]
    X_val_ohe_complete = X_val_ohe_complete.reset_index(drop=True)
    
    return X_build_ohe_complete , X_val_ohe_complete

X_build_ohe_complete , X_val_ohe_complete = ohe_category_features(X_build, X_val ,ohe_columns)
X_build_model = pd.concat([X_build[Normalize_features],X_build_ohe_complete], axis=1)
X_val_model = pd.concat([X_val[Normalize_features],X_val_ohe_complete], axis=1)
    


y_build = X_build['target']
y_val   = X_val['target']




def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
    
def nn_model(X_train):
    model = Sequential()
    
    model.add(Dense(200, input_dim = X_train.shape[1], init = 'glorot_normal'))
    model.add(PReLU())
    model.add(BatchNormalization()) 
    model.add(Dropout(0.4))
    
    model.add(Dense(100, init = 'glorot_normal'))
    model.add(PReLU())
    model.add(BatchNormalization()) 
    model.add(Dropout(0.2))
    
    model.add(Dense(50, init = 'glorot_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='sigmoid'))
    
    return(model)
    
nbags = 1
folds = 0
epochs = 55
batchsize = 128
verboselog = 1

patience    = 10
batch_size  = 128
nb_epoch    = 1
VERBOSEFLAG = 1
full_nb_epoch = 28
from sklearn.metrics import mean_squared_error

for i in range(1, folds+1):
    print('Fold ', i , ' Processing')
#    X_build = train[train['CVindices'] != i]
#    X_val   = train[train['CVindices'] == i]
    
#    X_train = X_build_model
#    X_valid = X_val_model
    
#    X_trainy = y_build
#    X_validy = y_val
        
    pred_cv = np.zeros(y_val.shape[0])
    
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros(y_val.shape[0])
        callbacks = [
                     EarlyStopping(monitor='val_loss', patience=3, verbose=1),
                    ]
        model = nn_model(X_build_model)
        model.compile( optimizer = 'adadelta',loss = 'mse')
        
        #fit      = model.fit(X_build_model.values, y_build, nb_epoch=epochs, batch_size=batchsize,  verbose=verboselog)
        model.fit(X_build_model.values, y_build, batch_size=batch_size, nb_epoch=nb_epoch,
                  shuffle=True, verbose=VERBOSEFLAG, validation_data=(X_val_model.values, y_val),
                  callbacks=callbacks
                 )
        bag_cv   = model.predict(X_val_model.values)[:,0]
        pred_cv += model.predict(X_val_model.values)[:,0]
       
        bag_score = mean_squared_error(y_val, bag_cv)
        print('bag ', '- rmse:', bag_score)
    pred_cv /= nbags
    
    fold_score = mean_absolute_error(y_val, pred_cv)
    print('Fold ', '- rmse:', fold_score)
    
    pred_cv_df = pd.DataFrame({'id': X_val['id'], 'target': pred_cv})
    
    
    if i == 1:  
        pred_cv_df.to_csv(inDir+'/submissions/Prav.nn01.fold1.csv', index = False)   
        

pred_fulltest = np.zeros(X_test.shape[0])


def nn_model():
    model = Sequential()
    
    model.add(Dense(400, input_dim = X_trainFull.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization()) 
    model.add(Dropout(0.4))
    
    model.add(Dense(200, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization()) 
    model.add(Dropout(0.2))
    
    model.add(Dense(50, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)
    
print('Full model training')
for j in range(1,nbags+1):
    print('bag ', j , ' Processing')        
    model = nn_model()
    fit      = model.fit(X_trainFull.values, X_trainFully, nb_epoch=epochs, batch_size=batchsize,  verbose=verboselog)
    pred_fulltest += np.exp(model.predict(X_test.values)[:,0])-200
pred_fulltest/= nbags
pred_fulltest_df = pd.DataFrame({'id': test['id'], 'loss': pred_fulltest})    
pred_fulltest_df.to_csv('C:/Users/SriPrav/Documents/R/14Allstate/submissions/prav.keras20.full.csv', index = False)


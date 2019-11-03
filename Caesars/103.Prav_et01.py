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

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor

inDir = 'C:/Users/SriPrav/Documents/R/30Caesars'

X_build = pd.read_csv(inDir + "/input/X_build.csv")
X_val = pd.read_csv(inDir + "/input/X_val.csv")

X_build['customer_visit_date_lag1'].head()
X_build['customer_visit_date_lag1']=X_build['customer_visit_date_lag1'].str.replace('-','').apply(int)
X_val['customer_visit_date_lag1']=X_val['customer_visit_date_lag1'].str.replace('-','').apply(int)

feature_names = [col for col in X_build.columns if col  not in ["customer_id", "date" ,"target","id","f_19","f_29"
                                                                 ,"customer_target_median1"
                                                                 ,"customer_target_lag_target_diff_flag"
                                                                 ,"market_target_median1","roll"
                                                                 ,"f0_target_lag1"               
                                                                 ,"f0_target_lead1"
                                                                 ,"f2_target_lag1"
                                                                 ,"f2_target_lead1"
                                                                 ,"marketf33_target_lag1"
                                                                 ,"marketf23_target_lag1"
                                                                 ,"f33_f23_target_lag1"
                                                                 ]]




#y_build = X_build['target']
#y_val   = X_val['target']



def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
    

from sklearn.metrics import mean_squared_error
from math import sqrt

nbags = 2
current_seed = 2017
folds = 0
for i in range(1, folds+1):
    print('Fold ', i , ' Processing')
        
    pred_cv = np.zeros(X_val['target'].shape[0])    
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros(X_val['target'].shape[0])
        bag_seed = current_seed + j
        RegressionModel = ExtraTreesRegressor(n_estimators=500, max_depth=8, max_features="auto", n_jobs=30, random_state=bag_seed,verbose=1)
        RegressionModel.fit(X_build[feature_names],X_build['target'])            
        
        bag_cv  = RegressionModel.predict(X_val[feature_names]) 
        pred_cv += RegressionModel.predict(X_val[feature_names])
       
        bag_score = sqrt(mean_squared_error(X_val['target'], bag_cv))
        print('bag ', '- rmse:', bag_score)
    pred_cv /= nbags
    
    fold_score = sqrt(mean_squared_error(X_val['target'], pred_cv))
    print('Fold ', '- rmse:', fold_score)
    
    pred_cv_df = pd.DataFrame({'id': X_val['id'], 'target': pred_cv})
    
    
    if i == 1:  
        pred_cv_df.to_csv(inDir+'/submissions/Prav.et01.fold1.csv', index = False)   
        
qwk(X_val['target'], (pred_cv+0.4), max_rat=20)
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


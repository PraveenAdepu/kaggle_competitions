import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, hstack
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string

inDir = 'C:/Users/SriPrav/Documents/R/21Rental'

train_file = inDir + "/input/Prav_trainingSet_Features_fromRef.csv"
test_file = inDir + "/input/Prav_testingSet_Features_fromRef.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train_df.shape) # (49352, 286)
print(test_df.shape)  # (74659, 286)

cv_file = inDir + "/CVSchema/Prav_CVindices_5folds.csv"
CV_Schema = pd.read_csv(cv_file)

CV_Schema.head()
del CV_Schema["interest_level"]

#train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
#test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()
train_df['interest_level'].head(5)
target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = pd.DataFrame(train_df['interest_level'].apply(lambda x: target_num_map[x]))

train_y.head()
features_to_use = cols = [col for col in train_df.columns if col not in ['interest_level']] 

train_df = pd.merge(train_df, CV_Schema, how = 'left', on = 'listing_id')

train_X = csr_matrix(train_df[features_to_use])
test_X  = csr_matrix(test_df[features_to_use])




param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.01
param['max_depth'] = 6
param['silent'] = 1
param['num_class'] = 3
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 25
param['seed'] = 2017
param['print_every_n'] = 50
num_rounds = 3300
plst = list(param.items())


def train_xgboost(i):
    trainindex = train_df[train_df['CVindices'] != i].index.tolist()
    valindex   = train_df[train_df['CVindices'] == i].index.tolist()
    
    X_val_df = train_df.iloc[valindex,:]
    
    X_build , X_valid = train_X[trainindex,:], train_X[valindex,:]
    y_build , y_valid = train_y.iloc[trainindex,:], train_y.iloc[valindex,:]
    
    xgbbuild = xgb.DMatrix(X_build, label=y_build)
    xgbval = xgb.DMatrix(X_valid, label=y_valid)
    watchlist = [ (xgbbuild,'build'), (xgbval, 'valid') ]
                 
    model = xgb.train(plst, 
                      xgbbuild, 
                      num_rounds, 
                      watchlist, 
                      verbose_eval = 100 #,
                      #early_stopping_rounds=20
                      )
    pred_cv = model.predict(xgbval)
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = ["high", "medium", "low"]
    pred_cv["listing_id"] = X_val_df.listing_id.values
    
    sub_valfile = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.xgb15.fold' + str(i) + '.csv'
    pred_cv.to_csv(sub_valfile, index=False)
    
    xgtest = xgb.DMatrix(test_X)
    pred_test = model.predict(xgtest)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["high", "medium", "low"]
    pred_test["listing_id"] = test_df.listing_id.values
   
    sub_file = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.xgb15.fold' + str(i) + '-test' + '.csv'
    pred_test.to_csv(sub_file, index=False)
    del pred_cv
    del pred_test

fullnum_rounds = int(num_rounds * 1.2)

def fulltrain_xgboost(bags):
    xgbtrain = xgb.DMatrix(train_X, label=train_y)
    watchlist = [ (xgbtrain,'train') ]
    fullmodel = xgb.train(plst, 
                              xgbtrain, 
                              fullnum_rounds, 
                              watchlist,
                              verbose_eval = 100,
                              )
    xgtest = xgb.DMatrix(test_X)
    predfull_test = fullmodel.predict(xgtest)
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["high", "medium", "low"]
    predfull_test["listing_id"] = test_df.listing_id.values
   
    sub_file = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.xgb15.full' + '.csv'
    predfull_test.to_csv(sub_file, index=False)

folds = 5
i = 1
if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost(i)
    fulltrain_xgboost(folds)
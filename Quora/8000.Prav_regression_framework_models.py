
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, auc
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

import keras as k
import keras.layers as l
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

inDir = 'C:/Users/SriPrav/Documents/R/24DSM'
train_file = inDir + "/input/build_set_FE_01.csv"

train_df = pd.read_csv(train_file)
print(train_df.shape) # (558352, 247)

train_file_02 = inDir + "/input/build_set_FE_30.csv"
train_df_02 = pd.read_csv(train_file_02)
print(train_df_02.shape) # (558352, 41)

##train_df["Patient_ID"] = train_df["Patient_ID"].astype(np.int32)
##train_df_02["Patient_ID"] = train_df_02["Patient_ID"].astype(np.int32)
train_df = pd.merge(train_df, train_df_02, how = 'left', on = 'Patient_ID')

cv_file = inDir + "/CVSchema/Prav_CVindices_5folds_10.csv"
CV_Schema = pd.read_csv(cv_file)
CV_Schema.head(2)

#train_df.fillna(0)
train_df.isnull().values.any()
trainingSet = train_df[train_df.Patient_ID < 279201]
testingSet  = train_df[train_df.Patient_ID >= 279201]

trainingSet = pd.merge(trainingSet, CV_Schema, how = 'left', on = 'Patient_ID')
feature_names = [col for col in trainingSet.columns if col not in ['Patient_ID', 'CVindices','DiabetesDispense']]

#############################################################################################################################################
# parameters : xgb regression ###############################################################################################################
#############################################################################################################################################
param = {}
param['objective'] = 'reg:linear'
param['eta'] = 0.01
param['max_depth'] = 8
param['silent'] = 1
param['eval_metric'] = "auc"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 25
param['seed'] = 2017
param['print_every_n'] = 50
xgb_num_rounds = 10



xgbParameters = list(param.items())

lgbm_params = {
          'boosting_type': 'gbdt',
          'objective': 'regression',
          'metric': 'auc',
          'min_child_weight': 1.5,
          'num_leaves': 2**5,
          'lambda_l2': 10,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'learning_rate': 0.03,
          'tree_method': 'exact',          
          'nthread': 25,
          'silent': True,
          'seed': 2017,
         }
lgbm_num_round = 700
lgbm_early_stopping_rounds = 100
seed = 2017
########################################################################################################

def nn_model(size):
    model = Sequential()
    
    model.add(Dense(200, input_dim = size, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
       
    model.add(Dense(100, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(50, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(10, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'he_normal', activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics=['accuracy'])
    return(model)
    
########################################################################################################

folds = 5
nbags = 5
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2017
#
#############################################################################################################################################
# parameters : xgb regression ###############################################################################################################
#############################################################################################################################################

randomforest = RandomForestRegressor(n_estimators=10, max_depth=10, n_jobs=20, random_state=2017, max_features="auto",verbose=1)
adaboost     = AdaBoostRegressor(n_estimators=500, random_state=2017, learning_rate=0.01)
gbdt         = GradientBoostingRegressor(n_estimators=500,learning_rate=0.04,  subsample=0.8, random_state=2017,max_depth=5,verbose=1)
extratree    = ExtraTreesRegressor(n_estimators=600, max_depth=8, max_features="auto", n_jobs=20, random_state=2017,verbose=1)
lr_reg       = LinearRegression(n_jobs=20)
kNN          = KNeighborsRegressor(n_neighbors=10,n_jobs=20,random_state=2017,verbose=1)
#############################################################################################################################################
# parameters : regression ###################################################################################################################
#############################################################################################################################################
xgbModelName  = 'xgb21'
lgbmModelName = 'lgbm02'
rfModelName   = 'rf21'
adaModelName  = 'ada20'
gbdtModelName = 'gbdt20'
etModelName   = 'et21'
lrModelName   = 'lr20'
nnModelName   = 'nn21'
knnModelName  = 'knn20'

if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, xgb_num_rounds)
    fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,xgbModelName,xgbParameters, xgb_num_rounds)
    
    for i in range(1, folds+1):
        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbm_params, lgbm_num_round)
    fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round)
    
    for i in range(1, folds+1):
        train_regression("rf",trainingSet, testingSet,feature_names,i,nbags,rfModelName,current_seed)
    fulltrain_regression("rf",trainingSet, testingSet,feature_names,nbags,rfModelName,current_seed)
    
    
    for i in range(1, folds+1):
        train_regression("et",trainingSet, testingSet,feature_names,i,nbags,etModelName,current_seed)
    fulltrain_regression("et",trainingSet, testingSet,feature_names,nbags,etModelName,current_seed)
    
    for i in range(1, folds+1):
        train_regression("lr",trainingSet, testingSet,feature_names,i,nbags,lrModelName,current_seed)
    fulltrain_regression("lr",trainingSet, testingSet,feature_names,nbags,lrModelName,current_seed)
    
    for i in range(1, folds+1):
        train_regression("knn",trainingSet, testingSet,feature_names,i,nbags,knnModelName,current_seed)
    fulltrain_regression("knn",trainingSet, testingSet,feature_names,nbags,knnModelName,current_seed)
    
    for i in range(1, folds+1):
        train_nn_regression(trainingSet, testingSet,feature_names,i,nbags,nnModelName,nn_epoch)
    fulltrain_nn_regression(trainingSet, testingSet,feature_names,nbags,nnModelName,full_nn_epoch)
    
#    for i in range(1, folds+1):
#        train_regression("ada",trainingSet, testingSet,feature_names,i,nbags,adaModelName,current_seed)
#    fulltrain_regression("ada",trainingSet, testingSet,feature_names,nbags,adaModelName,current_seed)
#    
#    for i in range(1, folds+1):
#        train_regression("gbdt",trainingSet, testingSet,feature_names,i,nbags,gbdtModelName,current_seed)
#    fulltrain_regression("gbdt",trainingSet, testingSet,feature_names,nbags,gbdtModelName,current_seed)


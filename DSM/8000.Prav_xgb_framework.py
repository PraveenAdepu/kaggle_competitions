
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, auc
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression


inDir = 'C:/Users/SriPrav/Documents/R/24DSM'
train_file = inDir + "/input/build_set_FE_01.csv"

train_df = pd.read_csv(train_file)
print(train_df.shape) # (558352, 247)

cv_file = inDir + "/CVSchema/Prav_CVindices_5folds_10.csv"
CV_Schema = pd.read_csv(cv_file)
CV_Schema.head(2)

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
num_rounds = 10
folds = 5
nbags = 5
xgbParameters = list(param.items())
xgbModelName = 'xgb20'
rfModelName = 'rf20'
lgbmModelName = 'lgbm01'
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

#
#############################################################################################################################################
# parameters : xgb regression ###############################################################################################################
#############################################################################################################################################

randomforest = RandomForestRegressor(n_estimators=600, max_depth=10, n_jobs=20, random_state=2017, max_features="auto",verbose=1)
adaboost     = AdaBoostRegressor(n_estimators=30, random_state=2017, learning_rate=0.01)
gbdt         = GradientBoostingRegressor(learning_rate=0.04, n_estimators=100, subsample=0.8, random_state=2017,max_depth=5,verbose=1)
extratree    = ExtraTreesRegressor(n_estimators=600, max_depth=8, max_features="auto", n_jobs=20, random_state=2017,verbose=1)
lr_reg       = LinearRegression(n_jobs=-1)

#############################################################################################################################################
# parameters : regression ###################################################################################################################
#############################################################################################################################################

if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,xgbModelName,xgbParameters, num_rounds)
    fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,xgbModelName,xgbParameters, num_rounds)
    
    for i in range(1, folds+1):
        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbm_params, lgbm_num_round)
    fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round)
    
    for i in range(1, folds+1):
        train_regression(randomforest,"rf",trainingSet, testingSet,feature_names,i,nbags,rfModelName)
    fulltrain_regression(randomforest,"rf",trainingSet, testingSet,feature_names,nbags,rfModelName)
    



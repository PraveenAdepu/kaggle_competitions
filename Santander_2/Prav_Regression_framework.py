# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 16:08:16 2018

@author: SriPrav
"""


import pandas as pd
import numpy as np
from sklearn.metrics import log_loss,mean_absolute_error,mean_squared_error, auc
import operator

import xgboost as xgb
import lightgbm as lightgbm

inDir = 'C:/Users/SriPrav/Documents/R/50Santander'


def ceate_feature_map(features):
    outfile = open(inDir +'/ModelLogs/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()
  
def train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,ModelName,xgbParameters, num_rounds,current_seed, verboseeval):

    
       
    X_build = trainingSet[trainingSet['CVindices'] != i]
    X_valid   = trainingSet[trainingSet['CVindices'] == i]
     
    xgbbuild = xgb.DMatrix(X_build[feature_names], label=X_build['target'])
    xgbval = xgb.DMatrix(X_valid[feature_names], label=X_valid['target'])
    watchlist = [ (xgbbuild,'build'), (xgbval, 'valid') ]
    
    xgtest = xgb.DMatrix(testingSet[feature_names])
                 
    pred_cv = np.zeros(X_valid.shape[0])
    pred_test = np.zeros(testingSet.shape[0])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_seed = current_seed + j
        xgbParameters[6] = ('seed',bag_seed)
        print('bag seed ', bag_seed , ' Processing')
        bag_cv = np.zeros(X_valid.shape[0])         
        model = xgb.train(xgbParameters, 
                          xgbbuild, 
                          num_rounds, 
                          watchlist,
#                          feval=gini_xgb, 
#                          maximize=True,
                          verbose_eval = verboseeval                  
                          )
        bag_cv  = model.predict(xgbval)        
        pred_test += model.predict(xgtest)
        pred_cv += bag_cv
        bag_score = np.sqrt(mean_squared_error(X_valid['target'], bag_cv))
#        bag_score = gini_normalized(X_valid['target'], bag_cv)
        print('bag ', j, '- rmse :', bag_score)
    pred_cv /= nbags
    pred_test/= nbags
    fold_score = np.sqrt(mean_squared_error(X_valid['target'], pred_cv))
    print('Fold ', i, '- rmse :', fold_score)
    
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = ["target"]
    pred_cv["ID"] = X_valid.ID.values
    pred_cv["target"] = np.expm1(pred_cv["target"])
    
    sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
    pred_cv[["ID","target"]].to_csv(sub_valfile, index=False)
    
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["target"]
    pred_test["ID"] = testingSet.ID.values
    
    pred_test["target"] = np.expm1(pred_test["target"])
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '-test' + '.csv'
    pred_test[["ID","target"]].to_csv(sub_file, index=False)
    
       
    del pred_cv
    del pred_test
    return fold_score


def fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,ModelName,xgbParameters, num_rounds,current_seed,verboseeval):
    fullnum_rounds = int(num_rounds * 1.2)
    xgbtrain = xgb.DMatrix(trainingSet[feature_names], label=trainingSet['target'])
    watchlist = [ (xgbtrain,'train') ]
    xgtest = xgb.DMatrix(testingSet[feature_names])
    predfull_test = np.zeros(testingSet.shape[0]) 
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_seed = current_seed + j
        xgbParameters[6] = ('seed',bag_seed)
        print('bag seed ', bag_seed , ' Processing')            
        fullmodel = xgb.train(xgbParameters, 
                              xgbtrain, 
                              fullnum_rounds, 
                              watchlist,
#                              feval=gini_xgb, 
#                              maximize=True,
                              verbose_eval = verboseeval,
                              )
    
        predfull_test += fullmodel.predict(xgtest)
    predfull_test/= nbags
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["target"]
    predfull_test["ID"] = testingSet.ID.values
   
    predfull_test["target"] = np.expm1(predfull_test["target"])
    
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
    predfull_test[["ID","target"]].to_csv(sub_file, index=False)
    ceate_feature_map(feature_names)
    importance = fullmodel.get_fscore(fmap=inDir +'/ModelLogs/xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    imp_file = inDir +'/ModelLogs/Prav.'+ str(ModelName)+'.featureImportance' + '.csv'
    df = df.sort_values(['fscore'], ascending=[False])
    df[['feature', 'fscore']].to_csv(imp_file, index=False)
    
def train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,ModelName,lgbmParameters, lgbm_num_rounds,current_seed,verboseeval):
    
    X_build  = trainingSet[trainingSet['CVindices'] != i]
    X_valid  = trainingSet[trainingSet['CVindices'] == i]
     
    lgbmbuild = lightgbm.Dataset(X_build[feature_names], X_build['target'])
    lgbmval   = lightgbm.Dataset(X_valid[feature_names], X_valid['target'])
                 
    pred_cv = np.zeros(X_valid.shape[0])
    pred_test = np.zeros(testingSet.shape[0])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros(X_valid.shape[0])
        lgbmParameters['bagging_seed'] =  current_seed + j    
        lgbmParameters['seed'] =  current_seed + j 
        model = lightgbm.train(lgbmParameters, lgbmbuild,lgbm_num_rounds#,feval=lgb_mclip#,early_stopping_rounds=100
                               ,valid_sets=[lgbmbuild,lgbmval],valid_names=['train','valid'],verbose_eval = verboseeval
                                  #,early_stopping_rounds=early_stopping_rounds
                                  )
        bag_cv= model.predict(X_valid[feature_names] #,num_iteration=model.best_iteration
                             )#.reshape(-1,1)
        pred_test += model.predict(testingSet[feature_names])
        pred_cv += bag_cv
        bag_score = np.sqrt(mean_squared_error(X_valid['target'], bag_cv))
        print('bag ', j, '- rmse :', bag_score)
    pred_cv /= nbags
    pred_test/= nbags
    fold_score = np.sqrt(mean_squared_error(X_valid['target'], pred_cv))
    print('Fold ', i, '- rmse :', fold_score)
    
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = ["target"]
    pred_cv["ID"] = X_valid.ID.values
    pred_cv["target"] = np.expm1(pred_cv["target"])
    
    sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
    pred_cv[["ID","target"]].to_csv(sub_valfile, index=False)
    
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["target"]
    pred_test["ID"] = testingSet.ID.values
    pred_test["target"] = np.expm1(pred_test["target"])
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '-test' + '.csv'
    
    pred_test[["ID","target"]].to_csv(sub_file, index=False)
    del pred_cv
    del pred_test
    return fold_score

def fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,ModelName,lgbmParameters, lgbm_num_rounds,current_seed,verboseeval):
    
    full_lgbm_num_rounds = int(lgbm_num_rounds * 1.2)
    lgbmtrain = lightgbm.Dataset(trainingSet[feature_names], trainingSet['target'])

    predfull_test = np.zeros(testingSet.shape[0]) 
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        lgbmParameters['bagging_seed'] =  current_seed + j  
        lgbmParameters['seed'] =  current_seed + j          
        fullmodel = lightgbm.train(lgbmParameters, lgbmtrain,full_lgbm_num_rounds#,feval=lgb_mclip
                                   ,valid_sets=[lgbmtrain,lgbmtrain],valid_names=['train','train'],verbose_eval = verboseeval
                                  #,early_stopping_rounds=early_stopping_rounds
                                  )
    
        predfull_test += fullmodel.predict(testingSet[feature_names])
    predfull_test/= nbags
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["target"]
    predfull_test["ID"] = testingSet.ID.values
    predfull_test["target"] = np.expm1(predfull_test["target"])
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
    
    predfull_test[["ID","target"]].to_csv(sub_file, index=False)
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 20:47:01 2017

@author: SriPrav
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
import lightgbm

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss,mean_absolute_error,mean_squared_error, auc,roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB


inDir = 'C:/Users/SriPrav/Documents/R/24DSM'

#############################################################################################################################################
# xgb regression ############################################################################################################################
#############################################################################################################################################
def train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,ModelName,xgbParameters, num_rounds):
    
    X_build = trainingSet[trainingSet['CVindices'] != i]
    X_valid   = trainingSet[trainingSet['CVindices'] == i]
     
    xgbbuild = xgb.DMatrix(X_build[feature_names], label=X_build['DiabetesDispense'])
    xgbval = xgb.DMatrix(X_valid[feature_names], label=X_valid['DiabetesDispense'])
    watchlist = [ (xgbbuild,'build'), (xgbval, 'valid') ]
    
    xgtest = xgb.DMatrix(testingSet[feature_names])
                 
    pred_cv = np.zeros(X_valid.shape[0])
    pred_test = np.zeros(testingSet.shape[0])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros(X_valid.shape[0])         
        model = xgb.train(xgbParameters, 
                          xgbbuild, 
                          num_rounds, 
                          watchlist, 
                          verbose_eval = 20                      
                          )
        bag_cv  = model.predict(xgbval)        
        pred_test += model.predict(xgtest)
        pred_cv += bag_cv
        bag_score = roc_auc_score(X_valid['DiabetesDispense'], bag_cv)
        print('bag ', j, '- AUC:', bag_score)
    pred_cv /= nbags
    pred_test/= nbags
    fold_score = roc_auc_score(X_valid['DiabetesDispense'], pred_cv)
    print('Fold ', i, '- AUC:', fold_score)
    
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = ["Diabetes"]
    pred_cv["Patient_ID"] = X_valid.Patient_ID.values
    
    sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
    pred_cv[["Patient_ID","Diabetes"]].to_csv(sub_valfile, index=False)
    
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["Diabetes"]
    pred_test["Patient_ID"] = testingSet.Patient_ID.values
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '-test' + '.csv'
    pred_test[["Patient_ID","Diabetes"]].to_csv(sub_file, index=False)
    del pred_cv
    del pred_test



def fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,ModelName,xgbParameters, num_rounds):
    fullnum_rounds = int(num_rounds * 1.2)
    xgbtrain = xgb.DMatrix(trainingSet[feature_names], label=trainingSet['DiabetesDispense'])
    watchlist = [ (xgbtrain,'train') ]
    xgtest = xgb.DMatrix(testingSet[feature_names])
    predfull_test = np.zeros(testingSet.shape[0]) 
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')            
        fullmodel = xgb.train(xgbParameters, 
                              xgbtrain, 
                              fullnum_rounds, 
                              watchlist,
                              verbose_eval = 20,
                              )
    
        predfull_test += fullmodel.predict(xgtest)
    predfull_test/= nbags
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["Diabetes"]
    predfull_test["Patient_ID"] = testingSet.Patient_ID.values
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
    predfull_test[["Patient_ID","Diabetes"]].to_csv(sub_file, index=False)

#############################################################################################################################################
# xgb regression ############################################################################################################################
#############################################################################################################################################

#############################################################################################################################################
# lgb regression ############################################################################################################################
#############################################################################################################################################
def train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,ModelName,lgbmParameters, lgbm_num_rounds):
    current_seed = 2017
    X_build  = trainingSet[trainingSet['CVindices'] != i]
    X_valid  = trainingSet[trainingSet['CVindices'] == i]
     
    lgbmbuild = lightgbm.Dataset(X_build[feature_names], X_build['DiabetesDispense'])
    lgbmval   = lightgbm.Dataset(X_valid[feature_names], X_valid['DiabetesDispense'])
                 
    pred_cv = np.zeros(X_valid.shape[0])
    pred_test = np.zeros(testingSet.shape[0])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros(X_valid.shape[0])
        lgbmParameters['seed'] =  current_seed + j         
        model = lightgbm.train(lgbmParameters, lgbmbuild,lgbm_num_rounds,valid_sets=lgbmval,verbose_eval = 100
                                  #,early_stopping_rounds=early_stopping_rounds
                                  )
        bag_cv= model.predict(X_valid[feature_names] #,num_iteration=model.best_iteration
                             )#.reshape(-1,1)
        pred_test += model.predict(testingSet[feature_names])
        pred_cv += bag_cv
        bag_score = roc_auc_score(X_valid['DiabetesDispense'], bag_cv)
        print('bag ', j, '- AUC:', bag_score)
    pred_cv /= nbags
    pred_test/= nbags
    fold_score = roc_auc_score(X_valid['DiabetesDispense'], pred_cv)
    print('Fold ', i, '- AUC:', fold_score)
    
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = ["Diabetes"]
    pred_cv["Patient_ID"] = X_valid.Patient_ID.values
    
    sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
    pred_cv[["Patient_ID","Diabetes"]].to_csv(sub_valfile, index=False)
    
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["Diabetes"]
    pred_test["Patient_ID"] = testingSet.Patient_ID.values
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '-test' + '.csv'
    pred_test[["Patient_ID","Diabetes"]].to_csv(sub_file, index=False)
    del pred_cv
    del pred_test



def fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,ModelName,lgbmParameters, lgbm_num_rounds):
    current_seed = 2017
    full_lgbm_num_rounds = int(lgbm_num_rounds * 1.2)
    lgbmtrain = lightgbm.Dataset(trainingSet[feature_names], trainingSet['DiabetesDispense'])

    predfull_test = np.zeros(testingSet.shape[0]) 
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        lgbmParameters['seed'] =  current_seed + j           
        fullmodel = lightgbm.train(lgbmParameters, lgbmtrain,full_lgbm_num_rounds,verbose_eval = 100
                                  #,early_stopping_rounds=early_stopping_rounds
                                  )
    
        predfull_test += fullmodel.predict(testingSet[feature_names])
    predfull_test/= nbags
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["Diabetes"]
    predfull_test["Patient_ID"] = testingSet.Patient_ID.values
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
    predfull_test[["Patient_ID","Diabetes"]].to_csv(sub_file, index=False)

#############################################################################################################################################
# lgb regression ############################################################################################################################
#############################################################################################################################################


#############################################################################################################################################
# oof regression ############################################################################################################################
#############################################################################################################################################
def train_regression(RegressionModel,RegressionModelName,trainingSet, testingSet,feature_names,i,nbags,ModelName):
    if RegressionModelName in ["rf","ada","gb","et","lr","lsvc","knn"]:    
        X_build = trainingSet[trainingSet['CVindices'] != i]
        X_valid   = trainingSet[trainingSet['CVindices'] == i]      
      
        pred_cv = np.zeros(X_valid.shape[0])
        pred_test = np.zeros(testingSet.shape[0])
        
        for j in range(1,nbags+1):
            print('bag ', j , ' Processing')
            bag_cv = np.zeros(X_valid.shape[0])         
            RegressionModel.fit(X_build[feature_names],X_build['DiabetesDispense'])
            bag_cv  = RegressionModel.predict(X_valid[feature_names])        
            pred_test += RegressionModel.predict(testingSet[feature_names])
            pred_cv += bag_cv
            bag_score = roc_auc_score(X_valid['DiabetesDispense'], bag_cv)
            print('bag ', j, '- AUC:', bag_score)
        pred_cv /= nbags
        pred_test/= nbags
        fold_score = roc_auc_score(X_valid['DiabetesDispense'], pred_cv)
        print('Fold ', i, '- AUC:', fold_score)
        
        pred_cv = pd.DataFrame(pred_cv)
        pred_cv.columns = ["Diabetes"]
        pred_cv["Patient_ID"] = X_valid.Patient_ID.values
        
        sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
        pred_cv[["Patient_ID","Diabetes"]].to_csv(sub_valfile, index=False)
        
        pred_test = pd.DataFrame(pred_test)
        pred_test.columns = ["Diabetes"]
        pred_test["Patient_ID"] = testingSet.Patient_ID.values
       
        sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '-test' + '.csv'
        pred_test[["Patient_ID","Diabetes"]].to_csv(sub_file, index=False)
        del pred_cv
        del pred_test

def fulltrain_regression(RegressionModel,RegressionModelName,trainingSet, testingSet,feature_names,nbags,ModelName):
    if RegressionModelName in ["rf","ada","gb","et","lr","lsvc","knn"]:    
        predfull_test = np.zeros(testingSet.shape[0]) 
        for j in range(1,nbags+1):
            print('bag ', j , ' Processing')            
            RegressionModel.fit(trainingSet[feature_names],trainingSet['DiabetesDispense'])    
            predfull_test += RegressionModel.predict(testingSet[feature_names])
        predfull_test/= nbags
        predfull_test = pd.DataFrame(predfull_test)
        predfull_test.columns = ["Diabetes"]
        predfull_test["Patient_ID"] = testingSet.Patient_ID.values
       
        sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
        predfull_test[["Patient_ID","Diabetes"]].to_csv(sub_file, index=False)

#############################################################################################################################################
# oof regression ############################################################################################################################
#############################################################################################################################################


def stacking_reg(clf,train_x,train_y,test_x,clf_name):
    train=np.zeros((train_x.shape[0],1))
    test=np.zeros((test_x.shape[0],1))
    test_pre=np.empty((folds,test_x.shape[0],1))
    cv_scores=[]
    for i,(train_index,test_index) in enumerate(kf):
        tr_x=train_x[train_index]
        tr_y=train_y[train_index]
        te_x=train_x[test_index]
        te_y = train_y[test_index]
        if clf_name in ["rf","ada","gb","et","lr","lsvc","knn"]:
            clf.fit(tr_x,tr_y)
            pre=clf.predict(te_x).reshape(-1,1)
            train[test_index]=pre
            test_pre[i,:]=clf.predict(test_x).reshape(-1,1)
            cv_scores.append(mean_squared_error(te_y, pre))
        elif clf_name in ["xgb"]:
            train_matrix = clf.DMatrix(tr_x, label=tr_y, missing=-1)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x, label=te_y, missing=-1)
            params = {'booster': 'gbtree',
                      'eval_metric': 'rmse',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.03,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 12
                      }
            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'),
                         (test_matrix, 'eval')
                         ]
            if test_matrix:
                model = clf.train(params, train_matrix, num_boost_round=num_round,evals=watchlist,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre= model.predict(test_matrix,ntree_limit=model.best_ntree_limit).reshape(-1,1)
                train[test_index]=pre
                test_pre[i, :]= model.predict(z, ntree_limit=model.best_ntree_limit).reshape(-1,1)
                cv_scores.append(mean_squared_error(te_y, pre))

        elif clf_name in ["lgb"]:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)
            #z = clf.Dataset(test_x, label=te_y)
            #z=test_x
            params = {
                      'boosting_type': 'gbdt',
                      'objective': 'regression_l2',
                      'metric': 'mse',
                      'min_child_weight': 1.5,
                      'num_leaves': 2**5,
                      'lambda_l2': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'learning_rate': 0.03,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 12,
                      'silent': True,
                      }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre= model.predict(te_x,num_iteration=model.best_iteration).reshape(-1,1)
                train[test_index]=pre
                test_pre[i, :]= model.predict(test_x, num_iteration=model.best_iteration).reshape(-1,1)
                cv_scores.append(mean_squared_error(te_y, pre))

        elif clf_name in ["nn"]:
            from keras.layers import Dense, Dropout, BatchNormalization
            from keras.optimizers import SGD,RMSprop
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from keras.utils import np_utils
            from keras.regularizers import l2
            from keras.models import Sequential
            clf = Sequential()
            clf.add(Dense(64, input_dim=tr_x.shape[1], activation="relu", W_regularizer=l2()))
            # model.add(Dropout(0.2))
            clf.add(Dense(64, activation="relu", W_regularizer=l2()))
            # model.add(Dropout(0.2))
            clf.add(Dense(1))
            clf.summary()
            early_stopping = EarlyStopping(monitor='val_loss', patience=20)
            reduce = ReduceLROnPlateau(min_lr=0.0002,factor=0.05)
            clf.compile(optimizer="rmsprop", loss="mse")
            clf.fit(tr_x, tr_y,
                      batch_size=640,
                      nb_epoch=5000,
                      validation_data=[te_x, te_y],
                      callbacks=[early_stopping, reduce])
            pre=clf.predict(te_x).reshape(-1,1)
            train[test_index]=pre
            test_pre[i,:]=clf.predict(test_x).reshape(-1,1)
            cv_scores.append(mean_squared_error(te_y, pre))
        else:
            raise IOError("Please add new clf.")
        print "%s now score is:"%clf_name,cv_scores
        with open("score.txt","a") as f:
            f.write("%s now score is:"%clf_name+str(cv_scores)+"\n")
    test[:]=test_pre.mean(axis=0)
    print "%s_score_list:"%clf_name,cv_scores
    print "%s_score_mean:"%clf_name,np.mean(cv_scores)
    with open("score.txt", "a") as f:
        f.write("%s_score_mean:"%clf_name+str(np.mean(cv_scores))+"\n")
    return train.reshape(-1,1),test.reshape(-1,1)

def rf_reg(x_train, y_train, x_valid):
    randomforest = RandomForestRegressor(n_estimators=600, max_depth=20, n_jobs=-1, random_state=2017, max_features="auto",verbose=1)
    rf_train, rf_test = stacking_reg(randomforest, x_train, y_train, x_valid,"rf")
    return rf_train, rf_test,"rf_reg"

def ada_reg(x_train, y_train, x_valid):
    adaboost = AdaBoostRegressor(n_estimators=30, random_state=2017, learning_rate=0.01)
    ada_train, ada_test = stacking_reg(adaboost, x_train, y_train, x_valid,"ada")
    return ada_train, ada_test,"ada_reg"

def gb_reg(x_train, y_train, x_valid):
    gbdt = GradientBoostingRegressor(learning_rate=0.04, n_estimators=100, subsample=0.8, random_state=2017,max_depth=5,verbose=1)
    gbdt_train, gbdt_test = stacking_reg(gbdt, x_train, y_train, x_valid,"gb")
    return gbdt_train, gbdt_test,"gb_reg"

def et_reg(x_train, y_train, x_valid):
    extratree = ExtraTreesRegressor(n_estimators=600, max_depth=35, max_features="auto", n_jobs=-1, random_state=2017,verbose=1)
    et_train, et_test = stacking_reg(extratree, x_train, y_train, x_valid,"et")
    return et_train, et_test,"et_reg"

def lr_reg(x_train, y_train, x_valid):
    lr_reg=LinearRegression(n_jobs=-1)
    lr_train, lr_test = stacking_reg(lr_reg, x_train, y_train, x_valid, "lr")
    return lr_train, lr_test, "lr_reg"

def xgb_reg(x_train, y_train, x_valid):
    xgb_train, xgb_test = stacking_reg(xgboost, x_train, y_train, x_valid,"xgb")
    return xgb_train, xgb_test,"xgb_reg"

def lgb_reg(x_train, y_train, x_valid):
    lgb_train, lgb_test = stacking_reg(lightgbm, x_train, y_train, x_valid,"lgb")
    return lgb_train, lgb_test,"lgb_reg"

def nn_reg(x_train, y_train, x_valid):
    nn_train, nn_test = stacking_reg("", x_train, y_train, x_valid, "nn")
    return nn_train, nn_test, "nn_reg"
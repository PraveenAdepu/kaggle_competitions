# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 20:47:01 2017

@author: SriPrav
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
import scipy
import lightgbm
from sklearn.metrics import log_loss,mean_absolute_error,mean_squared_error, auc,roc_auc_score,r2_score
from math import sqrt
import operator
from matplotlib import pylab as plt

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB



inDir = 'C:\\Users\\SriPrav\\Documents\\R\\34Corporacion'

#############################################################################################################################################
# xgb regression ############################################################################################################################
#############################################################################################################################################
def ceate_feature_map(features):
    outfile = open(inDir +'/ModelLogs/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)

def gini_normalized(y, pred):
    return gini(y, pred) / gini(y, y)

def train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,ModelName,xgbParameters, num_rounds,current_seed, verboseeval):
    
    X_build = trainingSet[trainingSet['CVindices'] != i]
    X_valid   = trainingSet[trainingSet['CVindices'] == i]
     
    xgbbuild = xgb.DMatrix(X_build[feature_names], label=X_build['unit_sales'])
    xgbval = xgb.DMatrix(X_valid[feature_names], label=X_valid['unit_sales'])
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
                          #feval=gini_xgb, 
                          maximize=True,
                          verbose_eval = verboseeval                  
                          )
        bag_cv  = model.predict(xgbval)        
        pred_test += model.predict(xgtest)
        pred_cv += bag_cv
        bag_score = sqrt(mean_squared_error(X_valid['unit_sales'], bag_cv))
        print('bag ', j, '- rmse :', bag_score)
    pred_cv /= nbags
    pred_test/= nbags
    fold_score = sqrt(mean_squared_error(X_valid['unit_sales'], pred_cv))
    print('Fold ', i, '- rmse :', fold_score)
    
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = ["unit_sales"]
    pred_cv["id"] = X_valid.id.values
    
    sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
    pred_cv[["id","unit_sales"]].to_csv(sub_valfile, index=False)
    
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["unit_sales"]
    pred_test["id"] = testingSet.id.values
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '-test' + '.csv'
    pred_test[["id","unit_sales"]].to_csv(sub_file, index=False)
    del pred_cv
    del pred_test



def fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,ModelName,xgbParameters, num_rounds,current_seed,verboseeval):
    fullnum_rounds = int(num_rounds * 1.2)
    xgbtrain = xgb.DMatrix(trainingSet[feature_names], label=trainingSet['unit_sales'])
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
                              #feval=gini_xgb, 
                              maximize=True,
                              verbose_eval = verboseeval,
                              )
    
        predfull_test += fullmodel.predict(xgtest)
    predfull_test/= nbags
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["unit_sales"]
    predfull_test["id"] = testingSet.id.values
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
    predfull_test[["id","unit_sales"]].to_csv(sub_file, index=False)
    ceate_feature_map(feature_names)
    importance = fullmodel.get_fscore(fmap=inDir +'/ModelLogs/xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    imp_file = inDir +'/ModelLogs/Prav.'+ str(ModelName)+'.featureImportance' + '.csv'
    df = df.sort_values(['fscore'], ascending=[False])
    df[['feature', 'fscore']].to_csv(imp_file, index=False)


#############################################################################################################################################
# xgb regression ############################################################################################################################
#############################################################################################################################################

#############################################################################################################################################
# lgb regression ############################################################################################################################
#############################################################################################################################################
def lgbm_r2_score(preds, train_data):
    labels = train_data.get_label()
    return 'r2', r2_score(labels, preds), False

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True
    
def train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,ModelName,lgbmParameters, lgbm_num_rounds,current_seed,verboseeval):
    
    X_build  = trainingSet[trainingSet['CVindices'] != i]
    X_valid  = trainingSet[trainingSet['CVindices'] == i]
     
    lgbmbuild = lightgbm.Dataset(X_build[feature_names], X_build['unit_sales'])
    lgbmval   = lightgbm.Dataset(X_valid[feature_names], X_valid['unit_sales'], reference=lgbmbuild)
                 
    pred_cv = np.zeros(X_valid.shape[0])
    pred_test = np.zeros(testingSet.shape[0])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros(X_valid.shape[0])
        lgbmParameters['seed'] =  current_seed + j         
        model = lightgbm.train(lgbmParameters, lgbmbuild,lgbm_num_rounds#,feval=gini_lgb
                               ,valid_sets=lgbmval,verbose_eval = verboseeval
                                  #,early_stopping_rounds=early_stopping_rounds
                                  )
        bag_cv= model.predict(X_valid[feature_names] #,num_iteration=model.best_iteration
                             )#.reshape(-1,1)
        pred_test += model.predict(testingSet[feature_names])
        pred_cv += bag_cv
        bag_score = sqrt(mean_squared_error(X_valid['unit_sales'], bag_cv))
        print('bag ', j, '- rmse :', bag_score)
    pred_cv /= nbags
    pred_test/= nbags
    fold_score = sqrt(mean_squared_error(X_valid['unit_sales'], pred_cv))
    print('Fold ', i, '- rmse :', fold_score)
    
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = ["unit_sales"]
    pred_cv["store_nbr"] = X_valid.store_nbr.values
    pred_cv["item_nbr"] = X_valid.item_nbr.values
    pred_cv["date"] = X_valid.date.values
    pred_cv['unit_sales'] = pred_cv['unit_sales'].apply(pd.np.expm1)
    sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
    pred_cv[["store_nbr","item_nbr","date","unit_sales"]].to_csv(sub_valfile, index=False)
    
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["unit_sales"]    
    pred_test["store_nbr"] = testingSet.store_nbr.values
    pred_test["item_nbr"] = testingSet.item_nbr.values
    pred_test["date"] = testingSet.date.values
    pred_test['unit_sales'] = pred_test['unit_sales'].apply(pd.np.expm1)
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '-test' + '.csv'
    pred_test[["store_nbr","item_nbr","date","unit_sales"]].to_csv(sub_file, index=False)
    del pred_cv
    del pred_test

#    print("Features importance...")
#    gain = model.feature_importance('gain')
#    ft = pd.DataFrame({'feature':model.feature_name(), 'split':model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
#    print(ft.head(25))

def fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,ModelName,lgbmParameters, lgbm_num_rounds,current_seed,verboseeval):
    
    full_lgbm_num_rounds = int(lgbm_num_rounds * 1.2)
    lgbmtrain = lightgbm.Dataset(trainingSet[feature_names], trainingSet['unit_sales'])

    predfull_test = np.zeros(testingSet.shape[0]) 
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        lgbmParameters['seed'] =  current_seed + j           
        fullmodel = lightgbm.train(lgbmParameters, lgbmtrain,full_lgbm_num_rounds#,feval=gini_lgb
                                   ,verbose_eval = verboseeval
                                  #,early_stopping_rounds=early_stopping_rounds
                                  )
    
        predfull_test += fullmodel.predict(testingSet[feature_names])
    predfull_test/= nbags
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["unit_sales"]
    predfull_test["id"] = testingSet.id.values
    predfull_test['unit_sales'] = predfull_test['unit_sales'].apply(pd.np.expm1)
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
    predfull_test[["id","unit_sales"]].to_csv(sub_file, index=False)

#############################################################################################################################################
# lgb regression ############################################################################################################################
#############################################################################################################################################


#randomforest = RandomForestRegressor(n_estimators=10, max_depth=10, n_jobs=20, random_state=2017, max_features="auto",verbose=1)
#adaboost     = AdaBoostRegressor(n_estimators=500, random_state=2017, learning_rate=0.01)
#gbdt         = GradientBoostingRegressor(n_estimators=500,learning_rate=0.04,  subsample=0.8, random_state=2017,max_depth=5,verbose=1)
#extratree    = ExtraTreesRegressor(n_estimators=600, max_depth=8, max_features="auto", n_jobs=20, random_state=2017,verbose=1)
#lr_reg       = LinearRegression(n_jobs=20)

#############################################################################################################################################
# oof regression ############################################################################################################################
#############################################################################################################################################
 
def train_regression(RegressionModelName,trainingSet, testingSet,feature_names,i,nbags,ModelName,current_seed):
    if RegressionModelName in ["rf","ada","gbdt","et","lr","lsvc","knn","logiR"]:    
        X_build = trainingSet[trainingSet['CVindices'] != i]
        X_valid   = trainingSet[trainingSet['CVindices'] == i]      
      
        pred_cv = np.zeros(X_valid.shape[0])
        pred_test = np.zeros(testingSet.shape[0])
        
        for j in range(1,nbags+1):
            print('bag ', j , ' Processing')
            bag_cv = np.zeros(X_valid.shape[0])
            bag_seed = current_seed + j
            if RegressionModelName == "rf":
                RegressionModel = RandomForestRegressor(n_estimators=500, max_depth=6, n_jobs=20, random_state=bag_seed, max_features="auto",verbose=0)
            if RegressionModelName == "ada":
                RegressionModel = AdaBoostRegressor(n_estimators=610, random_state=bag_seed, learning_rate=0.01)
            if RegressionModelName == "gbdt":
                RegressionModel = GradientBoostingRegressor(n_estimators=610,learning_rate=0.01,loss="huber", max_features=0.7, min_samples_leaf=18, min_samples_split=14,  subsample=0.9, random_state=bag_seed,max_depth=4,verbose=0)
            if RegressionModelName == "et":
                RegressionModel = ExtraTreesRegressor(n_estimators=500, max_depth=8, max_features="auto", n_jobs=20, random_state=bag_seed,verbose=0)
            if RegressionModelName == "lr":
                RegressionModel = LinearRegression(n_jobs=20)
            if RegressionModelName == "logiR":
                RegressionModel = LogisticRegression(n_jobs=20)
            if RegressionModelName == "knn":
                RegressionModel = KNeighborsRegressor(n_neighbors=2,n_jobs=20)
                
            RegressionModel.fit(X_build[feature_names],X_build['target'])
            bag_cv  = RegressionModel.predict(X_valid[feature_names])        
            pred_test += RegressionModel.predict(testingSet[feature_names])
            pred_cv += bag_cv
            bag_score = gini_normalized(X_valid['target'], bag_cv)
            print('bag ', j, '- gini:', bag_score)
        pred_cv /= nbags
        pred_test/= nbags
        fold_score = gini_normalized(X_valid['target'], pred_cv)
        print('Fold ', i, '- gini:', fold_score)
        
        pred_cv = pd.DataFrame(pred_cv)
        pred_cv.columns = ["target"]
        pred_cv["id"] = X_valid.id.values
        
        sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
        pred_cv[["id","target"]].to_csv(sub_valfile, index=False)
        
        pred_test = pd.DataFrame(pred_test)
        pred_test.columns = ["target"]
        pred_test["id"] = testingSet.id.values
       
        sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '-test' + '.csv'
        pred_test[["id","target"]].to_csv(sub_file, index=False)
        del pred_cv
        del pred_test

def fulltrain_regression(RegressionModelName,trainingSet, testingSet,feature_names,nbags,ModelName,current_seed):
    if RegressionModelName in ["rf","ada","gbdt","et","lr","lsvc","knn","logiR"]:    
        predfull_test = np.zeros(testingSet.shape[0]) 
        for j in range(1,nbags+1):
            print('bag ', j , ' Processing')
            bag_seed = current_seed + j
            if RegressionModelName == "rf":
                RegressionModel = RandomForestRegressor(n_estimators=600, max_depth=6, n_jobs=20, random_state=bag_seed, max_features="auto",verbose=0)
            if RegressionModelName == "ada":
                RegressionModel = AdaBoostRegressor(n_estimators=670, random_state=bag_seed, learning_rate=0.01)
            if RegressionModelName == "gbdt":
                RegressionModel = GradientBoostingRegressor(n_estimators=670,learning_rate=0.01,loss="huber", max_features=0.7, min_samples_leaf=18, min_samples_split=14,  subsample=0.9, random_state=bag_seed,max_depth=4,verbose=0)
            if RegressionModelName == "et":
                RegressionModel = ExtraTreesRegressor(n_estimators=600, max_depth=8, max_features="auto", n_jobs=20, random_state=bag_seed,verbose=0)
            if RegressionModelName == "lr":
                RegressionModel = LinearRegression(n_jobs=10)
            if RegressionModelName == "logiR":
                RegressionModel = LogisticRegression(n_jobs=20)
            if RegressionModelName == "knn":
                RegressionModel = KNeighborsRegressor(n_neighbors=2,n_jobs=20)
                
            RegressionModel.fit(trainingSet[feature_names],trainingSet['target'])    
            predfull_test += RegressionModel.predict(testingSet[feature_names])
        predfull_test/= nbags
        predfull_test = pd.DataFrame(predfull_test)
        predfull_test.columns = ["target"]
        predfull_test["id"] = testingSet.id.values
       
        sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
        predfull_test[["id","target"]].to_csv(sub_file, index=False)
        if RegressionModelName == "rf":
            VarImp = pd.DataFrame(RegressionModel.feature_importances_, columns =['Score'] )
            VarImp['Feature_Name'] = feature_names
            imp_file = inDir +'/ModelLogs/Prav.'+ str(ModelName)+'.featureImportance' + '.csv'
            VarImp = VarImp.sort_values(['Score'], ascending=[False])
            VarImp[['Feature_Name', 'Score']].to_csv(imp_file, index=False)

        
#############################################################################################################################################
# oof regression ############################################################################################################################
#############################################################################################################################################

#############################################################################################################################################
# Created by qqgeogor
# https://www.kaggle.com/qqgeogor
#############################################################################################################################################

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt,pow
import itertools
import math
from random import random,shuffle,uniform,seed
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
import pickle
import sys

seed(1024)

def data_generator(path,no_norm=False,task='c'):
    data = open(path,'r')
    for row in data:
        row = row.strip().split(" ")
        y = float(row[0])
        row = row[1:]
        x = []
        for feature in row:
            feature = feature.split(":")
            idx = int(feature[0])
            value = float(feature[1])
            x.append([idx,value])

        if not no_norm:
            r = 0.0
            for i in range(len(x)):
                r+=x[i][1]*x[i][1]
            for i in range(len(x)):
                x[i][1] /=r
        # if task=='c':
        #     if y ==0.0:
        #         y = -1.0

        yield x,y


def dot(u,v):
    u_v = 0.
    len_u = len(u)
    for idx in range(len_u):
        uu = u[idx]
        vv = v[idx]
        u_v+=uu*vv
    return u_v

def mse_loss_function(y,p):
    return (y - p)**2

def mae_loss_function(y,p):
    y = exp(y)
    p = exp(p)
    return abs(y - p)

def log_loss_function(y,p):
    return -(y*log(p)+(1-y)*log(1-p))

def exponential_loss_function(y,p):
    return log(1+exp(-y*p))

def sigmoid(inX):
    return 1/(1+exp(-inX))

def bounded_sigmoid(inX):
    return 1. / (1. + exp(-max(min(inX, 35.), -35.)))


class SGD(object):
    def __init__(self,lr=0.001,momentum=0.9,nesterov=True,adam=False,l2=0.0,l2_fm=0.0,l2_bias=0.0,ini_stdev= 0.01,dropout=0.5,task='c',n_components=4,nb_epoch=5,interaction=False,no_norm=False):
        self.W = []
        self.V = []        
        self.bias = uniform(-ini_stdev, ini_stdev)
        self.n_components=n_components
        self.lr = lr
        self.l2 = l2
        self.l2_fm = l2_fm
        self.l2_bias = l2_bias
        self.momentum = momentum
        self.nesterov = nesterov
        self.adam = adam
        self.nb_epoch = nb_epoch
        self.ini_stdev = ini_stdev
        self.task = task
        self.interaction = interaction
        self.dropout = dropout
        self.no_norm = no_norm
        if self.task!='c':
            # self.loss_function = mse_loss_function
            self.loss_function = mae_loss_function
        else:
            # self.loss_function = exponential_loss_function
            self.loss_function = log_loss_function

    def preload(self,train,test):
        train = data_generator(train,self.no_norm,self.task)
        dim = 0
        count = 0
        for x,y in train:
            for i in x:
                idx,value = i
                if idx >dim:
                    dim = idx
            count+=1
        print('Training samples:',count)
        test = data_generator(test,self.no_norm,self.task)
        count=0
        for x,y in test:
            for i in x:
                idx,value = i
                if idx >dim:
                    dim = idx
            count+=1
        print('Testing samples:',count)
        
        dim = dim+1
        print("Number of features:",dim)
        
        self.W = [uniform(-self.ini_stdev, self.ini_stdev) for _ in range(dim)]
        self.Velocity_W = [0.0 for _ in range(dim)]
        
        
        self.V = [[uniform(-self.ini_stdev, self.ini_stdev) for _ in range(self.n_components)] for _ in range(dim)]
        self.Velocity_V = [[0.0 for _ in range(self.n_components)] for _ in range(dim)]
        
        self.Velocity_bias = 0.0
        
        self.dim = dim
        
        
    def adam_init(self):
        self.iterations = 0
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon=1e-8
        self.decay = 0.
        self.inital_decay = self.decay 

        dim =self.dim

        self.m_W = [0.0 for _ in range(dim)]
        self.v_W = [0.0 for _ in range(dim)]

        self.m_V = [[0.0 for _ in range(self.n_components)] for _ in range(dim)]
        self.v_V = [[0.0 for _ in range(self.n_components)] for _ in range(dim)]

        self.m_bias = 0.0
        self.v_bias = 0.0


    def adam_update(self,lr,x,residual):

        if 0.<self.dropout<1.:
            self.droupout_x(x)
        
        lr = self.lr
        if self.inital_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1

        lr_t = lr * sqrt(1. - pow(self.beta_2, t)) / (1. - pow(self.beta_1, t))
        
        for sample in x:
            idx,value = sample
            g = residual*value

            m = self.m_W[idx]
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g

            v = self.v_W[idx]
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * (g**2)

            p = self.W[idx]
            p_t = p - lr_t *m_t / (sqrt(v_t) + self.epsilon)

            if self.l2>0:
                p_t = p_t - lr_t*self.l2*p

            self.m_W[idx] = m_t
            self.v_W[idx] = v_t
            self.W[idx] = p_t

        if self.interaction:
            self._adam_update_fm(lr_t,x,residual)


        m = self.m_bias
        m_t = (self.beta_1 * m) + (1. - self.beta_1)*residual

        v = self.v_bias
        v_t = (self.beta_2 * v) + (1. - self.beta_2)*(residual**2)

        p = self.bias
        p_t = p - lr_t * m_t / (sqrt(v_t) + self.epsilon)
        if self.l2_bias>0:
            pt = pt - lr_t * self.l2_bias*p

        self.m_bias = m_t
        self.v_bias = v_t
        self.bias = p_t

        self.iterations+=1

    def _adam_update_fm(self,lr_t,x,residual):
        len_x = len(x)
        sum_f_dict = self.sum_f_dict
        n_components = self.n_components
        for f in range(n_components):
            for i in range(len_x):
                idx_i,value_i = x[i]
                v = self.V[idx_i][f]
                sum_f = sum_f_dict[f]
                g = (sum_f*value_i - v *value_i*value_i)*residual

                m = self.m_V[idx_i][f]
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * g

                v = self.v_V[idx_i][f]
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * (g**2)

                p = self.V[idx_i][f]
                p_t = p - lr_t * m_t / (sqrt(v_t) + self.epsilon)

                if self.l2_fm>0:
                    p_t = p_t - lr_t * self.l2_fm*p

                self.m_V[idx_i][f] = m_t
                self.v_V[idx_i][f] = v_t
                self.V[idx_i][f] = p_t

    def droupout_x(self,x):
        new_x = []
        for i, var in enumerate(x):
            if random() > self.dropout:
                del x[i]

    def _predict_fm(self,x):
        len_x = len(x)
        n_components = self.n_components
        pred = 0.0
        self.sum_f_dict = {}
        for f in range(n_components):
            sum_f = 0.0
            sum_sqr_f = 0.0
            for i in range(len_x):
                idx_i,value_i = x[i]
                d = self.V[idx_i][f] * value_i
                sum_f +=d
                sum_sqr_f +=d*d
            pred+= 0.5 * (sum_f*sum_f - sum_sqr_f);
            self.sum_f_dict[f] = sum_f
        return pred

    def _predict_one(self,x):
        pred = self.bias
        # pred = 0.0
        for idx,value in x:
            pred+=self.W[idx]*value
        
        if self.interaction:
            pred+=self._predict_fm(x)

        if self.task=='c':
            pred = bounded_sigmoid(pred)
        return pred


    def _update_fm(self,lr,x,residual):
        len_x = len(x)
        sum_f_dict = self.sum_f_dict
        n_components = self.n_components
        for f in range(n_components):
            for i in range(len_x):
                idx_i,value_i = x[i]
                sum_f = sum_f_dict[f]
                v = self.V[idx_i][f]
                grad = (sum_f*value_i - v *value_i*value_i)*residual
                
                self.Velocity_V[idx_i][f] = self.momentum * self.Velocity_V[idx_i][f] - lr * grad
                if self.nesterov:
                    self.Velocity_V[idx_i][f] = self.momentum * self.Velocity_V[idx_i][f] - lr * grad
                self.V[idx_i][f] = self.V[idx_i][f] + self.Velocity_V[idx_i][f] - lr*self.l2_fm*self.V[idx_i][f]



    def update(self,lr,x,residual):

        if 0.<self.dropout<1.:
            self.droupout_x(x)

        for sample in x:
            idx,value = sample
            grad = residual*value
            self.Velocity_W[idx] =  self.momentum * self.Velocity_W[idx] - lr * grad
            if self.nesterov:
                 self.Velocity_W[idx] = self.momentum * self.Velocity_W[idx] - lr * grad
            self.W[idx] = self.W[idx] + self.Velocity_W[idx] - lr*self.l2*self.W[idx]
            
        if self.interaction:
            self._update_fm(lr,x,residual)

        self.Velocity_bias = self.momentum*self.Velocity_bias - lr*residual
        if self.nesterov:
            self.Velocity_bias = self.momentum*self.Velocity_bias - lr*residual
        self.bias = self.bias +self.Velocity_bias - lr*self.l2_bias*self.bias

    def predict(self,path,out):

        data = data_generator(path,self.no_norm,self.task)
        y_preds =[]
        with open(out, 'w') as outfile:
            ID = 0
            outfile.write('%s,%s\n' % ('test_id', 'is_duplicate'))
            for d in data:
                x,y = d
                p = self._predict_one(x)
                outfile.write('%s,%s\n' % (ID, str(p)))
                ID+=1


    def validate(self,path):
        data = data_generator(path,self.no_norm,self.task)
        loss = 0.0
        count = 0.0

        for d in data:
            x,y = d
            p = self._predict_one(x)
            loss+=self.loss_function(y,p)
            count+=1
        return loss/count

    def save_weights(self):
        weights = []
        weights.append(self.W)
        weights.append(self.V)
        weights.append(self.bias)
        # weights.append(self.Velocity_W)
        # weights.append(self.Velocity_V)
        weights.append(self.dim)
        pickle.dump(weights,open('sgd_fm.pkl','wb'))

    def load_weights(self):
        weights = pickle.load(open('sgd_fm.pkl','rb'))
        self.W = weights[0]
        self.V = weights[1]
        self.bias = weights[2]
        # self.Velocity_W = weights[3]
        # self.Velocity_V = weights[4]
        self.dim = weights[3]
        

    def train(self,path,valid_path = None,in_memory=False):

        start = datetime.now()
        lr = self.lr
        if self.adam:
            self.adam_init()
            self.update = self.adam_update

        if in_memory:
            data = data_generator(path,self.no_norm,self.task)
            data = [d for d in data]
        best_loss = 999999
        best_epoch = 0
        for epoch in range(1,self.nb_epoch+1):
            if not in_memory:
                data = data_generator(path,self.no_norm,self.task)
            train_loss = 0.0
            train_count = 0
            for x,y in data:
                p = self._predict_one(x)
                if self.task!='c':                    
                    residual = -(y-p)
                else:
                    # residual = -y*(1.0-1.0/(1.0+exp(-y*p)));
                    residual = -(y-p)

                self.update(lr,x,residual)
                if train_count%50000==0:
                    if train_count ==0:
                        print '\ttrain_count: %s, current loss: %.6f'%(train_count,0.0)
                    else:
                        print '\ttrain_count: %s, current loss: %.6f'%(train_count,train_loss/train_count)

                train_loss += self.loss_function(y,p)
                train_count += 1

            epoch_end = datetime.now()
            duration = epoch_end-start
            
            if valid_path:
                valid_loss = self.validate(valid_path)
                print('Epoch: %s, train loss: %.6f, valid loss: %.6f, time: %s'%(epoch,train_loss/train_count,valid_loss,duration))
                if valid_loss<best_loss:
                    best_loss = valid_loss
                    self.save_weights()
                    print 'save_weights'
            else:
                print('Epoch: %s, train loss: %.6f, time: %s'%(epoch,train_loss/train_count,duration))

#############################################################################################################################################
# Created by qqgeogor
# https://www.kaggle.com/qqgeogor
#############################################################################################################################################


sgd = SGD(lr=0.001,adam=True,dropout=0.8,l2=0.00,l2_fm=0.00,task='c',n_components=1,nb_epoch=30,interaction=True,no_norm=False)
sgd.preload(inDir+'/input/X_trainingSet.svm',inDir+'/input/X_testingSet.svm')
# sgd.load_weights()
sgd.train(inDir+'/input/X_train_tfidf.svm',inDir+'/input/X_test_tfidf.svm',in_memory=False)
sgd.load_weights()
sgd.predict(inDir+'/input/X_test_tfidf.svm',out='valid.csv')
print sgd.validate(inDir+'/input/X_test_tfidf.svm')
sgd.predict(inDir+'/input/X_t_tfidf.svm',out='out.csv')

def train_libfm_regression(trainingSet, testingSet,feature_names,i,nbags,ModelName,xgbParameters, num_rounds):
    
    X_build = trainingSet[trainingSet['CVindices'] != i]
    X_valid   = trainingSet[trainingSet['CVindices'] == i]
    
    dump_svmlight_file(X_build[feature_names],X_build['is_duplicate'],inDir+"/input/X_build_trainingSet.svm")
    dump_svmlight_file(X_valid[feature_names],X_valid['is_duplicate'],inDir+"/input/X_valid_testingSet.svm")
                     
    pred_cv = np.zeros(X_valid.shape[0])
    pred_test = np.zeros(testingSet.shape[0])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros(X_valid.shape[0])
        sgd = SGD(lr=0.001,adam=True,dropout=0.8,l2=0.00,l2_fm=0.00,task='c',n_components=1,nb_epoch=5,interaction=True,no_norm=False)
        sgd.preload(inDir+'/input/X_trainingSet.svm',inDir+'/input/X_testingSet.svm')

        sgd.train(inDir+'/input/X_build_trainingSet.svm',inDir+'/input/X_valid_testingSet.svm',in_memory=False)
        sgd.load_weights()   
        sgd.predict(inDir+'/input/X_valid_testingSet.svm',out=inDir+'/input/X_valid.csv')
        sgd.predict(inDir+'/input/X_testingSet.svm',out=inDir+'/input/X_testingSet.csv')
        X_valid_file = inDir+'/input/X_valid.csv'
        X_valid = pd.read_csv(X_valid_file)
        bag_cv  = X_valid['is_duplicate']  
        X_testingSet_file = inDir+'/input/X_testingSet.csv'
        X_testingSet = pd.read_csv(X_testingSet_file)
        pred_test += X_testingSet['is_duplicate']       
        
        pred_cv += bag_cv
        bag_score = log_loss(X_valid['is_duplicate'], bag_cv)
        print('bag ', j, '- logloss:', bag_score)
    pred_cv /= nbags
    pred_test/= nbags
    fold_score = log_loss(X_valid['is_duplicate'], pred_cv)
    print('Fold ', i, '- logloss:', fold_score)
    
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = ["is_duplicate"]
    pred_cv["id"] = X_valid.id.values
    
    sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
    pred_cv[["id","is_duplicate"]].to_csv(sub_valfile, index=False)
    
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["is_duplicate"]
    pred_test["test_id"] = testingSet.test_id.values
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '-test' + '.csv'
    pred_test[["test_id","is_duplicate"]].to_csv(sub_file, index=False)
    del pred_cv
    del pred_test



def fulltrain_libfm_regression(trainingSet, testingSet,feature_names,nbags,ModelName,xgbParameters, num_rounds):
    
    predfull_test = np.zeros(testingSet.shape[0]) 
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        sgd = SGD(lr=0.001,adam=True,dropout=0.8,l2=0.00,l2_fm=0.00,task='c',n_components=1,nb_epoch=30,interaction=True,no_norm=False)
        sgd.preload(inDir+'/input/X_trainingSet.svm',inDir+'/input/X_testingSet.svm')
        sgd.train(inDir+'/input/X_trainingSet.svm',valid_path = None,in_memory=False)
        sgd.predict(inDir+'/input/X_testingSet.svm',out=inDir+'/input/X_testingSet.csv')            
        X_testingSet_file = inDir+'/input/X_testingSet.csv'
        X_testingSet = pd.read_csv(X_testingSet_file)
        predfull_test += X_testingSet['is_duplicate']         
        
    predfull_test/= nbags
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["is_duplicate"]
    predfull_test["test_id"] = testingSet.test_id.values
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
    predfull_test[["test_id","is_duplicate"]].to_csv(sub_file, index=False)
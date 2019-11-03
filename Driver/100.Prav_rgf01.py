# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 13:01:43 2017

@author: SriPrav
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgbm
from rgf.sklearn import RGFRegressor



random_state = 2017
np.random.RandomState(random_state)


inDir = 'C:\\Users\\SriPrav\\Documents\\R\\36Driver'
train = pd.read_csv(inDir+'/input/train.csv')
test = pd.read_csv(inDir+'/input/test.csv')

trainfoldSource = pd.read_csv(inDir+'/input/Prav_5folds_CVindices.csv')
train = pd.merge(train, trainfoldSource, how='left',on="id")

train.groupby(['CVindices'])[['target']].sum()

# Preprocessing (Forza Baseline)
id_test = test['id'].values

col = [c for c in train.columns if c not in ['id','target']]
col = [c for c in col if not c.startswith('ps_calc_')]


def recon(reg):
    integer = int(np.round((40*reg)**2)) 
    for a in range(32):
        if (integer - a) % 31 == 0:
            A = a
    M = (integer - A)//31
    return A, M
train['ps_reg_A'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])
train['ps_reg_M'] = train['ps_reg_03'].apply(lambda x: recon(x)[1])
train['ps_reg_A'].replace(19,-1, inplace=True)
train['ps_reg_M'].replace(51,-1, inplace=True)
test['ps_reg_A'] = test['ps_reg_03'].apply(lambda x: recon(x)[0])
test['ps_reg_M'] = test['ps_reg_03'].apply(lambda x: recon(x)[1])
test['ps_reg_A'].replace(19,-1, inplace=True)
test['ps_reg_M'].replace(51,-1, inplace=True)

train = train.replace(-1, np.NaN)
d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
train = train.fillna(-1)
one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target','CVindices']}


def transform(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target','CVindices']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol:
        if '_bin' not in c:
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)

    for c in one_hot:
        if len(one_hot[c])>2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df



#cat("Feature engineering")
#data[, amount_nas := rowSums(data == -1, na.rm = T)]
#data[, high_nas := ifelse(amount_nas>4,1,0)]
#data[, ps_car_13_ps_reg_03 := ps_car_13*ps_reg_03]
#data[, ps_reg_mult := ps_reg_01*ps_reg_02*ps_reg_03]
#data[, ps_ind_bin_sum := ps_ind_06_bin+ps_ind_07_bin+ps_ind_08_bin+ps_ind_09_bin+ps_ind_10_bin+ps_ind_11_bin+ps_ind_12_bin+ps_ind_13_bin+ps_ind_16_bin+ps_ind_17_bin+ps_ind_18_bin]

train = transform(train)
test = transform(test)

col = [c for c in train.columns if c not in ['id','target','CVindices']]
col = [c for c in col if not c.startswith('ps_calc_')]

dups = train[train.duplicated(subset=col, keep=False)]

train = train[~(train['id'].isin(dups['id'].values))]

trainingSet = train
testingSet  = test


feature_names = col

seed = 2017
folds = 5
nbags = 1
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2017
verboseeval=250

rgfModelName = 'rgf001'

for i in range(1, folds+1):
        train_regression("rgf",trainingSet, testingSet,feature_names,i,nbags,rgfModelName,current_seed)
fulltrain_regression("rgf",trainingSet, testingSet,feature_names,nbags,rgfModelName,current_seed)

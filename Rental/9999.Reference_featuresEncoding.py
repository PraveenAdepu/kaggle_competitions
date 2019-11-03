# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 13:01:05 2017

@author: PAdepu
"""
import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
import random
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#input data
in_Dir="C:/Users/padepu/Documents/R/21SigmaRental"

train_file = in_Dir + "/input/train.json"
test_file = in_Dir + "/input/test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)
    
#basic features
train_df["price_t"] =train_df["price"]/train_df["bedrooms"]
test_df["price_t"] = test_df["price"]/test_df["bedrooms"] 
train_df["room_sum"] = train_df["bedrooms"]+train_df["bathrooms"] 
test_df["room_sum"] = test_df["bedrooms"]+test_df["bathrooms"] 

# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))


features_to_use=["bathrooms", "bedrooms", "latitude", "longitude", "price","price_t","num_photos", "num_features", "num_description_words","listing_id"]

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

index=list(range(train_df.shape[0]))
random.shuffle(index)
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

# 0000abd7518b94c35a90d64b56fbf3e6 # 3,5,0 -- total 4,8,0, test 1,3,0
#temp_manager = train_df[train_df['manager_id'] == '0000abd7518b94c35a90d64b56fbf3e6']
#i = 0
##################################################################################################
for i in range(5):
    manager_level={}
    for j in train_df['manager_id'].values:
        manager_level[j]=[0,0,0]
    test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            manager_level[temp['manager_id']][0]+=1
        if temp['interest_level']=='medium':
            manager_level[temp['manager_id']][1]+=1
        if temp['interest_level']=='high':
            manager_level[temp['manager_id']][2]+=1
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(manager_level[temp['manager_id']])!=0:
            a[j]=manager_level[temp['manager_id']][0]*1.0/sum(manager_level[temp['manager_id']])
            b[j]=manager_level[temp['manager_id']][1]*1.0/sum(manager_level[temp['manager_id']])
            c[j]=manager_level[temp['manager_id']][2]*1.0/sum(manager_level[temp['manager_id']])
train_df['manager_level_low']=a
train_df['manager_level_medium']=b
train_df['manager_level_high']=c



a=[]
b=[]
c=[]
manager_level={}
for j in train_df['manager_id'].values:
    manager_level[j]=[0,0,0]
for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        manager_level[temp['manager_id']][0]+=1
    if temp['interest_level']=='medium':
        manager_level[temp['manager_id']][1]+=1
    if temp['interest_level']=='high':
        manager_level[temp['manager_id']][2]+=1

for i in test_df['manager_id'].values:
    if i not in manager_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(manager_level[i][0]*1.0/sum(manager_level[i]))
        b.append(manager_level[i][1]*1.0/sum(manager_level[i]))
        c.append(manager_level[i][2]*1.0/sum(manager_level[i]))
test_df['manager_level_low']=a
test_df['manager_level_medium']=b
test_df['manager_level_high']=c

features_to_use.append('manager_level_low') 
features_to_use.append('manager_level_medium') 
features_to_use.append('manager_level_high')
#######################################################################################################
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)
for i in range(5):
    building_level={}
    for j in train_df['building_id'].values:
        building_level[j]=[0,0,0]
    test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            building_level[temp['building_id']][0]+=1
        if temp['interest_level']=='medium':
            building_level[temp['building_id']][1]+=1
        if temp['interest_level']=='high':
            building_level[temp['building_id']][2]+=1
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(building_level[temp['building_id']])!=0:
            a[j]=building_level[temp['building_id']][0]*1.0/sum(building_level[temp['building_id']])
            b[j]=building_level[temp['building_id']][1]*1.0/sum(building_level[temp['building_id']])
            c[j]=building_level[temp['building_id']][2]*1.0/sum(building_level[temp['building_id']])
train_df['building_level_low']=a
train_df['building_level_medium']=b
train_df['building_level_high']=c



a=[]
b=[]
c=[]
building_level={}
for j in train_df['building_id'].values:
    building_level[j]=[0,0,0]
for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        building_level[temp['building_id']][0]+=1
    if temp['interest_level']=='medium':
        building_level[temp['building_id']][1]+=1
    if temp['interest_level']=='high':
        building_level[temp['building_id']][2]+=1

for i in test_df['building_id'].values:
    if i not in building_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(building_level[i][0]*1.0/sum(building_level[i]))
        b.append(building_level[i][1]*1.0/sum(building_level[i]))
        c.append(building_level[i][2]*1.0/sum(building_level[i]))
test_df['building_level_low']=a
test_df['building_level_medium']=b
test_df['building_level_high']=c

features_to_use.append('building_level_low') 
features_to_use.append('building_level_medium') 
features_to_use.append('building_level_high')

#######################################################################################################


#train_df['manager_level_low'].head(5)
#train_df['manager_level_medium'].head(5)
#train_df['manager_level_high'].head(5)
categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)
            
#######################################################################################################
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)
for i in range(5):
    street_level={}
    for j in train_df['street_address'].values:
        street_level[j]=[0,0,0]
    test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            street_level[temp['street_address']][0]+=1
        if temp['interest_level']=='medium':
            street_level[temp['street_address']][1]+=1
        if temp['interest_level']=='high':
            street_level[temp['street_address']][2]+=1
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(street_level[temp['street_address']])!=0:
            a[j]=street_level[temp['street_address']][0]*1.0/sum(street_level[temp['street_address']])
            b[j]=street_level[temp['street_address']][1]*1.0/sum(street_level[temp['street_address']])
            c[j]=street_level[temp['street_address']][2]*1.0/sum(street_level[temp['street_address']])
train_df['street_level_low']=a
train_df['street_level_medium']=b
train_df['street_level_high']=c



a=[]
b=[]
c=[]
street_level={}
for j in train_df['street_address'].values:
    street_level[j]=[0,0,0]
for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        street_level[temp['street_address']][0]+=1
    if temp['interest_level']=='medium':
        street_level[temp['street_address']][1]+=1
    if temp['interest_level']=='high':
        street_level[temp['street_address']][2]+=1

for i in test_df['street_address'].values:
    if i not in street_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(street_level[i][0]*1.0/sum(street_level[i]))
        b.append(street_level[i][1]*1.0/sum(street_level[i]))
        c.append(street_level[i][2]*1.0/sum(street_level[i]))
test_df['street_level_low']=a
test_df['street_level_medium']=b
test_df['street_level_high']=c

features_to_use.append('street_level_low') 
features_to_use.append('street_level_medium') 
features_to_use.append('street_level_high')

#######################################################################################################

train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
print(train_df["features"].head())
tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])

 
train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape)

cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
        break

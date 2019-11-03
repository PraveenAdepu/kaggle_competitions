import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string

inDir = 'C:/Users/SriPrav/Documents/R/21Rental'

train_file = inDir + "/input/train.json"
test_file = inDir + "/input/test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape) # (49352, 15)
print(test_df.shape)  # (74659, 14)


cv_file = inDir + "/CVSchema/Prav_CVindices_5folds.csv"
CV_Schema = pd.read_csv(cv_file)

CV_Schema.head()
del CV_Schema["interest_level"]


train_df = pd.merge(train_df, CV_Schema, how = 'left', on = 'listing_id')

train_df.head(2)
##################################################################################################
folds = 5
##################################################################################################
#######################################################################################################
#######################################################################################################
train_df["managerPrice"] = train_df["manager_id"].map(str) + train_df["price"].map(str)
test_df["managerPrice"] = test_df["manager_id"].map(str) + test_df["price"].map(str)

##################################################################################################

##################################################################################################
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

for i in range(1, folds+1):
    manager_price={}
    for j in train_df['managerPrice'].values:
        manager_price[j]=[0,0,0]
    test_index=train_df[train_df['CVindices'] == i].index.tolist() #index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=train_df[train_df['CVindices'] != i].index.tolist() # list(set(index).difference(test_index))

    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            manager_price[temp['managerPrice']][0]+=1
        if temp['interest_level']=='medium':
            manager_price[temp['managerPrice']][1]+=1
        if temp['interest_level']=='high':
            manager_price[temp['managerPrice']][2]+=1
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(manager_price[temp['managerPrice']])!=0:
            a[j]=manager_price[temp['managerPrice']][0]*1.0/sum(manager_price[temp['managerPrice']])
            b[j]=manager_price[temp['managerPrice']][1]*1.0/sum(manager_price[temp['managerPrice']])
            c[j]=manager_price[temp['managerPrice']][2]*1.0/sum(manager_price[temp['managerPrice']])
train_df['manager_price_low']   =a
train_df['manager_price_medium']=b
train_df['manager_price_high']  =c



a=[]
b=[]
c=[]
manager_price={}
for j in train_df['managerPrice'].values:
    manager_price[j]=[0,0,0]
for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        manager_price[temp['managerPrice']][0]+=1
    if temp['interest_level']=='medium':
        manager_price[temp['managerPrice']][1]+=1
    if temp['interest_level']=='high':
        manager_price[temp['managerPrice']][2]+=1

for i in test_df['managerPrice'].values:
    if i not in manager_price.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(manager_price[i][0]*1.0/sum(manager_price[i]))
        b.append(manager_price[i][1]*1.0/sum(manager_price[i]))
        c.append(manager_price[i][2]*1.0/sum(manager_price[i]))
test_df['manager_price_low']=a
test_df['manager_price_medium']=b
test_df['manager_price_high']=c

#######################################################################################################
train_df["managerbathroom"] = train_df["manager_id"].map(str) + train_df["bathrooms"].map(str)
test_df["managerbathroom"] = test_df["manager_id"].map(str) + test_df["bathrooms"].map(str)

##################################################################################################

##################################################################################################
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

for i in range(1, folds+1):
    manager_bathroom={}
    for j in train_df['managerbathroom'].values:
        manager_bathroom[j]=[0,0,0]
    test_index=train_df[train_df['CVindices'] == i].index.tolist() #index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=train_df[train_df['CVindices'] != i].index.tolist() # list(set(index).difference(test_index))

    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            manager_bathroom[temp['managerbathroom']][0]+=1
        if temp['interest_level']=='medium':
            manager_bathroom[temp['managerbathroom']][1]+=1
        if temp['interest_level']=='high':
            manager_bathroom[temp['managerbathroom']][2]+=1
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(manager_bathroom[temp['managerbathroom']])!=0:
            a[j]=manager_bathroom[temp['managerbathroom']][0]*1.0/sum(manager_bathroom[temp['managerbathroom']])
            b[j]=manager_bathroom[temp['managerbathroom']][1]*1.0/sum(manager_bathroom[temp['managerbathroom']])
            c[j]=manager_bathroom[temp['managerbathroom']][2]*1.0/sum(manager_bathroom[temp['managerbathroom']])
train_df['manager_bathroom_low']   =a
train_df['manager_bathroom_medium']=b
train_df['manager_bathroom_high']  =c



a=[]
b=[]
c=[]
manager_bathroom={}
for j in train_df['managerbathroom'].values:
    manager_bathroom[j]=[0,0,0]
for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        manager_bathroom[temp['managerbathroom']][0]+=1
    if temp['interest_level']=='medium':
        manager_bathroom[temp['managerbathroom']][1]+=1
    if temp['interest_level']=='high':
        manager_bathroom[temp['managerbathroom']][2]+=1

for i in test_df['managerbathroom'].values:
    if i not in manager_bathroom.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(manager_bathroom[i][0]*1.0/sum(manager_bathroom[i]))
        b.append(manager_bathroom[i][1]*1.0/sum(manager_bathroom[i]))
        c.append(manager_bathroom[i][2]*1.0/sum(manager_bathroom[i]))
test_df['manager_bathroom_low']=a
test_df['manager_bathroom_medium']=b
test_df['manager_bathroom_high']=c



features_to_use  = ["listing_id",
                    'manager_price_low','manager_price_medium','manager_price_high',
                    'manager_bathroom_low','manager_bathroom_medium','manager_bathroom_high']
                    
#######################################################################################################
train_df = train_df.fillna(0)   
test_df = test_df.fillna(0) 

train_df[features_to_use].head()
test_df[features_to_use].head()

#######################################################################################################


sub_valfile = 'C:/Users/SriPrav/Documents/R/21Rental/input/Prav_train_features12.csv'
train_df[features_to_use].to_csv(sub_valfile, index=False)

sub_valfile = 'C:/Users/SriPrav/Documents/R/21Rental/input/Prav_test_features12.csv'
test_df[features_to_use].to_csv(sub_valfile, index=False)

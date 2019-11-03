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


##################################################################################################
folds = 5
##################################################################################################
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

for i in range(1, folds+1):
    manager_level={}
    for j in train_df['manager_id'].values:
        manager_level[j]=[0,0,0]
    test_index=train_df[train_df['CVindices'] == i].index.tolist() #index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=train_df[train_df['CVindices'] != i].index.tolist() # list(set(index).difference(test_index))

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
train_df['manager_level_low']   =a
train_df['manager_level_medium']=b
train_df['manager_level_high']  =c



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


#######################################################################################################
#######################################################################################################
train_df["managerBed"] = train_df["manager_id"].map(str) + train_df["bedrooms"].map(str)
test_df["managerBed"] = test_df["manager_id"].map(str) + test_df["bedrooms"].map(str)

#train_df["manager_id"].head()
#train_df["bedrooms"].head()
#train_df["managerBed"].head()
##################################################################################################

##################################################################################################
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

for i in range(1, folds+1):
    manager_bed={}
    for j in train_df['managerBed'].values:
        manager_bed[j]=[0,0,0]
    test_index=train_df[train_df['CVindices'] == i].index.tolist() #index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=train_df[train_df['CVindices'] != i].index.tolist() # list(set(index).difference(test_index))

    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            manager_bed[temp['managerBed']][0]+=1
        if temp['interest_level']=='medium':
            manager_bed[temp['managerBed']][1]+=1
        if temp['interest_level']=='high':
            manager_bed[temp['managerBed']][2]+=1
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(manager_bed[temp['managerBed']])!=0:
            a[j]=manager_bed[temp['managerBed']][0]*1.0/sum(manager_bed[temp['managerBed']])
            b[j]=manager_bed[temp['managerBed']][1]*1.0/sum(manager_bed[temp['managerBed']])
            c[j]=manager_bed[temp['managerBed']][2]*1.0/sum(manager_bed[temp['managerBed']])
train_df['manager_bed_low']   =a
train_df['manager_bed_medium']=b
train_df['manager_bed_high']  =c



a=[]
b=[]
c=[]
manager_bed={}
for j in train_df['managerBed'].values:
    manager_bed[j]=[0,0,0]
for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        manager_bed[temp['managerBed']][0]+=1
    if temp['interest_level']=='medium':
        manager_bed[temp['managerBed']][1]+=1
    if temp['interest_level']=='high':
        manager_bed[temp['managerBed']][2]+=1

for i in test_df['managerBed'].values:
    if i not in manager_bed.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(manager_bed[i][0]*1.0/sum(manager_bed[i]))
        b.append(manager_bed[i][1]*1.0/sum(manager_bed[i]))
        c.append(manager_bed[i][2]*1.0/sum(manager_bed[i]))
test_df['manager_bed_low']=a
test_df['manager_bed_medium']=b
test_df['manager_bed_high']=c

#######################################################################################################
ny_lat = 40.785091
ny_lon = -73.968285

train_df['distance_to_city'] = np.sqrt((train_df['longitude'] - ny_lon)**2  + (train_df['latitude'] - ny_lat)**2)
test_df['distance_to_city'] = np.sqrt((test_df['longitude'] - ny_lon)**2  + (test_df['latitude'] - ny_lat)**2)

train_df['distanceMeasure'] = train_df['distance_to_city']*100
test_df['distanceMeasure'] = test_df['distance_to_city']*100

train_df['distanceMeasure'] = train_df['distanceMeasure'].astype(int)
test_df['distanceMeasure'] = test_df['distanceMeasure'].astype(int)
##################################################################################################

a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

for i in range(1, folds+1):
    distance_level={}
    for j in train_df['distanceMeasure'].values:
        distance_level[j]=[0,0,0]
    test_index=train_df[train_df['CVindices'] == i].index.tolist() #index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=train_df[train_df['CVindices'] != i].index.tolist() # list(set(index).difference(test_index))

    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            distance_level[temp['distanceMeasure']][0]+=1
        if temp['interest_level']=='medium':
            distance_level[temp['distanceMeasure']][1]+=1
        if temp['interest_level']=='high':
            distance_level[temp['distanceMeasure']][2]+=1
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(distance_level[temp['distanceMeasure']])!=0:
            a[j]=distance_level[temp['distanceMeasure']][0]*1.0/sum(distance_level[temp['distanceMeasure']])
            b[j]=distance_level[temp['distanceMeasure']][1]*1.0/sum(distance_level[temp['distanceMeasure']])
            c[j]=distance_level[temp['distanceMeasure']][2]*1.0/sum(distance_level[temp['distanceMeasure']])
train_df['distance_level_low']   =a
train_df['distance_level_medium']=b
train_df['distance_level_high']  =c



a=[]
b=[]
c=[]
distance_level={}
for j in train_df['distanceMeasure'].values:
    distance_level[j]=[0,0,0]
for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        distance_level[temp['distanceMeasure']][0]+=1
    if temp['interest_level']=='medium':
        distance_level[temp['distanceMeasure']][1]+=1
    if temp['interest_level']=='high':
        distance_level[temp['distanceMeasure']][2]+=1

for i in test_df['distanceMeasure'].values:
    if i not in distance_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(distance_level[i][0]*1.0/sum(distance_level[i]))
        b.append(distance_level[i][1]*1.0/sum(distance_level[i]))
        c.append(distance_level[i][2]*1.0/sum(distance_level[i]))
test_df['distance_level_low']=a
test_df['distance_level_medium']=b
test_df['distance_level_high']=c


#######################################################################################################
train_df["distanceMeasureBed"] = train_df["distanceMeasure"].map(str) + train_df["bedrooms"].map(str)
test_df["distanceMeasureBed"] = test_df["distanceMeasure"].map(str) + test_df["bedrooms"].map(str)

#######################################################################################################

##################################################################################################

a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

for i in range(1, folds+1):
    distanceBed_level={}
    for j in train_df['distanceMeasureBed'].values:
        distanceBed_level[j]=[0,0,0]
    test_index=train_df[train_df['CVindices'] == i].index.tolist() #index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=train_df[train_df['CVindices'] != i].index.tolist() # list(set(index).difference(test_index))

    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            distanceBed_level[temp['distanceMeasureBed']][0]+=1
        if temp['interest_level']=='medium':
            distanceBed_level[temp['distanceMeasureBed']][1]+=1
        if temp['interest_level']=='high':
            distanceBed_level[temp['distanceMeasureBed']][2]+=1
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(distanceBed_level[temp['distanceMeasureBed']])!=0:
            a[j]=distanceBed_level[temp['distanceMeasureBed']][0]*1.0/sum(distanceBed_level[temp['distanceMeasureBed']])
            b[j]=distanceBed_level[temp['distanceMeasureBed']][1]*1.0/sum(distanceBed_level[temp['distanceMeasureBed']])
            c[j]=distanceBed_level[temp['distanceMeasureBed']][2]*1.0/sum(distanceBed_level[temp['distanceMeasureBed']])
train_df['distanceBed_level_low']   =a
train_df['distanceBed_level_medium']=b
train_df['distanceBed_level_high']  =c



a=[]
b=[]
c=[]
distanceBed_level={}
for j in train_df['distanceMeasureBed'].values:
    distanceBed_level[j]=[0,0,0]
for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        distanceBed_level[temp['distanceMeasureBed']][0]+=1
    if temp['interest_level']=='medium':
        distanceBed_level[temp['distanceMeasureBed']][1]+=1
    if temp['interest_level']=='high':
        distanceBed_level[temp['distanceMeasureBed']][2]+=1

for i in test_df['distanceMeasureBed'].values:
    if i not in distanceBed_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(distanceBed_level[i][0]*1.0/sum(distanceBed_level[i]))
        b.append(distanceBed_level[i][1]*1.0/sum(distanceBed_level[i]))
        c.append(distanceBed_level[i][2]*1.0/sum(distanceBed_level[i]))
test_df['distanceBed_level_low']=a
test_df['distanceBed_level_medium']=b
test_df['distanceBed_level_high']=c


#######################################################################################################
##################################################################################################

a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

for i in range(1, folds+1):
    building_level={}
    for j in train_df['building_id'].values:
        building_level[j]=[0,0,0]
    test_index=train_df[train_df['CVindices'] == i].index.tolist() #index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=train_df[train_df['CVindices'] != i].index.tolist() # list(set(index).difference(test_index))

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
train_df['building_level_low']   =a
train_df['building_level_medium']=b
train_df['building_level_high']  =c



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


#######################################################################################################



categorical = ["display_address", "manager_id", "building_id","street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            

#######################################################################################################
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)
for i in range(1, folds+1):
    street_level={}
    for j in train_df['street_address'].values:
        street_level[j]=[0,0,0]
    test_index=train_df[train_df['CVindices'] == i].index.tolist() #index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=train_df[train_df['CVindices'] != i].index.tolist() # list(set(index).difference(test_index))
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



#######################################################################################################

features_to_use  = ["listing_id",
                    'manager_level_low','manager_level_medium','manager_level_high',
                    'manager_bed_low','manager_bed_medium','manager_bed_high',
                    'distance_level_low','distance_level_medium','distance_level_high',
                    'distanceBed_level_low','distanceBed_level_medium','distanceBed_level_high',
                    'building_level_low','building_level_medium','building_level_high',
                    'street_level_low','street_level_medium','street_level_high']
sub_valfile = 'C:/Users/SriPrav/Documents/R/21Rental/input/Prav_train_features11.csv'
train_df[features_to_use].to_csv(sub_valfile, index=False)

sub_valfile = 'C:/Users/SriPrav/Documents/R/21Rental/input/Prav_test_features11.csv'
test_df[features_to_use].to_csv(sub_valfile, index=False)

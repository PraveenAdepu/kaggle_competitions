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

train_features_00 = inDir + "/input/train_features_00.csv"
trainfeatures_00 = pd.read_csv(train_features_00)

test_features_00 = inDir + "/input/test_features_00.csv"
testfeatures_00 = pd.read_csv(test_features_00)

train_df = pd.merge(train_df, trainfeatures_00, how = 'left', on = 'listing_id')

test_df = pd.merge(test_df, testfeatures_00, how = 'left', on = 'listing_id')

image_features = inDir + "/input/Prav_ImageFeatures.csv"
image_features = pd.read_csv(image_features)

train_df = pd.merge(train_df, image_features, how = 'left', on = 'listing_id')
test_df = pd.merge(test_df, image_features, how = 'left', on = 'listing_id')

image_Exiffeatures = inDir + "/input/Prav_ImageExif_01_features.csv"
image_Exiffeatures = pd.read_csv(image_Exiffeatures)

train_df = pd.merge(train_df, image_Exiffeatures, how = 'left', on = 'listing_id')
test_df = pd.merge(test_df, image_Exiffeatures, how = 'left', on = 'listing_id')


train_Ref = inDir + "/input/Prav_trainingSet_Features_fromRef.csv"
test_Ref = inDir + "/input/Prav_testingSet_Features_fromRef.csv"
train_Ref = pd.read_csv(train_Ref)
test_Ref = pd.read_csv(test_Ref)
print(train_Ref.shape) # (49352, 286)
print(test_Ref.shape)  # (74659, 286)

Ref_features = ['listing_id',
                 'building_id_mean_med',
                 'building_id_mean_high',
                 'manager_id_mean_med',
                 'manager_id_mean_high',
                ]
 
train_Ref = train_Ref[Ref_features]
test_Ref  = test_Ref[Ref_features]

print(train_Ref.shape) # (49352, 5)
print(test_Ref.shape)  # (74659, 5)

train_df = pd.merge(train_df, train_Ref, how = 'left', on = 'listing_id')
test_df = pd.merge(test_df, test_Ref, how = 'left', on = 'listing_id')

train_df = pd.merge(train_df, CV_Schema, how = 'left', on = 'listing_id')

features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price" ,"address_similarity_jw","address_similarity_sound"
                    ,"MeanWidth"
                   ,"MeanHeight" 
                   ,"Meansizebytes"
#                   ,"meanpixelsize"
#                   , "maxwidth"
#                   , "maxheight"
#                   , "maxpixelsize"
#                   , "minwidth"
#                   , "minheight"
#                   , "minpixelsize"
#                   , "maxsizebytes"
#                   , "minsizebytes"
#                   , "meanextrema00"
#                   , "meanextrema01"
#                   , "meanextrema10"
#                   , "meanextrema11"
#                   , "meanextrema20"
#                   , "meanextrema21"
#                   , "meancount00"
#                   , "meancount01"
#                   , "meancount02"
#                   , "meansum00"
#                   , "meansum01"
#                   , "meansum02"
#                   , "meanmean00"
#                   , "meanmean01"
#                   , "meanmean02"
#                   , "meanmedian00"
#                   , "meanmedian01"
#                   , "meanmedian02"
#                   , "meanrms00"
#                   , "meanrms01"
#                   , "meanrms02"
#                   , "meanvar00"
#                   , "meanvar01"
#                   , "meanvar02"
#                   , "meanstddev00"
#                   , "meanstddev01"
#                   , "meanstddev02"
#                   , "maxsizebytes"
#                   , "minsizebytes"
                   , "meanColorSpace"
                   , "meanContrast" 
                   , "meanCustomRendered"
                   , "meanExifOffset"
                   , "meanExposureMode"
                   , "meanFlash"
                   , "meanLightSource"
                   , "meanSharpness"
                   , "meanSubjectDistanceRange"
                   , "meanWhiteBalance"
                   , "meanYCbCrPositioning"                  
                    ,'building_id_mean_med'
                    ,'building_id_mean_high'
                    ,'manager_id_mean_med'
                    ,'manager_id_mean_high'
                   
                   ] 

           

train_df["price_t"] = train_df["price"]/train_df["bedrooms"] 
test_df["price_t"] = test_df["price"]/test_df["bedrooms"]

# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))





# convert the created column to datetime object so as to extract more features 
train_df["created"] = pd.to_datetime(train_df["created"])
test_df["created"] = pd.to_datetime(test_df["created"])

# Let us extract some features like year, month, day, hour from date columns #
train_df["created_year"] = train_df["created"].dt.year
test_df["created_year"] = test_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
test_df["created_month"] = test_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
test_df["created_day"] = test_df["created"].dt.day
train_df["created_hour"] = train_df["created"].dt.hour
test_df["created_hour"] = test_df["created"].dt.hour
train_df["weekday"] = train_df["created"].dt.weekday
test_df["weekday"] = test_df["created"].dt.weekday
train_df["weekend"] = train_df["weekday"].apply(lambda x: 0 if x < 5 else 1)
test_df["weekend"] = test_df["weekday"].apply(lambda x: 0 if x < 5 else 1)

train_df["num_description_words_MeanRatio"] = train_df["num_description_words"]/train_df["num_description_words"].mean()
test_df["num_description_words_MeanRatio"] = test_df["num_description_words"]/test_df["num_description_words"].mean()

train_df["price_Tobedbath"] = train_df["price"]/( train_df["bedrooms"] + train_df["bathrooms"] )
test_df["price_Tobedbath"] = test_df["price"]/ ( test_df["bedrooms"] + test_df["bathrooms"] ) 

train_df["price_TobedbathDiff"] = train_df["price"]/( train_df["bedrooms"] - train_df["bathrooms"] )
test_df["price_TobedbathDiff"] = test_df["price"]/ ( test_df["bedrooms"] - test_df["bathrooms"] )

train_df["price_BedtoBath"] = train_df["bedrooms"]/( train_df["bedrooms"] + train_df["bathrooms"] )
test_df["price_BedtoBath"] = test_df["bedrooms"]/ ( test_df["bedrooms"] + test_df["bathrooms"] )
 

train_df["Bedroom_Tobathroom_Ratio"] = train_df["bedrooms"] / train_df["bathrooms"].apply(lambda x: 1 if x == 0 else x) 
test_df["Bedroom_Tobathroom_Ratio"] =  test_df["bedrooms"] / test_df["bathrooms"].apply(lambda x: 1 if x == 0 else x)  

train_df["num_features_MeanRatio"] = train_df["num_features"]/train_df["num_features"].mean()
test_df["num_features_MeanRatio"] = test_df["num_features"]/test_df["num_features"].mean()

train_df["passed"] = train_df["created"].max() - train_df["created"]
test_df["passed"] = test_df["created"].max() - test_df["created"]

ny_lat = 40.785091
ny_lon = -73.968285

train_df['distance_to_city'] = np.sqrt((train_df['longitude'] - ny_lon)**2  + (train_df['latitude'] - ny_lat)**2)
test_df['distance_to_city'] = np.sqrt((test_df['longitude'] - ny_lon)**2  + (test_df['latitude'] - ny_lat)**2)

train_df['distanceMeasure'] = train_df['distance_to_city']*100
test_df['distanceMeasure'] = test_df['distance_to_city']*100

train_df['distanceMeasure'] = train_df['distanceMeasure'].astype(int)
test_df['distanceMeasure'] = test_df['distanceMeasure'].astype(int)


string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')

remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
address_map = {
    'w': 'west',
    'st.': 'street',
    'ave': 'avenue',
    'st': 'street',
    'e': 'east',
    'n': 'north',
    's': 'south'
}

def address_map_func(s):
    s = s.split(' ')
    out = []
    for x in s:
        if x in address_map:
            out.append(address_map[x])
        else:
            out.append(x)
    return ' '.join(out)

feature_map = {
  'hardwood_floors' : 'hardwood', 
  'laundry_in_building' : 'laundry',
  'laundry_in_unit' : 'laundry',
  'laundry_room' : 'laundry', 
  'on-site_laundry' : 'laundry',
  'dryer_in_unit' : 'laundry',
  'washer_in_unit' : 'laundry', 
  'washer/dryer' : 'laundry', 
  'roof-deck' : 'roof_deck',
  'common_roof_deck' : 'roof_deck',
  'roofdeck' : 'roof_deck', 
  'outdoor_space' : 'outdoor', 
  'common_outdoor_space' : 'outdoor', 
  'private_outdoor_space' : 'outdoor', 
  'publicoutdoor' : 'outdoor',
  'outdoor_areas' : 'outdoor', 
  'private_outdoor' : 'outdoor', 
  'common_outdoor' : 'outdoor',
  'garden/patio' : 'garden', 
  'residents_garden' : 'garden', 
  'parking_space' : 'parking', 
  'common_parking/garage' : 'parking',
  'on-site_garage' : 'parking', 
  'fitness_center' : 'fitness', 
  'gym' : 'fitness', 
  'gym/fitness' : 'fitness', 
  'fitness/fitness' : 'fitness', 
  'cats_allowed' : 'pets', 
  'dogs_allowed' : 'pets', 
  'pets_on_approval' : 'pets', 
  'live-in_superintendent' : 'live-in super',   
  'full-time_doorman' : 'doorman', 
  'newly_renovated' : 'renovated', 
  'pre-war' : 'prewar'
  }

def feature_map_func(s):
    s = s.split(' ')
    out = []
    for x in s:
        if x in feature_map:
            out.append(feature_map[x])
        else:
            out.append(x)
    return ' '.join(out)
    
    
train_df['address1'] = train_df['display_address']
train_df['address1'] = train_df['address1'].apply(lambda x: x.lower())

train_df['address1'] = train_df['address1'].apply(lambda x: x.translate(remove_punct_map))
train_df['address1'] = train_df['address1'].apply(lambda x: address_map_func(x))

test_df['address1'] = test_df['display_address']
test_df['address1'] = test_df['address1'].apply(lambda x: x.lower())

test_df['address1'] = test_df['address1'].apply(lambda x: x.translate(remove_punct_map))
test_df['address1'] = test_df['address1'].apply(lambda x: address_map_func(x))

new_cols = ['street', 'avenue', 'east', 'west', 'north', 'south']

for col in new_cols:
    train_df[col] = train_df['address1'].apply(lambda x: 1 if col in x else 0)
    test_df[col] = test_df['address1'].apply(lambda x: 1 if col in x else 0)
    
#train_df['manager_freq'] = train_df.groupby('manager_id')['manager_id'].transform('count')
#test_df['manager_freq']  = test_df.groupby('manager_id')['manager_id'].transform('count')
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

features_to_use.append('manager_level_low') 
features_to_use.append('manager_level_medium') 
features_to_use.append('manager_level_high')
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

features_to_use.append('manager_bed_low') 
features_to_use.append('manager_bed_medium') 
features_to_use.append('manager_bed_high')
#######################################################################################################

#######################################################################################################

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

features_to_use.append('distance_level_low') 
features_to_use.append('distance_level_medium') 
features_to_use.append('distance_level_high')
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

features_to_use.append('distanceBed_level_low') 
features_to_use.append('distanceBed_level_medium') 
features_to_use.append('distanceBed_level_high')
#######################################################################################################



# adding all these new features to use list #
features_to_use.extend(["price_t","num_photos", "num_features", "num_description_words"
                        ,"created_year", "created_month", "created_day", "listing_id"
                        , "created_hour","weekday" , "weekend","num_description_words_MeanRatio"
                        ,"price_Tobedbath","price_TobedbathDiff","price_BedtoBath","Bedroom_Tobathroom_Ratio","num_features_MeanRatio"
                        ,'street', 'avenue', 'east', 'west', 'north', 'south','distance_to_city']) # ,"passed" ,'manager_freq'
categorical = ["display_address", "manager_id", "building_id","street_address"]
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

features_to_use.append('street_level_low') 
features_to_use.append('street_level_medium') 
features_to_use.append('street_level_high')

#######################################################################################################

train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

train_df['features'] = train_df['features'].apply(lambda x: x.lower())
train_df['features'] = train_df['features'].apply(lambda x: feature_map_func(x))
print(train_df["features"].head())

test_df['features'] = test_df['features'].apply(lambda x: x.lower())
test_df['features'] = test_df['features'].apply(lambda x: feature_map_func(x))
print(test_df["features"].head())

train_df["meanColorSpace"].head()
train_df = train_df.fillna(0)   
test_df = test_df.fillna(0)   

tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])

print(tr_sparse.shape)
print(te_sparse.shape)

#Stfidf = CountVectorizer(stop_words='english', max_features=200)
#tr_Ssparse = Stfidf.fit_transform(train_df[ "street_address"])
#te_Ssparse = Stfidf.transform(test_df[ "street_address"])
#
#print(tr_Ssparse.shape)
#print(te_Ssparse.shape)

#train_X = sparse.hstack([train_df[features_to_use], tr_sparse, tr_Ssparse]).tocsr()
#test_X = sparse.hstack([test_df[features_to_use], te_sparse,te_Ssparse]).tocsr()

train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = pd.DataFrame(train_df['interest_level'].apply(lambda x: target_num_map[x]))

print(train_X.shape, train_y.shape ,test_X.shape)

train_df['interest_level'].head()


#from sklearn.metrics import roc_auc_score
#print('Original AUC:', roc_auc_score(train_y/3, train_df['price_t'].fillna(0)))
#print('   TFIDF AUC:', roc_auc_score(train_df['is_duplicate'], train_df['zbigrams_common_ratio'].fillna(0)))

param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.02
param['max_depth'] = 4
param['silent'] = 1
param['num_class'] = 3
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.45
param['nthread'] = 25
param['gamma'] = 1
param['seed'] = 2017
param['print_every_n'] = 50
num_rounds = 3610
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
    
    sub_valfile = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.xgb18.fold' + str(i) + '.csv'
    pred_cv.to_csv(sub_valfile, index=False)
    
    xgtest = xgb.DMatrix(test_X)
    pred_test = model.predict(xgtest)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["high", "medium", "low"]
    pred_test["listing_id"] = test_df.listing_id.values
   
    sub_file = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.xgb18.fold' + str(i) + '-test' + '.csv'
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
   
    sub_file = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.xgb18.full' + '.csv'
    predfull_test.to_csv(sub_file, index=False)

folds = 5
i = 1
if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost(i)
    fulltrain_xgboost(folds)
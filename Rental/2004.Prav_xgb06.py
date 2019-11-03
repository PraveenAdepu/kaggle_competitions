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



features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price","address_similarity_jw","address_similarity_sound"] 

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
# adding all these new features to use list #
features_to_use.extend(["price_t","num_photos", "num_features", "num_description_words"
                        ,"created_year", "created_month", "created_day", "listing_id"
                        , "created_hour","weekday" , "weekend","num_description_words_MeanRatio"
                        ,"price_Tobedbath","price_TobedbathDiff","price_BedtoBath","Bedroom_Tobathroom_Ratio","num_features_MeanRatio"]) # ,"passed" 
categorical = ["display_address", "manager_id", "building_id"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)

train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
print(train_df["features"].head())
tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])

print(tr_sparse.shape)
print(te_sparse.shape)

Stfidf = CountVectorizer(stop_words='english', max_features=200)
tr_Ssparse = Stfidf.fit_transform(train_df[ "street_address"])
te_Ssparse = Stfidf.transform(test_df[ "street_address"])

print(tr_Ssparse.shape)
print(te_Ssparse.shape)

train_X = sparse.hstack([train_df[features_to_use], tr_sparse, tr_Ssparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse,te_Ssparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = pd.DataFrame(train_df['interest_level'].apply(lambda x: target_num_map[x]))

print(train_X.shape, train_y.shape ,test_X.shape)

train_df['interest_level'].head()
train_df = pd.merge(train_df, CV_Schema, how = 'left', on = 'listing_id')


param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.01
param['max_depth'] = 6
param['silent'] = 1
param['num_class'] = 3
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 25
param['seed'] = 2017
param['print_every_n'] = 50
num_rounds = 3500
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
    
    sub_valfile = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.xgb06.fold' + str(i) + '.csv'
    pred_cv.to_csv(sub_valfile, index=False)
    
    xgtest = xgb.DMatrix(test_X)
    pred_test = model.predict(xgtest)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["high", "medium", "low"]
    pred_test["listing_id"] = test_df.listing_id.values
   
    sub_file = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.xgb06.fold' + str(i) + '-test' + '.csv'
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
   
    sub_file = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.xgb06.full' + '.csv'
    predfull_test.to_csv(sub_file, index=False)

folds = 5
if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost(i)
    fulltrain_xgboost(folds)
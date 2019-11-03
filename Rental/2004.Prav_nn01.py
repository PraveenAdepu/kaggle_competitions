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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical
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



features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price","address_similarity_jw","address_similarity_sound"] 

train_df["price_t"] = train_df["price"]/train_df["bedrooms"].apply(lambda x: 1 if x == 0 else x) 
test_df["price_t"] = test_df["price"]/test_df["bedrooms"].apply(lambda x: 1 if x == 0 else x)

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

train_df["bed_plus_bath_rooms"] = train_df["bedrooms"] + train_df["bathrooms"]
test_df["bed_plus_bath_rooms"]  = test_df["bedrooms"] + test_df["bathrooms"]

train_df["bed_minus_bath_rooms"] = train_df["bedrooms"] - train_df["bathrooms"]
test_df["bed_minus_bath_rooms"]  = test_df["bedrooms"] - test_df["bathrooms"]


train_df["price_Tobedbath"] = train_df["price"]/train_df["bed_plus_bath_rooms"].apply(lambda x: 1 if x == 0 else x)
test_df["price_Tobedbath"] = test_df["price"]/test_df["bed_plus_bath_rooms"].apply(lambda x: 1 if x == 0 else x) 

train_df["price_TobedbathDiff"] = train_df["price"]/train_df["bed_minus_bath_rooms"].apply(lambda x: 1 if x == 0 else x)
test_df["price_TobedbathDiff"] = test_df["price"]/test_df["bed_minus_bath_rooms"].apply(lambda x: 1 if x == 0 else x)

train_df["price_BedtoBath"] = train_df["bedrooms"]/train_df["bed_plus_bath_rooms"].apply(lambda x: 1 if x == 0 else x)
test_df["price_BedtoBath"] = test_df["bedrooms"]//test_df["bed_plus_bath_rooms"].apply(lambda x: 1 if x == 0 else x)
 

train_df["Bedroom_Tobathroom_Ratio"] = train_df["bedrooms"] / train_df["bathrooms"].apply(lambda x: 1 if x == 0 else x) 
test_df["Bedroom_Tobathroom_Ratio"] =  test_df["bedrooms"] / test_df["bathrooms"].apply(lambda x: 1 if x == 0 else x)  

train_df["num_features_MeanRatio"] = train_df["num_features"]/train_df["num_features"].mean()
test_df["num_features_MeanRatio"] = test_df["num_features"]/test_df["num_features"].mean()

train_df["passed"] = train_df["created"].max() - train_df["created"]
test_df["passed"] = test_df["created"].max() - test_df["created"]


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
    

# adding all these new features to use list #
features_to_use.extend(["price_t","num_photos", "num_features", "num_description_words"
                        ,"created_year", "created_month", "created_day", "listing_id"
                        , "created_hour","weekday" , "weekend","num_description_words_MeanRatio"
                        ,"price_Tobedbath","price_TobedbathDiff","price_BedtoBath","Bedroom_Tobathroom_Ratio","num_features_MeanRatio"
                        ,'street', 'avenue', 'east', 'west', 'north', 'south']) # ,"passed" 
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

train_df['features'] = train_df['features'].apply(lambda x: x.lower())
train_df['features'] = train_df['features'].apply(lambda x: feature_map_func(x))
print(train_df["features"].head())

test_df['features'] = test_df['features'].apply(lambda x: x.lower())
test_df['features'] = test_df['features'].apply(lambda x: feature_map_func(x))
print(test_df["features"].head())


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

train_X = train_X.toarray()
test_X = test_X.toarray()
print(train_X.shape)
print(test_X.shape)

# Scale train_X and test_X together
traintest = np.vstack((train_X, test_X))
print(traintest.shape)
traintest = preprocessing.StandardScaler().fit_transform(traintest)

train_X = traintest[range(train_X.shape[0])]
test_X = traintest[range(train_X.shape[0], traintest.shape[0])]
print(train_X.shape)
print(test_X.shape)                 

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

train_y = to_categorical(train_y)

print(train_X.shape, train_y.shape ,test_X.shape)


train_df['interest_level'].head()
train_df = pd.merge(train_df, CV_Schema, how = 'left', on = 'listing_id')

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

#def nn_model(trainSet):
#    model = Sequential()
#    model.add(Dense(200, input_dim = trainSet.shape[1], init = 'normal', activation='tanh'))
#    #model.add(PReLU())
#    #model.add(Dropout(0.4))
#    model.add(Dense(100, init = 'normal', activation='tanh'))
#    #model.add(PReLU())
#    #model.add(Dropout(0.2))
#    model.add(Dense(3, init = 'normal', activation='sigmoid'))#'softmax'
#    model.compile(loss='categorical_crossentropy', optimizer="adadelta",metrics=["accuracy"])
#    return(model)
#callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=0)] 
             
def nn_model(trainSet):
    model = Sequential()
    
    model.add(Dense(500, input_dim = trainSet.shape[1], init = 'he_normal', activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(PReLU())
    
    model.add(Dense(50, init = 'he_normal', activation='sigmoid'))
    model.add(BatchNormalization())    
    model.add(Dropout(0.35))
    model.add(PReLU())
	
    model.add(Dense(3, init = 'he_normal', activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')#, metrics=['accuracy'])
    return(model)
nepochs = 1000
i = 1
#def train_nn(i):
#    trainindex = train_df[train_df['CVindices'] != i].index.tolist()
#    valindex   = train_df[train_df['CVindices'] == i].index.tolist()
#    
#    X_val_df = train_df.iloc[valindex,:]
#    
#    X_build , X_valid = train_X[trainindex,:], train_X[valindex,:]
#    y_build , y_valid = train_y[trainindex,:], train_y[valindex,:]
#    
#    y_build = np.array(y_build)
#    y_valid = np.array(y_valid)
#    model = nn_model(X_build)
#    fitmodel = model.fit_generator(generator = batch_generator(X_build, y_build, 128, True),
#                                  nb_epoch = nepochs,
#                                  samples_per_epoch = X_build.shape[0],
#                                  validation_data=(X_valid,y_valid),
#                                  verbose = 2)
#    pred_cv = fitmodel.predict_generator(generator = batch_generatorp(X_valid, 800, False), val_samples = X_valid.shape[0])
#    pred_cv = pd.DataFrame(pred_cv)
#    pred_cv.head()
#    pred_cv.columns = ["high", "medium", "low"]
#    pred_cv["listing_id"] = X_val_df.listing_id.values
#    
#    sub_valfile = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.nn01.fold' + str(i) + '.csv'
#    pred_cv.to_csv(sub_valfile, index=False)
#    pred_test = fitmodel.predict_generator(generator = batch_generatorp(test_X, 800, False), val_samples = test_X.shape[0])
#    pred_test = pd.DataFrame(pred_test)
#    pred_test.columns = ["high", "medium", "low"]
#    pred_test["listing_id"] = test_df.listing_id.values
#   
#    sub_file = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.nn01.fold' + str(i) + '-test' + '.csv'
#    pred_test.to_csv(sub_file, index=False)   
#    
#    del pred_cv
#    del pred_test
filepath="weights.best.hdf5"    
def train_nn(i):
    trainindex = train_df[train_df['CVindices'] != i].index.tolist()
    valindex   = train_df[train_df['CVindices'] == i].index.tolist()
    
    X_val_df = train_df.iloc[valindex,:]
    
    X_build , X_valid = train_X[trainindex,:], train_X[valindex,:]
    y_build , y_valid = train_y[trainindex,:], train_y[valindex,:]
    
    y_build = np.array(y_build)
    y_valid = np.array(y_valid)
    model = nn_model(X_build)
    callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]
#    early_stop = EarlyStopping(monitor='val_loss', patience=75, verbose=0)
#    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)
    model.fit(X_build, y_build,
                                  nb_epoch = nepochs,
                                  batch_size=1000,
                                  validation_data=(X_valid,y_valid),
                                  verbose = 1,
                                  callbacks=callbacks )
    pred_cv = model.predict_proba(X_valid, verbose=1)
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.head()
    pred_cv.columns = ["high", "medium", "low"]
    pred_cv["listing_id"] = X_val_df.listing_id.values
    
    sub_valfile = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.nn01.fold' + str(i) + '.csv'
    pred_cv = pred_cv[["listing_id","high", "medium", "low"]]
    pred_cv.to_csv(sub_valfile, index=False)
    pred_test = model.predict_proba(test_X,verbose=1)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["high", "medium", "low"]
    pred_test["listing_id"] = test_df.listing_id.values
    pred_test = pred_test[["listing_id","high", "medium", "low"]]
    sub_file = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.nn01.fold' + str(i) + '-test' + '.csv'
    pred_test.to_csv(sub_file, index=False)   
   
    del pred_cv
    del pred_test

full_epochs = int(nepochs * 1.2)
#
def full_train_nn(i):
    trainindex = train_df[train_df['CVindices'] != i].index.tolist()
    valindex   = train_df[train_df['CVindices'] == i].index.tolist()
    
    X_val_df = train_df.iloc[valindex,:]
    
    X_build , X_valid = train_X[trainindex,:], train_X[valindex,:]
    y_build , y_valid = train_y[trainindex,:], train_y[valindex,:]
    
    y_build = np.array(y_build)
    y_valid = np.array(y_valid)
    model = nn_model(train_X)
    callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]
#    early_stop = EarlyStopping(monitor='val_loss', patience=75, verbose=0)
#    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)
    model.fit(train_X, train_y,
                                  nb_epoch = full_epochs,
                                  batch_size=1000,
                                  validation_data=(train_X, train_y),
                                  verbose = 1,
                                  callbacks=callbacks )
   
    pred_test = model.predict_proba(test_X,verbose=1)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["high", "medium", "low"]
    pred_test["listing_id"] = test_df.listing_id.values
    pred_test = pred_test[["listing_id","high", "medium", "low"]]
    sub_file = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.nn01.full' + '.csv'
    pred_test.to_csv(sub_file, index=False)   
   
    del pred_cv
    del pred_test

folds = 5
if __name__ == '__main__':
    for i in range(1, folds+1):
        train_nn(i)
#    fulltrain_xgboost(folds)
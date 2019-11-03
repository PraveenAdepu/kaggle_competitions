import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.sparse import csr_matrix
import sys
import string
from sklearn import model_selection, preprocessing, ensemble
#reload(sys)
#sys.setdefaultencoding('utf8')

inDir = 'C:/Users/SriPrav/Documents/R/21Rental'

#create the data based on the raw json files
def load_data_sparse():
    train_file = inDir + "/input/train.json"
    test_file = inDir + "/input/test.json"
    train_df = pd.read_json(train_file)
    test_df = pd.read_json(test_file)
    print(train_df.shape) # (49352, 15)
    print(test_df.shape)  # (74659, 14)
    
    
    
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
                       ] 
    
               
    
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
    
    train_df["price_Tobedbath"] = train_df["price"]/( train_df["bedrooms"] + train_df["bathrooms"] ).apply(lambda x: 1 if x == 0 else x) 
    test_df["price_Tobedbath"] = test_df["price"]/ ( test_df["bedrooms"] + test_df["bathrooms"] ).apply(lambda x: 1 if x == 0 else x)  
    
    train_df["price_TobedbathDiff"] = train_df["price"]/( train_df["bedrooms"] - train_df["bathrooms"] ).apply(lambda x: 1 if x == 0 else x) 
    test_df["price_TobedbathDiff"] = test_df["price"]/ ( test_df["bedrooms"] - test_df["bathrooms"] ).apply(lambda x: 1 if x == 0 else x) 
    
    train_df["price_BedtoBath"] = train_df["bedrooms"]/( train_df["bedrooms"] + train_df["bathrooms"] ).apply(lambda x: 1 if x == 0 else x) 
    test_df["price_BedtoBath"] = test_df["bedrooms"]/ ( test_df["bedrooms"] + test_df["bathrooms"] ).apply(lambda x: 1 if x == 0 else x) 
     
    
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
    
    train_df["listing_id"] = train_df["listing_id"] - 68119576.0
    test_df["listing_id"] =  test_df["listing_id"] - 68119576.0
    
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
    
    train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
    test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

   
    target_num_map = {'high':0, 'medium':1, 'low':2}
    train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
    ids= test_df.listing_id.values
    print(train_X.shape, test_X.shape) 
    
    return train_X,test_X,train_y,ids


       
def main():
    

        #training and test files, created using SRK's python script
        train_file="train_stacknet03.csv"
        test_file="test_stacknet03.csv"
        
        ######### Load files ############

        X,X_test,y,ids=load_data_sparse ()# you might need to change that to whatever folder the json files are in
        ids= np.array([int(k)+68119576 for k in ids ]) # we add the id value we removed before for scaling reasons.
        print(X.shape, X_test.shape) 
        
        #create to numpy arrays (dense format)        
        X=X.toarray()
        X_test=X_test.toarray()  
        
        print ("scalling") 
        #scale the data
        stda=StandardScaler()  
        X_test=stda.fit_transform (X_test)          
        X=stda.transform(X)

        
              
        #Create Arrays for meta
        train_stacker=[ [0.0 for s in range(3)]  for k in range (0,(X.shape[0])) ]
        test_stacker=[[0.0 for s in range(3)]   for k in range (0,(X_test.shape[0]))]
        
        number_of_folds=5 # number of folds to use
        print("kfolder")
        #cerate 5 fold object
        mean_logloss = 0.0
        kfolder=StratifiedKFold(y, n_folds=number_of_folds,shuffle=True, random_state=15)   

        #xgboost_params
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

        
        i=0 # iterator counter
        print ("starting cross validation with %d kfolds " % (number_of_folds))
        for train_index, test_index in kfolder:
                # creaning and validation sets
                X_train, X_cv = X[train_index], X[test_index]
                y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
                
                print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))
                #training
                X1=xgb.DMatrix(X_train, label=np.array(y_train),missing =0)
                X1cv=xgb.DMatrix(X_cv, missing =-999.0)
                bst = xgb.train(param.items(), X1, 3500) 
                #predictions
                predictions = bst.predict(X1cv)     
                preds=predictions.reshape( X_cv.shape[0], 3)

                #scalepreds(preds)     
                logs = log_loss(y_cv,preds)
                print ("size train: %d size cv: %d loglikelihood (fold %d/%d): %f" % ((X_train.shape[0]), (X_cv.shape[0]), i + 1, number_of_folds, logs))
             
                mean_logloss += logs
                #save the results
                no=0
                for real_index in test_index:
                    for d in range (0,3):
                        train_stacker[real_index][d]=(preds[no][d])
                    no+=1
                i+=1
        mean_logloss/=number_of_folds
        print (" Average Lolikelihood: %f" % (mean_logloss) )
                
                      
        #X_test=np.column_stack((X_test,woe_cv))      
        print (" making test predictions ")
        
        X1=xgb.DMatrix(X, label=np.array(y) , missing =0)
        X1cv=xgb.DMatrix(X_test, missing =-999.0)
        bst = xgb.train(param.items(), X1, 3500) 
        predictions = bst.predict(X1cv)     
        preds=predictions.reshape( X_test.shape[0], 3)        
       
        for pr in range (0,len(preds)):  
                for d in range (0,3):            
                    test_stacker[pr][d]=(preds[pr][d]) 
        
        
        
        print ("merging columns")   
        #stack xgboost predictions
        X=np.column_stack((X,train_stacker))
        # stack id to test
        X_test=np.column_stack((X_test,test_stacker))        
        
        # stack target to train
        X=np.column_stack((y,X))
        # stack id to test
        X_test=np.column_stack((ids,X_test))
        
        #export to txt files (, del.)
        print ("exporting files")
        np.savetxt(train_file, X, delimiter=",", fmt='%.5f')
        np.savetxt(test_file, X_test, delimiter=",", fmt='%.5f')        

        print("Write results...")
        output_file = "submission03_"+str( (mean_logloss ))+".csv"
        print("Writing submission to %s" % output_file)
        f = open(output_file, "w")   
        f.write("listing_id,high,medium,low\n")# the header   
        for g in range(0, len(test_stacker))  :
          f.write("%s" % (ids[g]))
          for prediction in test_stacker[g]:
             f.write(",%f" % (prediction))    
          f.write("\n")
        f.close()
        print("Done.")
        

if __name__=="__main__":
  main()
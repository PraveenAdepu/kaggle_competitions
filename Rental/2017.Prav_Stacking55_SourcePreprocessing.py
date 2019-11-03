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
folds = 5
#create the data based on the raw json files
def load_data_sparse():
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
    
    Prav_train_features11 = inDir + "/input/Prav_train_features11.csv"
    Prav_train_features11 = pd.read_csv(Prav_train_features11)
    
    Prav_test_features11 = inDir + "/input/Prav_test_features11.csv"
    Prav_test_features11 = pd.read_csv(Prav_test_features11)
    
    train_df = pd.merge(train_df, Prav_train_features11, how = 'left', on = 'listing_id')
    test_df = pd.merge(test_df, Prav_test_features11, how = 'left', on = 'listing_id')
    
    Prav_imagetime = inDir + "/input/listing_image_time.csv"
    Prav_imagetime = pd.read_csv(Prav_imagetime) #listing_image_time
    
    train_df = pd.merge(train_df, Prav_imagetime, how = 'left', on = 'listing_id')
    test_df = pd.merge(test_df, Prav_imagetime, how = 'left', on = 'listing_id')
    
    Prav_features20 = inDir + "/input/Prav_features20.csv"
    Prav_features20 = pd.read_csv(Prav_features20) 
    
    train_df = pd.merge(train_df, Prav_features20, how = 'left', on = 'listing_id')
    test_df = pd.merge(test_df, Prav_features20, how = 'left', on = 'listing_id')
    
    train_df = pd.merge(train_df, CV_Schema, how = 'left', on = 'listing_id')
    
    #basic features
    train_df["price_t"] =train_df["price"]/train_df["bedrooms"].apply(lambda x: 1 if x == 0 else x)
    test_df["price_t"] = test_df["price"]/test_df["bedrooms"].apply(lambda x: 1 if x == 0 else x) 
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
    
    
    features_to_use=["bathrooms", "bedrooms", "latitude", "longitude", "price","price_t","num_photos", "num_features", "num_description_words","listing_id"
                     ,'manager_level_low','manager_level_medium','manager_level_high', 'time_stamp', 'price_comparison']
    
             
    train_df["listing_id"] = train_df["listing_id"] - 68119576.0
    test_df["listing_id"] =  test_df["listing_id"] - 68119576.0
    
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
    
    tfidf = CountVectorizer(stop_words='english', max_features=200)
    tr_sparse = tfidf.fit_transform(train_df["features"])
    te_sparse = tfidf.transform(test_df["features"])



    train_df.replace(np.inf, np.nan)
    test_df.replace(np.inf, np.nan)
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
        train_file="train_stacknet55.csv"
        test_file="test_stacknet55.csv"
        
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
                bst = xgb.train(param.items(), X1, 2910) 
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
        bst = xgb.train(param.items(), X1, 2910) 
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
        output_file = "submission55_"+str( (mean_logloss ))+".csv"
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
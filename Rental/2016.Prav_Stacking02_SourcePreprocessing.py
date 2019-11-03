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
#reload(sys)
#sys.setdefaultencoding('utf8')

inDir = 'C:/Users/SriPrav/Documents/R/21Rental'

#create the data based on the raw json files
def load_data_sparse():
    train_file = inDir + "/input/Prav_trainingSet_Features_fromRef.csv"
    test_file = inDir + "/input/Prav_testingSet_Features_fromRef.csv"
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    print(train_df.shape) # (49352, 15)
    print(test_df.shape)  # (74659, 14)
    
    
    train_df["listing_id"] = train_df["listing_id"] - 68119576.0
    test_df["listing_id"] =  test_df["listing_id"] - 68119576.0
    
    features_to_use = [col for col in train_df.columns if col not in ['interest_level']] 

    train_X = csr_matrix(train_df[features_to_use])
    test_X  = csr_matrix(test_df[features_to_use])
    
    target_num_map = {'high':0, 'medium':1, 'low':2}
    train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
    ids= test_df.listing_id.values
    print(train_X.shape, test_X.shape)    
    return train_X,test_X,train_y,ids


       
def main():
    

        #training and test files, created using SRK's python script
        train_file="train_stacknet02.csv"
        test_file="test_stacknet02.csv"
        
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
        param['booster']='gbtree'
        param['objective'] = 'multi:softprob'
        param['bst:eta'] = 0.01
        param['seed']=  1
        param['bst:max_depth'] = 6
        param['bst:min_child_weight']= 1.
        param['silent'] =  1  
        param['nthread'] = 24 # put more if you have
        param['bst:subsample'] = 0.7
        param['gamma'] = 1.0
        param['colsample_bytree']= 1.0
        param['num_parallel_tree']= 3   
        param['colsample_bylevel']= 0.7                  
        param['lambda']=5  
        param['num_class']= 3 

        
        i=0 # iterator counter
        print ("starting cross validation with %d kfolds " % (number_of_folds))
        for train_index, test_index in kfolder:
                # creaning and validation sets
                X_train, X_cv = X[train_index], X[test_index]
                y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
                
                print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))
                #training
                X1=xgb.DMatrix(X_train, label=np.array(y_train),missing =-999.0)
                X1cv=xgb.DMatrix(X_cv, missing =-999.0)
                bst = xgb.train(param.items(), X1, 3300) 
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
        
        X1=xgb.DMatrix(X, label=np.array(y) , missing =-999.0)
        X1cv=xgb.DMatrix(X_test, missing =-999.0)
        bst = xgb.train(param.items(), X1, 3300) 
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
        output_file = "submission02_"+str( (mean_logloss ))+".csv"
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
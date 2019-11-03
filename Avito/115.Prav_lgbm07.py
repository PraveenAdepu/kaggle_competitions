# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 09:46:18 2018

@author: SriPrav
"""


import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc


# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import sparse

# Gradient Boosting
import lightgbm as lgb
import lightgbm as lightgbm

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

print("\nData Load Stage")
inDir = 'C:/Users/SriPrav/Documents/R/48Avito'


training = pd.read_csv(inDir+'/input/train.csv', parse_dates = ["activation_date"])
Prav_5folds_CVIndices = pd.read_csv(inDir+'./input/Prav_5folds_CVindices_weekdayStratified.csv')

training = pd.merge(training, Prav_5folds_CVIndices, how = 'inner', on = 'item_id')

training = training.reset_index(drop=True)

traindex = training.index

testing = pd.read_csv(inDir+'/input/test.csv', parse_dates = ["activation_date"])
testdex = testing.index

training.columns
testing.columns

testing['deal_probability'] = 0
testing['CVindices'] = 0


Prav_train_FE01 = pd.read_csv(inDir+'/input/Prav_train_FE_01.csv')
Prav_test_FE01 = pd.read_csv(inDir+'/input/Prav_test_FE_01.csv')

training = pd.merge(training, Prav_train_FE01, how = 'left', on = 'item_id')
testing  = pd.merge(testing, Prav_test_FE01, how = 'left', on = 'item_id')

y = training.deal_probability.copy()



Prav_train_FE03 = pd.read_csv(inDir+'/input/Prav_train_FE_03.csv')
Prav_test_FE03 = pd.read_csv(inDir+'/input/Prav_test_FE_03.csv')

training = pd.merge(training, Prav_train_FE03, how = 'left', on = 'item_id')
testing  = pd.merge(testing, Prav_test_FE03, how = 'left', on = 'item_id')

traintest_FE02 = pd.read_csv(inDir+"/input/traintest_FE_02.csv")

training = pd.merge(training, traintest_FE02, how = 'left', on = 'user_id')
testing  = pd.merge(testing, traintest_FE02, how = 'left', on = 'user_id')

#training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

###########################################################################################################


print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)
#del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print("Feature Engineering")
df["price"] = np.log(df["price"]+0.001)
df["price"].fillna(-99,inplace=True)
df["image_top_1"].fillna(-99,inplace=True)


# Create Validation Index and Remove Dead Variables

df.drop(["activation_date","image"],axis=1,inplace=True)

print("\nEncode Variables")
categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1"]
print("Encoding :",categorical)

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))
    
print("\nText Features")

# Feature Engineering 
df['text_feat'] = df.apply(lambda row: ' '.join([
    str(row['param_1']), 
    str(row['param_2']), 
    str(row['param_3'])]),axis=1) # Group Param Features
df.drop(["param_1","param_2","param_3"],axis=1,inplace=True)

# Meta Text Features
textfeats = ["description","text_feat", "title"]
for cols in textfeats:
    df[cols] = df[cols].astype(str) 
    df[cols] = df[cols].astype(str).fillna('nicapotato') # FILL NA
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
#    df[cols + '_num_chars'] = df[cols].apply(len) # Count number of Characters
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words

print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
#russian_stop = set(stopwords.words('russian'))

tfidf_para = {
    #"stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
}
def get_col(col_name): return lambda x: x[col_name]
vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=17000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('text_feat',CountVectorizer(
            ngram_range=(1, 2),
            max_features=700,
            preprocessor=get_col('text_feat'))),
        ('title',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            max_features=700,
            preprocessor=get_col('title')))
    ])
    
start_vect=time.time()
vectorizer.fit(df.loc[traindex,:].to_dict('records'))
ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))


# Drop Text Cols
df.drop(textfeats, axis=1,inplace=True)

# Dense Features Correlation Matrix
#f, ax = plt.subplots(figsize=[10,7])
#sns.heatmap(pd.concat([df.loc[traindex,[x for x in df.columns if x not in categorical]], y], axis=1).corr(),
#            annot=False, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'},cmap="plasma",ax=ax, linewidths=.5)
#ax.set_title("Dense Features Correlation Matrix")
#plt.savefig('correlation_matrix.png')


feature_names = [c for c in df if c not in ['item_id', 'deal_probability','CVindices']]

df_requiredset = df[feature_names]

df_requiredset.fillna(-99,inplace=True)

print("Modeling Stage")
# Combine Dense Features with Sparse Text Bag of Words Features
#trainingSet = sparse.hstack([df_requiredset.iloc[traindex,:],ready_df[0:traindex.shape[0]]]).tocsr()
#testingSet = sparse.hstack([df_requiredset.iloc[testdex,:],ready_df[traindex.shape[0]:]]).tocsr()

trainingSet = ready_df[0:traindex.shape[0]]
testingSet = ready_df[traindex.shape[0]:]

tfvocab = df_requiredset.columns.tolist() + tfvocab

for shape in [trainingSet,testingSet]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
#del df
gc.collect();

from sklearn.linear_model import Ridge

def train_ridge_regression(trainingSet, testingSet,feature_names,i,nbags,ModelName, current_seed,verboseeval):
    
    trainindex = training[training['CVindices'] != i].index.tolist()
    valindex   = training[training['CVindices'] == i].index.tolist()
    
    X_build_set  = training[training['CVindices'] != i]
    X_valid_set  = training[training['CVindices'] == i]
      
#    X_build , X_valid = (trainingSet[trainindex,:]).tocsr(), (trainingSet[valindex,:]).tocsr()
    X_build , X_valid = trainingSet[trainindex,:], trainingSet[valindex,:]
    y_build , y_valid = X_build_set.deal_probability.values, X_valid_set.deal_probability.values    
                 
    pred_cv = np.zeros(X_valid.shape[0])
    pred_test = np.zeros(testing.shape[0])
    
    j = 1
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros(X_valid.shape[0])

        model = Ridge(solver="sag", fit_intercept=True, random_state=current_seed + j, alpha=3.3)
        
        model.fit(X_build, y_build)
        
        bag_cv= model.predict(X_valid #,num_iteration=model.best_iteration
                             )#.reshape(-1,1)
        pred_test += model.predict(testingSet)
        pred_cv += bag_cv
        bag_score = np.sqrt(metrics.mean_squared_error(X_valid_set['deal_probability'], bag_cv))
        print('bag ', j, '- rmse :', bag_score)
    pred_cv /= nbags
    pred_test/= nbags
    fold_score = np.sqrt(metrics.mean_squared_error(X_valid_set['deal_probability'], pred_cv))
    print('Fold ', i, '- rmse :', fold_score)
    
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = ["deal_probability"]
    pred_cv["item_id"] = X_valid_set.item_id.values
    
    sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
    pred_cv['deal_probability'] = pred_cv['deal_probability'].clip(0.0, 1.0)
    pred_cv[["item_id","deal_probability"]].to_csv(sub_valfile, index=False)
    
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["deal_probability"]
    pred_test["item_id"] = testing.item_id.values
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '-test' + '.csv'
    pred_test['deal_probability'] = pred_test['deal_probability'].clip(0.0, 1.0)
    pred_test[["item_id","deal_probability"]].to_csv(sub_file, index=False)
    del pred_cv
    del pred_test
    

def fulltrain_ridge_regression(trainingSet, testingSet,feature_names,nbags,ModelName, current_seed,verboseeval):
    
    predfull_test = np.zeros(testingSet.shape[0]) 
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')

        fullmodel = Ridge(solver="sag", fit_intercept=True, random_state=current_seed + j, alpha=3.3)        
        fullmodel.fit(trainingSet, training['deal_probability'])
        predfull_test += fullmodel.predict(testingSet)
    predfull_test/= nbags
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["deal_probability"]
    predfull_test["item_id"] = testing.item_id.values
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
    predfull_test['deal_probability'] = predfull_test['deal_probability'].clip(0.0, 1.0)
    predfull_test[["item_id","deal_probability"]].to_csv(sub_file, index=False)


lgbmModelName = 'ridge01'
model_run = 'full_test'
current_seed = 201801
verboseeval = 1

folds = 5
nbags = 2
  

if __name__ == '__main__':
    if model_run == 'fast_test':
        i = 1
        train_ridge_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName, current_seed,verboseeval)
        fulltrain_ridge_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName, current_seed,verboseeval)
    else:
        for i in range(1, folds+1):
            train_ridge_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName, current_seed,verboseeval)
        fulltrain_ridge_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName, current_seed,verboseeval)




#for shape in [X_build,X_valid]:
#    print("{} Rows and {} Cols".format(*shape.shape))
#print("Feature Names Length: ",len(tfvocab))
#
#X_build.shape
#X_valid.shape
#y_build.shape
#y_valid.shape


def lgb_mclip(preds, y):
    y = np.array(list(y.get_label()))
    score = np.sqrt(metrics.mean_squared_error(y.clip(0.,1.), preds.clip(0.,1.)))
    return 'R_CLIP', score, False

def train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,ModelName,lgbmParameters, lgbm_num_rounds,current_seed,verboseeval):
    
    trainindex = training[training['CVindices'] != i].index.tolist()
    valindex   = training[training['CVindices'] == i].index.tolist()
    
    X_build_set  = training[training['CVindices'] != i]
    X_valid_set  = training[training['CVindices'] == i]
      
#    X_build , X_valid = (trainingSet[trainindex,:]).tocsr(), (trainingSet[valindex,:]).tocsr()
    X_build , X_valid = trainingSet[trainindex,:], trainingSet[valindex,:]
    y_build , y_valid = X_build_set.deal_probability.values, X_valid_set.deal_probability.values
    
#    X_build  = trainingSet[trainingSet['CVindices'] != i]
#    X_valid  = trainingSet[trainingSet['CVindices'] == i]
#    X_build, X_valid, y_build, y_valid = train_test_split(
#    trainingSet, y, test_size=0.10, random_state=23)
    
    lgbmbuild = lightgbm.Dataset(X_build, y_build
                )
    lgbmval   = lightgbm.Dataset(X_valid, y_valid)
                 
    pred_cv = np.zeros(X_valid.shape[0])
    pred_test = np.zeros(testing.shape[0])
    
    j = 1
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros(X_valid.shape[0])
        lgbmParameters['bagging_seed'] =  current_seed + j    
        lgbmParameters['seed'] =  current_seed + j 
        model = lightgbm.train(lgbmParameters, lgbmbuild,lgbm_num_rounds#,feval=lgb_mclip#,early_stopping_rounds=100
                               ,valid_sets=[lgbmbuild,lgbmval],valid_names=['train','valid'],verbose_eval = verboseeval
                                  #,early_stopping_rounds=early_stopping_rounds
                                  )
        bag_cv= model.predict(X_valid #,num_iteration=model.best_iteration
                             )#.reshape(-1,1)
        pred_test += model.predict(testingSet)
        pred_cv += bag_cv
        bag_score = np.sqrt(metrics.mean_squared_error(X_valid_set['deal_probability'], bag_cv))
        print('bag ', j, '- rmse :', bag_score)
    pred_cv /= nbags
    pred_test/= nbags
    fold_score = np.sqrt(metrics.mean_squared_error(X_valid_set['deal_probability'], pred_cv))
    print('Fold ', i, '- rmse :', fold_score)
    
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = ["deal_probability"]
    pred_cv["item_id"] = X_valid_set.item_id.values
    
    sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
    pred_cv['deal_probability'] = pred_cv['deal_probability'].clip(0.0, 1.0)
    pred_cv[["item_id","deal_probability"]].to_csv(sub_valfile, index=False)
    
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["deal_probability"]
    pred_test["item_id"] = testing.item_id.values
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '-test' + '.csv'
    pred_test['deal_probability'] = pred_test['deal_probability'].clip(0.0, 1.0)
    pred_test[["item_id","deal_probability"]].to_csv(sub_file, index=False)
    del pred_cv
    del pred_test

def fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,ModelName,lgbmParameters, lgbm_num_rounds,current_seed,verboseeval):
    
    full_lgbm_num_rounds = int(lgbm_num_rounds * 1.2)
    lgbmtrain = lightgbm.Dataset(trainingSet, training['deal_probability'])

    predfull_test = np.zeros(testingSet.shape[0]) 
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        lgbmParameters['bagging_seed'] =  current_seed + j  
        lgbmParameters['seed'] =  current_seed + j          
        fullmodel = lightgbm.train(lgbmParameters, lgbmtrain,full_lgbm_num_rounds,feval=lgb_mclip
                                   ,valid_sets=[lgbmtrain,lgbmtrain],valid_names=['train','train'],verbose_eval = verboseeval
                                  #,early_stopping_rounds=early_stopping_rounds
                                  )
    
        predfull_test += fullmodel.predict(testingSet)
    predfull_test/= nbags
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["deal_probability"]
    predfull_test["item_id"] = testing.item_id.values
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
    predfull_test['deal_probability'] = predfull_test['deal_probability'].clip(0.0, 1.0)
    predfull_test[["item_id","deal_probability"]].to_csv(sub_file, index=False)


print("Light Gradient Boosting Regressor")

lgbmParameters = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "learning_rate" : 0.05,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        'nthread': 30,
        "seed" : 2018,
        "verbosity" : -1
    }
lgbm_num_rounds = 1500
lgbm_early_stopping_rounds = 100


seed = 2018
nn_epoch = 12
full_nn_epoch = 12
verboseeval=500

lgbmModelName = 'lgbm07'

current_seed = 201801


folds = 5
nbags = 2
  
model_run = 'full_test'

if __name__ == '__main__':
    if model_run == 'fast_test':
        i = 1
        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbmParameters, lgbm_num_rounds,current_seed,verboseeval)
        fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbmParameters, lgbm_num_rounds,current_seed,verboseeval)
    else:
        for i in range(1, folds+1):
            train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbmParameters, lgbm_num_rounds,current_seed,verboseeval)
        fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbmParameters, lgbm_num_rounds,current_seed,verboseeval)



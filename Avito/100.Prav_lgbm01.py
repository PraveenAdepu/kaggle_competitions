# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:42:20 2018

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

# Gradient Boosting
import lightgbm as lgb

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


y = training.deal_probability.copy()
#training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))


print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)
#del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print("Feature Engineering")
df["price"] = np.log(df["price"]+0.001)
df["price"].fillna(-999,inplace=True)
df["image_top_1"].fillna(-999,inplace=True)

print("\nCreate Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["Day of Month"] = df['activation_date'].dt.day

# Create Validation Index and Remove Dead Variables
training_index = df.loc[df.activation_date<=pd.to_datetime('2017-04-07')].index
validation_index = df.loc[df.activation_date>=pd.to_datetime('2017-04-08')].index
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
    df[cols + '_num_chars'] = df[cols].apply(len) # Count number of Characters
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
            max_features=18000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('text_feat',CountVectorizer(
            ngram_range=(1, 2),
            #max_features=7000,
            preprocessor=get_col('text_feat'))),
        ('title',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            #max_features=7000,
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

print("Modeling Stage")
# Combine Dense Features with Sparse Text Bag of Words Features
trainingSet = hstack([csr_matrix(df_requiredset.iloc[traindex,:].values),ready_df[0:traindex.shape[0]]]) # Sparse Matrix
testingSet = hstack([csr_matrix(df_requiredset.iloc[testdex,:].values),ready_df[traindex.shape[0]:]])
tfvocab = df_requiredset.columns.tolist() + tfvocab
for shape in [trainingSet,testingSet]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
#del df
gc.collect();

print("\nModeling Stage")

## Training and Validation Set
#"""
#Using Randomized train/valid split doesn't seem to generalize LB score, so I will try time cutoff
#"""
#X_train, X_valid, y_train, y_valid = train_test_split(
#    X, y, test_size=0.10, random_state=23)
#    
#print("Light Gradient Boosting Regressor")
#lgbm_params =  {
#    'task': 'train',
#    'boosting_type': 'gbdt',
#    'objective': 'regression',
#    'metric': 'rmse',
#    'max_depth': 15,
#    # 'num_leaves': 31,
#    # 'feature_fraction': 0.65,
#    'bagging_fraction': 0.8,
#    # 'bagging_freq': 5,
#    'learning_rate': 0.019,
#    'verbose': 0
#}  
#
## LGBM Dataset Formatting 
#lgtrain = lgb.Dataset(X_train, y_train,
#                feature_name=tfvocab,
#                categorical_feature = categorical)
#lgvalid = lgb.Dataset(X_valid, y_valid,
#                feature_name=tfvocab,
#                categorical_feature = categorical)
#
## Go Go Go
#modelstart = time.time()
#
#def lgb_mclip(preds, y):
#    y = np.array(list(y.get_label()))
#    score = np.sqrt(metrics.mean_squared_error(y.clip(0.,1.), preds.clip(0.,1.)))
#    return 'R_CLIP', score, False
#    
#lgb_clf = lgb.train(
#    lgbm_params,
#    lgtrain,
#    num_boost_round=9000,
#    valid_sets=[lgtrain, lgvalid],
#    valid_names=['train','valid'],
#    feval=lgb_mclip,
#    early_stopping_rounds=200,
#    verbose_eval=200
#)
#
## Feature Importance Plot
#f, ax = plt.subplots(figsize=[7,10])
#lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
#plt.title("Light GBM Feature Importance")
#plt.savefig('feature_import.png')
#
#print("Model Evaluation Stage")
#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
#lgpred = lgb_clf.predict(testing)
#lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
#lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
#lgsub.to_csv("lgsub.csv",index=True,header=True)
#print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
#print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))



print("Light Gradient Boosting Regressor")
#lgbm_params =  {
#    'task': 'train',
#    'boosting_type': 'gbdt',
#    'objective': 'regression',
#    'metric': 'rmse',
#    'max_depth': 8,
#    # 'num_leaves': 31,
#    'feature_fraction': 0.7,
#    'bagging_fraction': 0.7,
#    # 'bagging_freq': 5,
#    'learning_rate': 0.01,
#    'verbose': 1
#}  


lgbm_params = {
          'task': 'train',
          'boosting_type': 'gbdt',
          'objective': 'regression',
          'metric': 'rmse',
          'min_child_weight': 16,
          'num_leaves': 2**4, #2**4,
          #'lambda_l2': 2,
          'feature_fraction': 0.7,
          'bagging_fraction': 0.7,#0.7
          'bagging_freq': 4,#2
          'learning_rate': 0.01,
          'tree_method': 'exact',
          'min_data_in_leaf':4,
          'min_sum_hessian_in_leaf': 0.8,          
          'nthread': 20,
          'silent': False,
          'seed': 2018,
          'verbose': 1
         }
#lgbm_num_round = 610
#lgbm_early_stopping_rounds = 100


def lgb_mclip(preds, y):
    y = np.array(list(y.get_label()))
    score = np.sqrt(metrics.mean_squared_error(y.clip(0.,1.), preds.clip(0.,1.)))
    return 'R_CLIP', score, False

def train_lgbm(i):
    trainindex = training[training['CVindices'] != i].index.tolist()
    valindex   = training[training['CVindices'] == i].index.tolist()
    
    X_build_set  = training[training['CVindices'] != i]
    X_valid_set  = training[training['CVindices'] == i]
    
    X_val_df = training.iloc[valindex,:]
    
    
    X_build , X_valid = trainingSet.tocsc()[trainindex,:], trainingSet.tocsc()[valindex,:]
    y_build , y_valid = X_build_set.deal_probability.values, X_valid_set.deal_probability.values
    
    lgtrain = lgb.Dataset(X_build, y_build,
                feature_name=tfvocab,
                categorical_feature = categorical
                )
    lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=tfvocab,
                categorical_feature = categorical
                )
    
    lgb_clf = lgb.train(
                    lgbm_params,
                    lgtrain,
                    num_boost_round=5000,
                    valid_sets=[lgtrain, lgvalid],
                    valid_names=['train','valid'],
                    feval=lgb_mclip,
                    early_stopping_rounds=100,
                    verbose_eval=200
                )
                    
    print("Model Evaluation Stage")
    
    pred_cv = lgb_clf.predict(X_valid)
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, pred_cv)))
    
    pred_cv = pd.DataFrame(pred_cv)    
    pred_cv.columns = ["deal_probability"]
    pred_cv["item_id"] = X_val_df.item_id.values
    pred_cv = pred_cv[["item_id","deal_probability"]]
    
    sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
    pred_cv.to_csv(sub_valfile, index=False)
    
    pred_test = lgb_clf.predict(testingSet)
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["deal_probability"]
    pred_test["item_id"] = testing.item_id.values
    pred_test = pred_test[["item_id","deal_probability"]]
    
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '-test' + '.csv'
    pred_test.to_csv(sub_file, index=False)
    del pred_cv
    del pred_test
    
folds = 5
#i = 1
#train_lgbm(i)
ModelName = 'lgbm01'

if __name__ == '__main__':
    for i in range(1, folds+1):
        train_lgbm(i)
#    fulltrain_lgbm(folds)    







#param = {}
#param['objective'] = 'multi:softprob'
#param['eta'] = 0.01
#param['max_depth'] = 6
#param['silent'] = 1
#param['num_class'] = 3
#param['eval_metric'] = "mlogloss"
#param['min_child_weight'] = 1
#param['subsample'] = 0.7
#param['colsample_bytree'] = 0.7
#param['nthread'] = 25
#param['seed'] = 2017
#param['print_every_n'] = 50
#num_rounds = 6000
#plst = list(param.items())
#
#
#def train_xgboost(i):
#    trainindex = train_df[train_df['CVindices'] != i].index.tolist()
#    valindex   = train_df[train_df['CVindices'] == i].index.tolist()
#    
#    X_val_df = train_df.iloc[valindex,:]
#    
#    X_build , X_valid = train_X[trainindex,:], train_X[valindex,:]
#    y_build , y_valid = train_y.iloc[trainindex,:], train_y.iloc[valindex,:]
#    
#    xgbbuild = xgb.DMatrix(X_build, label=y_build)
#    xgbval = xgb.DMatrix(X_valid, label=y_valid)
#    watchlist = [ (xgbbuild,'build'), (xgbval, 'valid') ]
#                 
#    model = xgb.train(plst, 
#                      xgbbuild, 
#                      num_rounds, 
#                      watchlist, 
#                      verbose_eval = 100 #,
#                      #early_stopping_rounds=20
#                      )
#    pred_cv = model.predict(xgbval)
#    pred_cv = pd.DataFrame(pred_cv)
#    pred_cv.columns = ["high", "medium", "low"]
#    pred_cv["listing_id"] = X_val_df.listing_id.values
#    
#    sub_valfile = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.xgb13.fold' + str(i) + '.csv'
#    pred_cv.to_csv(sub_valfile, index=False)
#    
#    xgtest = xgb.DMatrix(test_X)
#    pred_test = model.predict(xgtest)
#    pred_test = pd.DataFrame(pred_test)
#    pred_test.columns = ["high", "medium", "low"]
#    pred_test["listing_id"] = test_df.listing_id.values
#   
#    sub_file = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.xgb13.fold' + str(i) + '-test' + '.csv'
#    pred_test.to_csv(sub_file, index=False)
#    del pred_cv
#    del pred_test
#
#fullnum_rounds = int(num_rounds * 1.2)
#
#def fulltrain_xgboost(bags):
#    xgbtrain = xgb.DMatrix(train_X, label=train_y)
#    watchlist = [ (xgbtrain,'train') ]
#    fullmodel = xgb.train(plst, 
#                              xgbtrain, 
#                              fullnum_rounds, 
#                              watchlist,
#                              verbose_eval = 100,
#                              )
#    xgtest = xgb.DMatrix(test_X)
#    predfull_test = fullmodel.predict(xgtest)
#    predfull_test = pd.DataFrame(predfull_test)
#    predfull_test.columns = ["high", "medium", "low"]
#    predfull_test["listing_id"] = test_df.listing_id.values
#   
#    sub_file = 'C:/Users/SriPrav/Documents/R/21Rental/submissions/Prav.xgb13.full' + '.csv'
#    predfull_test.to_csv(sub_file, index=False)
#
#folds = 5
#i = 1
#if __name__ == '__main__':
#    for i in range(1, folds+1):
#        train_xgboost(i)
#    fulltrain_xgboost(folds)
    
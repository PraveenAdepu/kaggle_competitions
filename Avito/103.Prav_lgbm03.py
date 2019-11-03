# -*- coding: utf-8 -*-
"""
Created on Mon May 14 20:24:12 2018

@author: SriPrav
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb
import lightgbm as lightgbm

color = sns.color_palette()
%matplotlib inline

#import plotly.offline as py
#py.init_notebook_mode(connected=True)
#import plotly.graph_objs as go
#import plotly.tools as tls

#pd.options.mode.chained_assignment = None
#pd.options.display.max_columns = 999

inDir = 'C:/Users/SriPrav/Documents/R/48Avito'

train_df =  pd.read_csv(inDir+'/input/train.csv', parse_dates = ["activation_date"])
test_df = pd.read_csv(inDir+'/input/test.csv', parse_dates=["activation_date"])


print("Train file rows and columns are : ", train_df.shape)
print("Test file rows and columns are : ", test_df.shape)

train_df.head()

train_df["title_nwords"] = train_df["title"].apply(lambda x: len(x.split()))
test_df["title_nwords"] = test_df["title"].apply(lambda x: len(x.split()))

### TFIDF Vectorizer ###
tfidf_vec = TfidfVectorizer(ngram_range=(1,1))
full_tfidf = tfidf_vec.fit_transform(train_df['title'].values.tolist() + test_df['title'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['title'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['title'].values.tolist())

### SVD Components ###
n_comp = 10
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd

## Filling missing values ##
train_df["description"].fillna("NA", inplace=True)
test_df["description"].fillna("NA", inplace=True)

train_df["desc_nwords"] = train_df["description"].apply(lambda x: len(x.split()))
test_df["desc_nwords"] = test_df["description"].apply(lambda x: len(x.split()))

### TFIDF Vectorizer ###
tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features=100000)
full_tfidf = tfidf_vec.fit_transform(train_df['description'].values.tolist() + test_df['description'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['description'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['description'].values.tolist())

### SVD Components ###
n_comp = 30
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = ['svd_desc_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_desc_'+str(i+1) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd

# Target and ID variables #
train_y = train_df["deal_probability"].values
test_id = test_df["item_id"].values

# New variable on weekday #
train_df["activation_weekday"] = train_df["activation_date"].dt.weekday
test_df["activation_weekday"] = test_df["activation_date"].dt.weekday

# Label encode the categorical variables #
cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]
for col in cat_vars:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

train_df.columns

cols_to_drop = ["item_id", "user_id", "title", "description", "activation_date", "image"]
train_X = train_df.drop(cols_to_drop +["deal_probability"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)


Prav_5folds_CVIndices = pd.read_csv(inDir+'./input/Prav_5folds_CVindices_weekdayStratified.csv')

trainingSet = pd.merge(train_df, Prav_5folds_CVIndices, how = 'inner', on = 'item_id')

testingSet = test_df

Prav_train_FE01 = pd.read_csv(inDir+'/input/Prav_train_FE_01.csv')
Prav_test_FE01 = pd.read_csv(inDir+'/input/Prav_test_FE_01.csv')

trainingSet = pd.merge(trainingSet, Prav_train_FE01, how = 'left', on = 'item_id')
testingSet  = pd.merge(testingSet, Prav_test_FE01, how = 'left', on = 'item_id')

feature_names = [c for c in trainingSet if c not in ['item_id', "user_id", "title", "description", "activation_date", "image", 'deal_probability','CVindices','title_nwords','desc_nwords']]


print("Light Gradient Boosting Regressor")

lgbm_params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        'nthread': 30,
        "seed" : 2018,
        "verbosity" : -1
    }
lgbm_num_round = 1000
lgbm_early_stopping_rounds = 100


seed = 2018
folds = 5
nbags = 5
nn_epoch = 12
full_nn_epoch = 12
current_seed = 2017
verboseeval=500


def lgb_mclip(preds, y):
    y = np.array(list(y.get_label()))
    score = np.sqrt(metrics.mean_squared_error(y.clip(0.,1.), preds.clip(0.,1.)))
    return 'R_CLIP', score, False

def train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,ModelName,lgbmParameters, lgbm_num_rounds,current_seed,verboseeval):
    
    X_build  = trainingSet[trainingSet['CVindices'] != i]
    X_valid  = trainingSet[trainingSet['CVindices'] == i]
     
    lgbmbuild = lightgbm.Dataset(X_build[feature_names], X_build['deal_probability'])
    lgbmval   = lightgbm.Dataset(X_valid[feature_names], X_valid['deal_probability'])
                 
    pred_cv = np.zeros(X_valid.shape[0])
    pred_test = np.zeros(testingSet.shape[0])
    
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        bag_cv = np.zeros(X_valid.shape[0])
        lgbmParameters['bagging_seed'] =  current_seed + j    
        lgbmParameters['seed'] =  current_seed + j 
        model = lightgbm.train(lgbmParameters, lgbmbuild,lgbm_num_rounds,feval=lgb_mclip#,early_stopping_rounds=100
                               ,valid_sets=[lgbmbuild,lgbmval],valid_names=['train','valid'],verbose_eval = verboseeval
                                  #,early_stopping_rounds=early_stopping_rounds
                                  )
        bag_cv= model.predict(X_valid[feature_names] #,num_iteration=model.best_iteration
                             )#.reshape(-1,1)
        pred_test += model.predict(testingSet[feature_names])
        pred_cv += bag_cv
        bag_score = np.sqrt(metrics.mean_squared_error(X_valid['deal_probability'], bag_cv))
        print('bag ', j, '- rmse :', bag_score)
    pred_cv /= nbags
    pred_test/= nbags
    fold_score = np.sqrt(metrics.mean_squared_error(X_valid['deal_probability'], pred_cv))
    print('Fold ', i, '- rmse :', fold_score)
    
    pred_cv = pd.DataFrame(pred_cv)
    pred_cv.columns = ["deal_probability"]
    pred_cv["item_id"] = X_valid.item_id.values
    
    sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
    pred_cv['deal_probability'] = pred_cv['deal_probability'].clip(0.0, 1.0)
    pred_cv[["item_id","deal_probability"]].to_csv(sub_valfile, index=False)
    
    pred_test = pd.DataFrame(pred_test)
    pred_test.columns = ["deal_probability"]
    pred_test["item_id"] = testingSet.item_id.values
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '-test' + '.csv'
    pred_test['deal_probability'] = pred_test['deal_probability'].clip(0.0, 1.0)
    pred_test[["item_id","deal_probability"]].to_csv(sub_file, index=False)
    del pred_cv
    del pred_test

def fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,ModelName,lgbmParameters, lgbm_num_rounds,current_seed,verboseeval):
    
    full_lgbm_num_rounds = int(lgbm_num_rounds * 1.2)
    lgbmtrain = lightgbm.Dataset(trainingSet[feature_names], trainingSet['deal_probability'])

    predfull_test = np.zeros(testingSet.shape[0]) 
    for j in range(1,nbags+1):
        print('bag ', j , ' Processing')
        lgbmParameters['bagging_seed'] =  current_seed + j  
        lgbmParameters['seed'] =  current_seed + j          
        fullmodel = lightgbm.train(lgbmParameters, lgbmtrain,full_lgbm_num_rounds,feval=lgb_mclip
                                   ,valid_sets=[lgbmtrain,lgbmtrain],valid_names=['train','train'],verbose_eval = verboseeval
                                  #,early_stopping_rounds=early_stopping_rounds
                                  )
    
        predfull_test += fullmodel.predict(testingSet[feature_names])
    predfull_test/= nbags
    predfull_test = pd.DataFrame(predfull_test)
    predfull_test.columns = ["deal_probability"]
    predfull_test["item_id"] = testingSet.item_id.values
   
    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
    predfull_test['deal_probability'] = predfull_test['deal_probability'].clip(0.0, 1.0)
    predfull_test[["item_id","deal_probability"]].to_csv(sub_file, index=False)

lgbmModelName = 'lgbm003'

if __name__ == '__main__':        
    for i in range(1, folds+1):
        train_lgbm_regression(trainingSet, testingSet,feature_names,i,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)
    fulltrain_lgbm_regression(trainingSet, testingSet,feature_names,nbags,lgbmModelName,lgbm_params, lgbm_num_round,current_seed,verboseeval)

########################################################################################################
#def train_lgbm(i):
#    trainindex = training[training['CVindices'] != i].index.tolist()
#    valindex   = training[training['CVindices'] == i].index.tolist()
#    
#    X_build_set  = training[training['CVindices'] != i]
#    X_valid_set  = training[training['CVindices'] == i]
#    
#    X_val_df = training.iloc[valindex,:]
#    
#    
#    X_build , X_valid = trainingSet.tocsc()[trainindex,:], trainingSet.tocsc()[valindex,:]
#    y_build , y_valid = X_build_set.deal_probability.values, X_valid_set.deal_probability.values
#    
#    lgtrain = lgb.Dataset(X_build, y_build,
#                feature_name=tfvocab,
#                categorical_feature = categorical
#                )
#    lgvalid = lgb.Dataset(X_valid, y_valid,
#                feature_name=tfvocab,
#                categorical_feature = categorical
#                )
#    
#    lgb_clf = lgb.train(
#                    lgbm_params,
#                    lgtrain,
#                    num_boost_round=5000,
#                    valid_sets=[lgtrain, lgvalid],
#                    valid_names=['train','valid'],
#                    feval=lgb_mclip,
#                    early_stopping_rounds=100,
#                    verbose_eval=200
#                )
#                    
#    print("Model Evaluation Stage")
#    
#    pred_cv = lgb_clf.predict(X_valid)
#    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, pred_cv)))
#    
#    pred_cv = pd.DataFrame(pred_cv)    
#    pred_cv.columns = ["deal_probability"]
#    pred_cv["item_id"] = X_val_df.item_id.values
#    pred_cv = pred_cv[["item_id","deal_probability"]]
#    
#    sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
#    pred_cv.to_csv(sub_valfile, index=False)
#    
#    pred_test = lgb_clf.predict(testingSet)
#    pred_test = pd.DataFrame(pred_test)
#    pred_test.columns = ["deal_probability"]
#    pred_test["item_id"] = testing.item_id.values
#    pred_test = pred_test[["item_id","deal_probability"]]
#    
#    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '-test' + '.csv'
#    pred_test.to_csv(sub_file, index=False)
#    del pred_cv
#    del pred_test
#    
#folds = 5
#i = 1
#ModelName = 'lgbm02'
#
#if __name__ == '__main__':
#    for i in range(1, folds+1):
#        train_lgbm(i)
#
#####################################################################################################################################
#def run_lgb(train_X, train_y, val_X, val_y, test_X):
#    params = {
#        "objective" : "regression",
#        "metric" : "rmse",
#        "num_leaves" : 30,
#        "learning_rate" : 0.1,
#        "bagging_fraction" : 0.7,
#        "feature_fraction" : 0.7,
#        "bagging_frequency" : 5,
#        "bagging_seed" : 2018,
#        "verbosity" : -1
#    }
#    
#    lgtrain = lgb.Dataset(train_X, label=train_y)
#    lgval = lgb.Dataset(val_X, label=val_y)
#    evals_result = {}
#    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgtrain, lgval],
#                    valid_names=['train','valid'], early_stopping_rounds=100, verbose_eval=20, evals_result=evals_result)
#    
#    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
#    return pred_test_y, model, evals_result
#
## Splitting the data for model training#
#dev_X = train_X.iloc[:-200000,:]
#val_X = train_X.iloc[-200000:,:]
#dev_y = train_y[:-200000]
#val_y = train_y[-200000:]
#print(dev_X.shape, val_X.shape, test_X.shape)
#
## Training the model #
#pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
#
## Making a submission file #
#pred_test[pred_test>1] = 1
#pred_test[pred_test<0] = 0
#sub_df = pd.DataFrame({"item_id":test_id})
#sub_df["deal_probability"] = pred_test
#sub_df.to_csv("baseline_lgb.csv", index=False)


###########
# ORIGINAL:
# the1owl1 - https://www.kaggle.com/the1owl/surprise-me
# 
# The only addition is a KNN Regressor and a small ET tweak and a removal of the Linear Regression Model
#############################################################


import numpy as np
import pandas as pd
from sklearn import ensemble, neighbors, linear_model, metrics, preprocessing
from datetime import datetime
import datetime

import xgboost as xgb


inDir =r'C:\Users\SriPrav\Documents\R\40Recruit'

data = {
    'tra': pd.read_csv(inDir + '/input/air_visit_data.csv'),
    'as': pd.read_csv(inDir + '/input/air_store_info.csv'),
    'hs': pd.read_csv(inDir + '/input/hpg_store_info.csv'),
    'ar': pd.read_csv(inDir + '/input/air_reserve.csv'),
    'hr': pd.read_csv(inDir + '/input/hpg_reserve.csv'),
    'id': pd.read_csv(inDir + '/input/store_id_relation.csv'),
    'tes': pd.read_csv(inDir + '/input/sample_submission.csv'),
    'hol': pd.read_csv(inDir + '/input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
#    data[df] = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date'})
    print(data[df].head())

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date


data['build'] = data['tra'][(data['tra']['visit_date']<datetime.date(2017,3,12))] 
data['valid'] = data['tra'][(data['tra']['visit_date']>= datetime.date(2017,3,12)) & (data['tra']['visit_date']<= datetime.date(2017,4,19)) ]

unique_stores = data['valid']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)


tmp = data['build'].groupby(['air_store_id','dow']).agg({'visitors': [min,np.mean,np.median,max,'count', np.std]})    
tmp.columns = ["_".join(x) for x in tmp.columns.ravel()]
tmp.reset_index(level=tmp.index.names, inplace=True)

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 


stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 
lbl = preprocessing.LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])


data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

X_build = pd.merge(data['build'], data['hol'], how='left', on=['visit_date']) 
X_valid = pd.merge(data['valid'], data['hol'], how='left', on=['visit_date']) 

X_build = pd.merge(X_build, stores, how='left', on=['air_store_id','dow']) 
X_valid = pd.merge(X_valid , stores, how='left', on=['air_store_id','dow'])


artemp = data['ar'].groupby(['air_store_id','visit_datetime']).agg({'reserve_datetime_diff': [min,np.mean,np.median,max,'count', np.std] ,
                                                                                   'reserve_visitors': [min,np.mean,np.median,max,'count', np.std]
                                                                                   })

artemp.columns = ["_".join(x) for x in artemp.columns.ravel()]
artemp.reset_index(level=artemp.index.names, inplace=True)
artemp = artemp.rename(columns={'visit_datetime': 'visit_date'})

hrtemp = data['hr'].groupby(['air_store_id','visit_datetime']).agg({'reserve_datetime_diff': [min,np.mean,np.median,max,'count', np.std] ,
                                                                                   'reserve_visitors': [min,np.mean,np.median,max,'count', np.std]
                                                                                   })
hrtemp.columns = ["_".join(x) for x in hrtemp.columns.ravel()]
hrtemp.reset_index(level=hrtemp.index.names, inplace=True)

hrtemp = hrtemp.rename(columns={'visit_datetime': 'visit_date'})


X_build = pd.merge(X_build, artemp, how='left', on=['air_store_id','visit_date'])
X_build = pd.merge(X_build, hrtemp, how='left', on=['air_store_id','visit_date'])

X_valid = pd.merge(X_valid, artemp, how='left', on=['air_store_id','visit_date'])
X_valid = pd.merge(X_valid, hrtemp, how='left', on=['air_store_id','visit_date'])


    
print(X_build.describe())
print(X_build.head())

X_build.columns
X_valid.columns

col = [c for c in X_build if c not in ['id', 'air_store_id','visit_date','visitors']]
X_build = X_build.fillna(-1)
X_valid = X_valid.fillna(-1)

feature_names = col

#['dow',
# 'year',
# 'month',
# 'day_of_week',
# 'holiday_flg',
# 'min_visitors',
# 'mean_visitors',
# 'median_visitors',
# 'max_visitors',
# 'count_observations',
# 'air_genre_name',
# 'air_area_name',
# 'latitude',
# 'longitude',
# 'reserve_datetime_diff_x',
# 'reserve_visitors_x',
# 'reserve_datetime_diff_y',
# 'reserve_visitors_y']
#############################################################################################################################################
# xgb regression ############################################################################################################################
#############################################################################################################################################


param = {}
param['seed'] = 2017
param['objective'] = 'reg:linear'
param['eta'] = 0.01
param['max_depth'] = 5
param['silent'] = 1
param['min_child_weight'] = 10
param['subsample'] = 0.9
param['colsample_bytree'] = 0.7
param['nthread'] = 16
param['print_every_n'] = 25
param['booster'] = 'gbtree'
#param['base_score'] = y_mean
param['eval_metric'] = "rmse"
xgb_num_rounds = 2250

xgbParameters = list(param.items())

seed = 2017
nbags = 5
current_seed = 2017
verboseeval=250
num_rounds = 2250

ModelName = 'xgb02'

def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

xgbbuild = xgb.DMatrix(X_build[feature_names], label=np.log1p(X_build['visitors']))
xgbval = xgb.DMatrix(X_valid[feature_names], label=np.log1p(X_valid['visitors']))
watchlist = [ (xgbbuild,'build'), (xgbval, 'valid') ]
            
pred_cv = np.zeros(X_valid.shape[0])

for j in range(1,nbags+1):
    print('bag ', j , ' Processing')
    bag_seed = current_seed + j
    xgbParameters[6] = ('seed',bag_seed)
    print('bag seed ', bag_seed , ' Processing')
    bag_cv = np.zeros(X_valid.shape[0])         
    model = xgb.train(xgbParameters, 
                      xgbbuild, 
                      num_rounds, 
                      watchlist,
                      #feval=gini_xgb, 
                      maximize=True,
                      verbose_eval = verboseeval                  
                      )
    bag_cv  = model.predict(xgbval)        
    
    pred_cv += bag_cv
    bag_score = RMSLE(np.log1p(X_valid['visitors']), bag_cv)
    print('bag ', j, '- rmsle :', bag_score)
pred_cv /= nbags

fold_score = RMSLE(np.log1p(X_valid['visitors']), pred_cv)
print('fold score ', '- rmsle :', fold_score)
X_valid['visitor_pred_xgb'] = np.expm1(pred_cv)

sub_file = inDir +'/submissions/Prav_xgb02_validation_preds.csv'
X_valid[['air_store_id','visit_date','visitor_pred_xgb']].to_csv(sub_file, index=False)

#################################################################################################################################

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)


tmp = data['tra'].groupby(['air_store_id','dow']).agg({'visitors': [min,np.mean,np.median,max,'count', np.std]})    
tmp.columns = ["_".join(x) for x in tmp.columns.ravel()]
tmp.reset_index(level=tmp.index.names, inplace=True)

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 


stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 
lbl = preprocessing.LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

#data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
#data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
#data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 

train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 
test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])




train = pd.merge(train, artemp, how='left', on=['air_store_id','visit_date'])
train = pd.merge(train, hrtemp, how='left', on=['air_store_id','visit_date'])

test = pd.merge(test, artemp, how='left', on=['air_store_id','visit_date'])
test = pd.merge(test, hrtemp, how='left', on=['air_store_id','visit_date'])

    
print(train.describe())
print(train.head())

#col = [c for c in train if c not in ['id', 'air_store_id','visit_date','visitors']]
train = train.fillna(-1)
test = test.fillna(-1)

#feature_names = col

fullnum_rounds = int(num_rounds * 1.2)
xgbtrain = xgb.DMatrix(train[feature_names], label=np.log1p(train['visitors']))
watchlist = [ (xgbtrain,'train') ]
xgtest = xgb.DMatrix(test[feature_names])
predfull_test = np.zeros(test.shape[0]) 
for j in range(1,nbags+1):
    print('bag ', j , ' Processing')
    bag_seed = current_seed + j
    xgbParameters[6] = ('seed',bag_seed)
    print('bag seed ', bag_seed , ' Processing')            
    fullmodel = xgb.train(xgbParameters, 
                          xgbtrain, 
                          fullnum_rounds, 
                          watchlist,
                          #feval=gini_xgb, 
                          #maximize=True,
                          verbose_eval = verboseeval,
                          )

    predfull_test += fullmodel.predict(xgtest)
predfull_test/= nbags
test['visitors_pred'] = np.expm1(predfull_test)

   
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
test[['air_store_id','visit_date','visitors_pred']].to_csv(sub_file, index=False)
    
########################################################################################################################################################### 
########################################################################################################################################################### 

   
#def ceate_feature_map(features):
#    outfile = open(inDir +'/ModelLogs/xgb.fmap', 'w')
#    i = 0
#    for feat in features:
#        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
#        i = i + 1
#
#    outfile.close()
#
#def xgb_r2_score(preds, dtrain):
#    labels = dtrain.get_label()
#    return 'r2', r2_score(labels, preds)
#
#def gini(y, pred):
#    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
#    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
#    gs = g[:,0].cumsum().sum() / g[:,0].sum()
#    gs -= (len(y) + 1) / 2.
#    return gs / len(y)
#
#def gini_xgb(pred, y):
#    y = y.get_label()
#    return 'gini', gini(y, pred) / gini(y, y)
#
#def gini_normalized(y, pred):
#    return gini(y, pred) / gini(y, y)

#def train_xgboost_regression(trainingSet, testingSet,feature_names,i,nbags,ModelName,xgbParameters, num_rounds,current_seed, verboseeval):
#    
#    X_build = trainingSet[trainingSet['CVindices'] != i]
#    X_valid   = trainingSet[trainingSet['CVindices'] == i]
#     
#    xgbbuild = xgb.DMatrix(X_build[feature_names], label=X_build['target'])
#    xgbval = xgb.DMatrix(X_valid[feature_names], label=X_valid['target'])
#    watchlist = [ (xgbbuild,'build'), (xgbval, 'valid') ]
#    
#    xgtest = xgb.DMatrix(testingSet[feature_names])
#                 
#    pred_cv = np.zeros(X_valid.shape[0])
#    pred_test = np.zeros(testingSet.shape[0])
#    
#    for j in range(1,nbags+1):
#        print('bag ', j , ' Processing')
#        bag_seed = current_seed + j
#        xgbParameters[6] = ('seed',bag_seed)
#        print('bag seed ', bag_seed , ' Processing')
#        bag_cv = np.zeros(X_valid.shape[0])         
#        model = xgb.train(xgbParameters, 
#                          xgbbuild, 
#                          num_rounds, 
#                          watchlist,
#                          feval=gini_xgb, 
#                          maximize=True,
#                          verbose_eval = verboseeval                  
#                          )
#        bag_cv  = model.predict(xgbval)        
#        pred_test += model.predict(xgtest)
#        pred_cv += bag_cv
#        bag_score = gini_normalized(X_valid['target'], bag_cv)
#        print('bag ', j, '- gini :', bag_score)
#    pred_cv /= nbags
#    pred_test/= nbags
#    fold_score = gini_normalized(X_valid['target'], pred_cv)
#    print('Fold ', i, '- gini :', fold_score)
#    
#    pred_cv = pd.DataFrame(pred_cv)
#    pred_cv.columns = ["target"]
#    pred_cv["id"] = X_valid.id.values
#    
#    sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '.csv'
#    pred_cv[["id","target"]].to_csv(sub_valfile, index=False)
#    
#    pred_test = pd.DataFrame(pred_test)
#    pred_test.columns = ["target"]
#    pred_test["id"] = testingSet.id.values
#   
#    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.fold' + str(i) + '-test' + '.csv'
#    pred_test[["id","target"]].to_csv(sub_file, index=False)
#    del pred_cv
#    del pred_test
#
#
#
#def fulltrain_xgboost_regression(trainingSet, testingSet,feature_names,nbags,ModelName,xgbParameters, num_rounds,current_seed,verboseeval):
#    fullnum_rounds = int(num_rounds * 1.2)
#    xgbtrain = xgb.DMatrix(trainingSet[feature_names], label=trainingSet['target'])
#    watchlist = [ (xgbtrain,'train') ]
#    xgtest = xgb.DMatrix(testingSet[feature_names])
#    predfull_test = np.zeros(testingSet.shape[0]) 
#    for j in range(1,nbags+1):
#        print('bag ', j , ' Processing')
#        bag_seed = current_seed + j
#        xgbParameters[6] = ('seed',bag_seed)
#        print('bag seed ', bag_seed , ' Processing')            
#        fullmodel = xgb.train(xgbParameters, 
#                              xgbtrain, 
#                              fullnum_rounds, 
#                              watchlist,
#                              feval=gini_xgb, 
#                              maximize=True,
#                              verbose_eval = verboseeval,
#                              )
#    
#        predfull_test += fullmodel.predict(xgtest)
#    predfull_test/= nbags
#    predfull_test = pd.DataFrame(predfull_test)
#    predfull_test.columns = ["target"]
#    predfull_test["id"] = testingSet.id.values
#   
#    sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
#    predfull_test[["id","target"]].to_csv(sub_file, index=False)
#    ceate_feature_map(feature_names)
#    importance = fullmodel.get_fscore(fmap=inDir +'/ModelLogs/xgb.fmap')
#    importance = sorted(importance.items(), key=operator.itemgetter(1))
#    
#    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
#    df['fscore'] = df['fscore'] / df['fscore'].sum()
#    imp_file = inDir +'/ModelLogs/Prav.'+ str(ModelName)+'.featureImportance' + '.csv'
#    df = df.sort_values(['fscore'], ascending=[False])
#    df[['feature', 'fscore']].to_csv(imp_file, index=False)
#

#############################################################################################################################################
# xgb regression ############################################################################################################################
#############################################################################################################################################

#
##lr = linear_model.LinearRegression(n_jobs=-1)
#etc = ensemble.ExtraTreesRegressor(n_estimators=225, max_depth=5, n_jobs=-1, random_state=3)
#knn = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)
##lr.fit(train[col], np.log1p(train['visitors'].values))
#etc.fit(X_build[col], np.log1p(X_build['visitors'].values))
#knn.fit(X_build[col], np.log1p(X_build['visitors'].values))
#
#X_valid.head()
#X_valid['visitor_pred_ET'] = np.expm1(etc.predict(X_valid[col]))
#X_valid['visitor_pred_knn'] = np.expm1(knn.predict(X_valid[col]))
#
#print('RMSE ExtraTreesRegressor: ', RMSLE(np.log1p(X_valid['visitors'].values), np.log1p(X_valid['visitor_pred_ET'].values)))
#print('RMSE KNNRegressor: ', RMSLE(np.log1p(X_valid['visitors'].values), np.log1p(X_valid['visitor_pred_knn'].values)))
#
##print('RMSE LinearRegressor: ', RMSLE(np.log1p(train['visitors'].values), lr.predict(train[col])))
#print('RMSE ExtraTreesRegressor: ', RMSLE(np.log1p(X_build['visitors'].values), etc.predict(X_build[col])))
#print('RMSE KNNRegressor: ', RMSLE(np.log1p(X_build['visitors'].values), knn.predict(X_build[col])))
#
#sub_file = inDir +'/submissions/Prav_ET_knn_validation_preds.csv'
#X_valid[['air_store_id','visit_date','visitor_pred_ET','visitor_pred_knn']].to_csv(sub_file, index=False)
#
#unique_stores = data['tes']['air_store_id'].unique()
#stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)
#
#
##sure it can be compressed...
#tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
#stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 
#tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
#stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
#tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
#stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
#tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
#stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
#tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
#stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 
#
#stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 
#lbl = preprocessing.LabelEncoder()
#stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
#stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])
#
#data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
#data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
#data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
#train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
#test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 
#
#train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 
#test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])
#
#for df in ['ar','hr']:
#    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
#    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])
#    
#print(train.describe())
#print(train.head())
#
#col = [c for c in train if c not in ['id', 'air_store_id','visit_date','visitors']]
#train = train.fillna(-1)
#test = test.fillna(-1)
#
#def RMSLE(y, pred):
#    return metrics.mean_squared_error(y, pred)**0.5
#
##lr = linear_model.LinearRegression(n_jobs=-1)
#etc = ensemble.ExtraTreesRegressor(n_estimators=225, max_depth=5, n_jobs=-1, random_state=3)
#knn = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)
##lr.fit(train[col], np.log1p(train['visitors'].values))
#etc.fit(train[col], np.log1p(train['visitors'].values))
#knn.fit(train[col], np.log1p(train['visitors'].values))
##print('RMSE LinearRegressor: ', RMSLE(np.log1p(train['visitors'].values), lr.predict(train[col])))
#print('RMSE ExtraTreesRegressor: ', RMSLE(np.log1p(train['visitors'].values), etc.predict(train[col])))
#print('RMSE KNNRegressor: ', RMSLE(np.log1p(train['visitors'].values), knn.predict(train[col])))
#
#test['visitors'] = (etc.predict(test[col]) / 2) +(knn.predict(test[col]) / 2)
#test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)
#test[['id','visitors']].to_csv('lr_submission.csv', index=False, float_format='%.2f')
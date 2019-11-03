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
datetime.datetime.strptime


inDir =r'C:\Users\SriPrav\Documents\R\40Recruit'

def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

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

#lr = linear_model.LinearRegression(n_jobs=-1)
etc = ensemble.ExtraTreesRegressor(n_estimators=600, max_depth=5, n_jobs=-1, random_state=3)
knn = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)
#lr.fit(train[col], np.log1p(train['visitors'].values))
etc.fit(X_build[col], np.log1p(X_build['visitors'].values))
knn.fit(X_build[col], np.log1p(X_build['visitors'].values))

X_valid.head()
X_valid['visitor_pred_ET'] = np.expm1(etc.predict(X_valid[col]))
X_valid['visitor_pred_knn'] = np.expm1(knn.predict(X_valid[col]))

print('RMSE ExtraTreesRegressor: ', RMSLE(np.log1p(X_valid['visitors'].values), np.log1p(X_valid['visitor_pred_ET'].values)))
print('RMSE KNNRegressor: ', RMSLE(np.log1p(X_valid['visitors'].values), np.log1p(X_valid['visitor_pred_knn'].values)))

#print('RMSE LinearRegressor: ', RMSLE(np.log1p(train['visitors'].values), lr.predict(train[col])))
print('RMSE ExtraTreesRegressor: ', RMSLE(np.log1p(X_build['visitors'].values), etc.predict(X_build[col])))
print('RMSE KNNRegressor: ', RMSLE(np.log1p(X_build['visitors'].values), knn.predict(X_build[col])))

sub_file = inDir +'/submissions/Prav_ET_knn_02_validation_preds.csv'
X_valid[['air_store_id','visit_date','visitor_pred_ET','visitor_pred_knn']].to_csv(sub_file, index=False)
###################################################################################################################################

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



#lr = linear_model.LinearRegression(n_jobs=-1)
etc = ensemble.ExtraTreesRegressor(n_estimators=720, max_depth=5, n_jobs=-1, random_state=3)
knn = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)
#lr.fit(train[col], np.log1p(train['visitors'].values))
etc.fit(train[col], np.log1p(train['visitors'].values))
knn.fit(train[col], np.log1p(train['visitors'].values))

test['visitor_pred_ET'] = np.expm1(etc.predict(test[col]))
test['visitor_pred_knn'] = np.expm1(knn.predict(test[col]))

#print('RMSE LinearRegressor: ', RMSLE(np.log1p(train['visitors'].values), lr.predict(train[col])))
print('RMSE ExtraTreesRegressor: ', RMSLE(np.log1p(train['visitors'].values), etc.predict(train[col])))
print('RMSE KNNRegressor: ', RMSLE(np.log1p(train['visitors'].values), knn.predict(train[col])))


sub_file = inDir +'/submissions/Prav.ET_knn_02.full.csv'
test[['air_store_id','visit_date','visitor_pred_ET','visitor_pred_knn']].to_csv(sub_file, index=False)

#test['visitors'] = (etc.predict(test[col]) / 2) +(knn.predict(test[col]) / 2)
#test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)
#test[['id','visitors']].to_csv('lr_submission.csv', index=False, float_format='%.2f')
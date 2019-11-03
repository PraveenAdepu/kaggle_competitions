# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 21:29:08 2017

@author: SriPrav
"""

"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


inDir = 'C:\\Users\\SriPrav\\Documents\\R\\34Corporacion'

df_train = pd.read_csv(
    inDir+'/input/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # 2016-01-01
)

df_test = pd.read_csv(
    inDir+"/input/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)

items = pd.read_csv(
    inDir+"/input/items.csv",
).set_index("item_nbr")

df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]
del df_train

promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train

df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)

items = items.reindex(df_2017.index.get_level_values(1))

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
        "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "mean_30_2017": get_timespan(df_2017, t2017, 30, 30).mean(axis=1).values,
        "mean_60_2017": get_timespan(df_2017, t2017, 60, 60).mean(axis=1).values,
        "mean_140_2017": get_timespan(df_2017, t2017, 140, 140).mean(axis=1).values,
        
        "min_3_2017": get_timespan(df_2017, t2017, 3, 3).min(axis=1).values,
        "min_7_2017": get_timespan(df_2017, t2017, 7, 7).min(axis=1).values,
        "min_14_2017": get_timespan(df_2017, t2017, 14, 14).min(axis=1).values,
        "min_30_2017": get_timespan(df_2017, t2017, 30, 30).min(axis=1).values,
        "min_60_2017": get_timespan(df_2017, t2017, 60, 60).min(axis=1).values,
        "min_140_2017": get_timespan(df_2017, t2017, 140, 140).min(axis=1).values,
        
        "max_3_2017": get_timespan(df_2017, t2017, 3, 3).max(axis=1).values,
        "max_7_2017": get_timespan(df_2017, t2017, 7, 7).max(axis=1).values,
        "max_14_2017": get_timespan(df_2017, t2017, 14, 14).max(axis=1).values,
        "max_30_2017": get_timespan(df_2017, t2017, 30, 30).max(axis=1).values,
        "max_60_2017": get_timespan(df_2017, t2017, 60, 60).max(axis=1).values,
        "max_140_2017": get_timespan(df_2017, t2017, 140, 140).max(axis=1).values,
        
        "std_3_2017": get_timespan(df_2017, t2017, 3, 3).std(axis=1).values,
        "std_7_2017": get_timespan(df_2017, t2017, 7, 7).std(axis=1).values,
        "std_14_2017": get_timespan(df_2017, t2017, 14, 14).std(axis=1).values,
        "std_30_2017": get_timespan(df_2017, t2017, 30, 30).std(axis=1).values,
        "std_60_2017": get_timespan(df_2017, t2017, 60, 60).std(axis=1).values,
        "std_140_2017": get_timespan(df_2017, t2017, 140, 140).std(axis=1).values,
        
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values,
        "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values
    })
    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values
    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)
    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X

print("Preparing dataset...")
t2017 = date(2017, 5, 31)
X_l, y_l = [], []
for i in range(6):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2017 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
X_val, y_val = prepare_dataset(date(2017, 7, 26))
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)

non_promo_features = ['day_1_2017', 'max_140_2017', 'max_14_2017', 'max_30_2017',
       'max_3_2017', 'max_60_2017', 'max_7_2017', 'mean_140_2017',
       'mean_14_2017', 'mean_30_2017', 'mean_3_2017', 'mean_60_2017',
       'mean_7_2017', 'min_140_2017', 'min_14_2017', 'min_30_2017',
       'min_3_2017', 'min_60_2017', 'min_7_2017', 'std_140_2017', 'std_14_2017',
       'std_30_2017', 'std_3_2017', 'std_60_2017', 'std_7_2017',
       'mean_4_dow0_2017', 'mean_20_dow0_2017', 'mean_4_dow1_2017',
       'mean_20_dow1_2017', 'mean_4_dow2_2017', 'mean_20_dow2_2017',
       'mean_4_dow3_2017', 'mean_20_dow3_2017', 'mean_4_dow4_2017',
       'mean_20_dow4_2017', 'mean_4_dow5_2017', 'mean_20_dow5_2017',
       'mean_4_dow6_2017', 'mean_20_dow6_2017']
# , 'promo_140_2017', 'promo_14_2017', 'promo_60_2017' , 'promo_0', 'promo_1',
#       'promo_2', 'promo_3', 'promo_4', 'promo_5', 'promo_6', 'promo_7',
#       'promo_8', 'promo_9', 'promo_10', 'promo_11', 'promo_12', 'promo_13',
#       'promo_14', 'promo_15'




all_data = pd.concat([X_train[non_promo_features], X_val[non_promo_features]])

all_data = pd.concat([all_data, X_test[non_promo_features]])

from sklearn.preprocessing import MinMaxScaler

def scale_data(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

X_train.shape
X_val.shape
all_data_scaled, scaler = scale_data(all_data)
X_train1 = all_data_scaled[:1005090, :]
X_val1 = all_data_scaled[1005090:1005090+167515, :]
X_test1 = all_data_scaled[1005090+167515:, :]

#stores_items = pd.read_csv(indir + 'stores_items.csv', index_col=['store_nbr','item_nbr'])
#test_ids = pd.read_csv( indir + 'test_ids.csv',  parse_dates=['date']).set_index(
#                        ['store_nbr', 'item_nbr', 'date'] )
#items = pd.read_csv( indir2 + 'items.csv' ).set_index("item_nbr")
#items = items.reindex( stores_items.index.get_level_values(1) )

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import callbacks
from keras.callbacks import ModelCheckpoint

#def dlmodel():
#    model = Sequential()
#    model.add(Dense(32, kernel_initializer='normal', activation='relu', input_shape=(X_train.shape[1],)))
#    model.add(Dropout(.2))
#    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
#    model.add(Dropout(.2))
#    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
#    model.add(Dropout(.2))
#    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
#    model.add(Dropout(.1))
#    model.add(Dense(1, kernel_initializer='normal'))    
#    return model

def dlmodel():
    model = Sequential()
    model.add(Dense(32, kernel_initializer='normal', activation='relu', input_shape=(X_train[non_promo_features].shape[1],)))
    model.add(Dropout(.2))
#    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
#    model.add(Dropout(.1))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(1, kernel_initializer='normal'))    
    return model

N_EPOCHS = 3
val_pred = []
test_pred = []
# wtpath = 'weights.hdf5'  # To save best epoch. But need Keras bug to be fixed first.
sample_weights=np.array( pd.concat([items["perishable"]] * 6) * 0.25 + 1 )

for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    y = y_train[:, i]
    xv = X_val1
    yv = y_val[:, i]
    model = dlmodel()
    model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
#    bestepoch = ModelCheckpoint( filepath=wtpath, verbose=1, save_best_only=True )
    model.fit( X_train1, y, batch_size = 32, epochs = N_EPOCHS, verbose=1,
               sample_weight=sample_weights, validation_data=(xv,yv) ) 
             #, callbacks=[bestepoch] # bestepoch doesn't work: keras bug
#    model.load_weights( wtpath )
    val_pred.append(model.predict(X_val1))
    test_pred.append(model.predict(X_test1))
    del model
y_val.shape
y_preds = np.array(val_pred).squeeze(axis=2).transpose()
print("Validation mse:", mean_squared_error(
    y_val, y_preds))

print("Making submission...")
y_test = np.array(test_pred).squeeze(axis=2).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv(inDir+'/submissions/Prav_nn03.csv', float_format='%.4f', index=None)

    
n_public = 5 # Number of days in public test set
weights=pd.concat([items["perishable"]]) * 0.25 + 1
print("Unweighted validation mse: ", mean_squared_error(
    y_val, np.array(val_pred).squeeze(axis=2).transpose()) )
print("Full validation mse:       ", mean_squared_error(
    y_val, np.array(val_pred).squeeze(axis=2).transpose(), sample_weight=weights) )
print("'Public' validation mse:   ", mean_squared_error(
    y_val[:,:n_public], np.array(val_pred).squeeze(axis=2).transpose()[:,:n_public], 
    sample_weight=weights) )
print("'Private' validation mse:  ", mean_squared_error(
    y_val[:,n_public:], np.array(val_pred).squeeze(axis=2).transpose()[:,n_public:], 
    sample_weight=weights) )


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

from scipy.stats import kurtosis
from scipy.stats import skew


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
        
        "skew_3_2017": get_timespan(df_2017, t2017, 3, 3).skew(axis=1).values,
        "skew_7_2017": get_timespan(df_2017, t2017, 7, 7).skew(axis=1).values,
        "skew_14_2017": get_timespan(df_2017, t2017, 14, 14).skew(axis=1).values,
        "skew_30_2017": get_timespan(df_2017, t2017, 30, 30).skew(axis=1).values,
        "skew_60_2017": get_timespan(df_2017, t2017, 60, 60).skew(axis=1).values,
        "skew_140_2017": get_timespan(df_2017, t2017, 140, 140).skew(axis=1).values,
        
        "kurtosis_3_2017": get_timespan(df_2017, t2017, 3, 3).kurtosis(axis=1).values,
        "kurtosis_7_2017": get_timespan(df_2017, t2017, 7, 7).kurtosis(axis=1).values,
        "kurtosis_14_2017": get_timespan(df_2017, t2017, 14, 14).kurtosis(axis=1).values,
        "kurtosis_30_2017": get_timespan(df_2017, t2017, 30, 30).kurtosis(axis=1).values,
        "kurtosis_60_2017": get_timespan(df_2017, t2017, 60, 60).kurtosis(axis=1).values,
        "kurtosis_140_2017": get_timespan(df_2017, t2017, 140, 140).kurtosis(axis=1).values,
        
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

X_train.fillna(0, inplace=True)
X_val.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)


print("Training and predicting models...")
params = {
    'num_leaves': 33,
    'objective': 'regression',
    'min_data_in_leaf': 250,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'l2',
    'num_threads': 30
}

MAX_ROUNDS = 3000
val_pred = []
test_pred = []
cate_vars = []
for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=pd.concat([items["perishable"]] * 6) * 0.25 + 1
    )
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        weight=items["perishable"] * 0.25 + 1,
        categorical_feature=cate_vars)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=125, verbose_eval=500
    )
#    print("\n".join(("%s: %.2f" % x) for x in sorted(
#        zip(X_train.columns, bst.feature_importance("gain")),
#        key=lambda x: x[1], reverse=True
#    )))
    val_pred.append(bst.predict(
        X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    test_pred.append(bst.predict(
        X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))

print("Validation mse:", mean_squared_error(
    y_val, np.array(val_pred).transpose()))

print("Making submission...")
y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv(inDir+'/submissions/Prav_lgbm04.csv', float_format='%.4f', index=None)
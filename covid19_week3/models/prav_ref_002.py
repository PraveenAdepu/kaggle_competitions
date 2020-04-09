# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


"""
started from this public kernal - https://www.kaggle.com/aerdem4/covid-19-basic-model-not-leaky
thanks to author for his kindness to share with community
"""
"""
Target : Learn different modeling techniques, enjoy and have fun 
"""

inDir = r"C:\Users\SriPrav\Documents\R\kaggle_competitions\covid19_week3"
train = pd.read_csv(inDir+"\\data\\train.csv")

"""
parameters
"""
train_first_date = "2020-01-22"
cv_date = "2020-03-26"
test_date = "2020-03-26"
cv_days = 8
test_days = 43

loc_group = ["Province_State", "Country_Region"]
TARGETS = ["ConfirmedCases", "Fatalities"]

"""
Helper functions
"""
def preprocess(df):
    df["Date"] = df["Date"].astype("datetime64[ms]")
    for col in loc_group:
        df[col].fillna("none", inplace=True)
    return df

def log_transform(df, TARGETS):
    for col in TARGETS:
        df[col] = np.log1p(df[col])
    return df

def exp_transform(df, TARGETS):
    for col in TARGETS:
        df[col] = np.expm1(df["pred_{}".format(col)])
    return df

def lag_features(df, TARGETS):
    for col in TARGETS:
        df["prev_{}".format(col)] = df.groupby(loc_group)[col].shift(1)
    return df

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate(df):
    error = 0
    for col in TARGETS:
        error += rmse(df[col].values, df["pred_{}".format(col)].values)
    return np.round(error/len(TARGETS), 5)

def predict(test_df, first_day, num_days, val=False):

    y_pred = np.clip(model.predict(test_df.loc[test_df["Date"] == first_day][features]), None, 16)

    for i, col in enumerate(TARGETS):
        test_df["pred_{}".format(col)] = 0
        test_df.loc[test_df["Date"] == first_day, "pred_{}".format(col)] = y_pred[:, i]

    if val:
        print(first_day, evaluate(test_df[test_df["Date"] == first_day]))

    for d in range(1, num_days):
        y_pred = np.clip(model.predict(y_pred), None, 16)
        date = first_day + timedelta(days=d)

        for i, col in enumerate(TARGETS):
            test_df.loc[test_df["Date"] == date, "pred_{}".format(col)] = y_pred[:, i]

        if val:
            print(date, evaluate(test_df[test_df["Date"] == date]))
        
    return test_df


"""
validation starts
"""

train = preprocess(train)

train = log_transform(train, TARGETS)    
train = lag_features(train, TARGETS)

train = train[train["Date"]>train_first_date].copy()

x_build = train[train["Date"]<cv_date].copy()
x_valid = train[train["Date"]>=cv_date].copy()

x_valid.loc[x_valid["Date"]>cv_date, ['prev_ConfirmedCases', 'prev_Fatalities']] = 0 , 0

model = Pipeline([('linear', LinearRegression())])
features = ["prev_{}".format(col) for col in TARGETS]

model.fit(x_build[features], x_build[TARGETS])

[mean_squared_error(x_build[TARGETS[i]], model.predict(x_build[features])[:, i]) for i in range(len(TARGETS))]
    
validation = predict(x_valid, pd.to_datetime(cv_date), cv_days, val=True)
evaluate(validation)
"""
validation completed
"""

train = pd.read_csv(inDir+"\\data\\train.csv")
test = pd.read_csv(inDir+"\\data\\test.csv")

train = preprocess(train)
test = preprocess(test)

train = log_transform(train, TARGETS)  

x_build = train[train["Date"]<cv_date].copy()

test_full = x_build.append(test, sort=False)

test_full = lag_features(test_full, TARGETS)
test_full = test_full[test_full["Date"]>train_first_date].copy()

x_build = test_full[test_full["Date"]<cv_date].copy()
x_test = test_full[test_full["Date"]>=cv_date].copy()

x_test.loc[x_test["Date"]>test_date, ['prev_ConfirmedCases', 'prev_Fatalities']] = 0 , 0

x_test = predict(x_test, pd.to_datetime(test_date), test_days)

x_test["ForecastId"] = x_test["ForecastId"].astype(np.int16)

x_test = exp_transform(x_test, TARGETS)

x_test.head(10)

sub_columns = ["ForecastId"]+TARGETS


x_test.to_csv("submission.csv", index=False, columns=sub_columns)








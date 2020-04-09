# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from datetime import timedelta
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

"""
started from this public kernal - https://www.kaggle.com/aerdem4/covid-19-basic-model-not-leaky
thanks to author for his kindness to share with community
"""
"""
Target : Learn different modeling techniques and have fun 
"""

inDir = r"C:\Users\SriPrav\Documents\R\kaggle_competitions\covid19_week3"
train = pd.read_csv(inDir+"\\data\\train.csv")
test = pd.read_csv(inDir+"\\data\\test.csv")
# submission = pd.read_csv(inDir+"\\data\\submission.csv")

"""
parameters
"""

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

def evaluation_daily(df):
    for date in df["Date"].unique():
        print(date, evaluate(df[df["Date"] == date]))
        
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
test = preprocess(test)

# parameters
train_first_date = "2020-01-22"
public_start_date = "2020-03-26"

cv_days = (train["Date"].max() - test["Date"].min()).days + 1
private_test_days = (test["Date"].max() - test["Date"].min()).days + 1 - cv_days
 
train = log_transform(train, TARGETS) 

x_build = train[train["Date"]<public_start_date].copy()
x_valid = train[train["Date"]>=public_start_date].copy()

public_test = test[test["Date"] <= train["Date"].max()].copy()
private_test = test[test["Date"] > train["Date"].max()].copy()

public_train = x_build.append(public_test, sort=False)
private_train = train.append(private_test, sort=False)

# public train
# use cross validation, forget about LB score
public_train = lag_features(public_train, TARGETS)

public_train_build = public_train[public_train["Date"]<public_start_date].copy()
public_train_valid = public_train[public_train["Date"]>=public_start_date].copy()

public_train_build = public_train_build[public_train_build["Date"]>train_first_date].copy()
public_train_valid.loc[public_train_valid["Date"]>public_start_date, ['prev_ConfirmedCases', 'prev_Fatalities']] = 0 , 0

#model = Pipeline([('linear', LinearRegression())])

#model = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),('linear', LinearRegression())]) # 0.5119
model = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),('Ridge', Ridge())]) # 0.5116
features = ["prev_{}".format(col) for col in TARGETS]

model.fit(public_train_build[features], public_train_build[TARGETS])

[mean_squared_error(public_train_build[TARGETS[i]], model.predict(public_train_build[features])[:, i]) for i in range(len(TARGETS))]

public_train_valid = predict(public_train_valid, pd.to_datetime(public_start_date), cv_days, val=False)

x_valid = pd.merge(x_valid, public_train_valid[["Province_State","Country_Region","Date","pred_ConfirmedCases","pred_Fatalities"]], on = ["Province_State","Country_Region","Date"], how="inner")
  
evaluation_daily(x_valid)
print(evaluate(x_valid))

# private train
# use all train data for better forecast at private LB date range

private_train = lag_features(private_train, TARGETS)

private_train_build = private_train[private_train["Date"]<= train["Date"].max()].copy()
private_train_valid = private_train[private_train["Date"]> train["Date"].max()].copy()

private_train_build = private_train_build[private_train_build["Date"]>train_first_date].copy()
private_train_valid.loc[private_train_valid["Date"]>private_train_valid["Date"].min(), ['prev_ConfirmedCases', 'prev_Fatalities']] = 0 , 0

model = Pipeline([('linear', LinearRegression())])
features = ["prev_{}".format(col) for col in TARGETS]

model.fit(private_train_build[features], private_train_build[TARGETS])

private_train_valid = predict(private_train_valid, private_train_valid["Date"].min(), private_train_valid["Date"].nunique(), val=False)


test_full = public_train_valid.append(private_train_valid, sort=False)


test_full["ForecastId"] = test_full["ForecastId"].astype(np.int16)
test_full = exp_transform(test_full, TARGETS)
test_full.head(10)

sub_columns = ["ForecastId"]+TARGETS
test_full.to_csv("submission.csv", index=False, columns=sub_columns)


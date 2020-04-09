# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import timedelta
import pystan
import datetime
from sklearn.metrics import mean_squared_error

"""
Target : Learn different modeling techniques and have fun 
         Most of the code learned on the way, thanks to open source community
         
"""

inDir = r"C:\Users\SriPrav\Documents\R\kaggle_competitions\covid19_week3"
train = pd.read_csv(inDir+"\\data\\train.csv")
test = pd.read_csv(inDir+"\\data\\test.csv")
submission = pd.read_csv(inDir+"\\data\\submission.csv")

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
    df["Country_Region_State"] = df["Country_Region"]+"-"+df["Province_State"]
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
        error += rmse(np.log1p(df[col].values), np.log1p(df["pred_{}".format(col)].values))
    return np.round(error/len(TARGETS), 5)

def evaluation_daily(df):
    for date in df["Date"].unique():
        print(date, evaluate(df[df["Date"] == date]))
"""
Stan model
            Still lot more to learn in this space
"""
model_logistic = """
    data {
        int<lower=1> n;
        int<lower=1> n_pred;
        vector[n] y;
        vector[n] t;
        vector[n_pred] t_pred;
    }
    parameters {
        real<lower=0> alpha;
        real<lower=0> beta;
        real<lower=0> t0;
        real<lower=0> sigma; 
    }
    model {
    alpha~normal(1,1);
    beta~normal(1,1);
    t0~normal(10,10);
    y ~ normal(alpha ./ (1 + exp(-(beta*(t-t0)))), sigma);
    }
    generated quantities {
      vector[n_pred] pred;
      for (i in 1:n_pred)
      pred[i] = normal_rng(alpha / (1 + exp(-(beta*(t_pred[i]-t0)))),sigma);
    }
    """

stan_model= pystan.StanModel(model_code=model_logistic)

def model_forecast(stan_model,df,time_var_norm_coef,target_field_norm_coef,n_days_predict):
    print ('Time Series size:',df.shape[0])
    n_train=df.shape[0]
    maxdate=df.date.max()
    for i in np.arange(1,n_days_predict+1):
            df=df.append(pd.DataFrame({'date':\
                [maxdate+datetime.timedelta(days=int(i))]}))
    df['t']=time_var_norm_coef*np.arange(df.shape[0])
    df.y=target_field_norm_coef*df.y
    df.set_index('date',inplace=True)
    data_df = {'n': n_train,
               'n_pred':df.shape[0],
               'y': df.iloc[:n_train,:].y.values,
               't':df.iloc[:n_train,:].t.values,
               't_pred':df.t.values}
    fit=stan_model.sampling(data=data_df, iter=5000, chains=3)
    fit_samples = fit.extract(permuted=True)
    pred=fit_samples['pred']
    df['predictions']=pred.mean(axis=0)
    #df['predictions_quantile']=(pd.DataFrame(pred).quantile(q=0.95,axis=0).values-pd.DataFrame(pred).quantile(q=0.05,axis=0).values)/2
    df.y=df.y/target_field_norm_coef
    df.predictions=df.predictions/target_field_norm_coef
    df = df.tail(n_days_predict)
    df['Date'] = df.index
    return df
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

public_train_build = public_train[public_train["Date"]<public_start_date].copy()
public_train_valid = public_train[public_train["Date"]>=public_start_date].copy()

    
# Normalization coefficients
target_field_norm_coef=1/100000
train_days = public_train_build["Date"].nunique()
time_var_norm_coef=1/train_days
n_days_predict = cv_days

validation_ConfirmedCases = pd.DataFrame()
validation_Fatalities = pd.DataFrame()

row = 0

for crs in public_train_build['Country_Region_State'].unique():
    print(crs)
    print(row)
    data = public_train_build[public_train_build['Country_Region_State']==crs]
    
    df = data[['Date','ConfirmedCases']].copy()
    df.reset_index(drop=True, inplace=True)
    df.columns =['date','y'] 
    df = model_forecast(stan_model,df,time_var_norm_coef,target_field_norm_coef,n_days_predict)     
    df.rename(columns={'predictions': 'pred_ConfirmedCases'}, inplace=True)
    df['Country_Region_State'] = crs    
    
    validation_ConfirmedCases = validation_ConfirmedCases.append(df[['Country_Region_State','Date','pred_ConfirmedCases']])  

    df = data[['Date','Fatalities']].copy()
    df.reset_index(drop=True, inplace=True)
    df.columns =['date','y']      
    df = model_forecast(stan_model,df,time_var_norm_coef,target_field_norm_coef,n_days_predict)  
    df.rename(columns={'predictions': 'pred_Fatalities'}, inplace=True)
    df['Country_Region_State'] = crs
    
    validation_Fatalities = validation_Fatalities.append(df[['Country_Region_State','Date','pred_Fatalities']])    

    row = row + 1

validations = pd.merge(validation_ConfirmedCases, validation_Fatalities, on=['Country_Region_State','Date'], how="inner")

x_valid = pd.merge(x_valid, validations[["Country_Region_State","Date","pred_ConfirmedCases","pred_Fatalities"]], on = ["Country_Region_State","Date"], how="inner")

evaluation_daily(x_valid)
print(evaluate(x_valid))


# private train
# use all train data for better forecast at private LB date range

private_train_build = private_train[private_train["Date"]<= train["Date"].max()].copy()
private_train_valid = private_train[private_train["Date"]> train["Date"].max()].copy()

test_ConfirmedCases = pd.DataFrame()
test_Fatalities = pd.DataFrame()

# Normalization coefficients
target_field_norm_coef=1/100000
train_days = private_train_build["Date"].nunique()
time_var_norm_coef=1/train_days
n_days_predict = private_test_days
row = 0

for crs in private_train_build['Country_Region_State'].unique():
    print(crs)
    print(row)
#    csr = 'Afghanistan-none'
    data = private_train_build[private_train_build['Country_Region_State']==crs]
            
    df = data[['Date','ConfirmedCases']].copy()
    df.reset_index(drop=True, inplace=True)
    df.columns =['date','y'] 
    df = model_forecast(stan_model,df,time_var_norm_coef,target_field_norm_coef,n_days_predict)     
    df.rename(columns={'predictions': 'pred_ConfirmedCases'}, inplace=True)
    df['Country_Region_State'] = crs      
    
    test_ConfirmedCases = test_ConfirmedCases.append(df[['Country_Region_State','Date','pred_ConfirmedCases']])  

    df = data[['Date','Fatalities']].copy()
    df.reset_index(drop=True, inplace=True)
    df.columns =['date','y']      
    df = model_forecast(stan_model,df,time_var_norm_coef,target_field_norm_coef,n_days_predict)  
    df.rename(columns={'predictions': 'pred_Fatalities'}, inplace=True)
    df['Country_Region_State'] = crs
    
    test_Fatalities = test_Fatalities.append(df[['Country_Region_State','Date','pred_Fatalities']])     

    row = row + 1

testing = pd.merge(test_ConfirmedCases, test_Fatalities, on=['Country_Region_State','Date'], how="inner")

test_full = validations.append(testing, sort=False)

test_full= pd.merge(test_full, test[["ForecastId","Date","Country_Region_State"]]
, on=['Country_Region_State','Date'], how="inner")

test_full["ForecastId"] = test_full["ForecastId"].astype(np.int16)
test_full = exp_transform(test_full, TARGETS)
test_full.head(10)

sub_columns = ["ForecastId"]+TARGETS
test_full.to_csv("submission.csv", index=False, columns=sub_columns)


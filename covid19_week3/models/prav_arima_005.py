# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
#public_train = lag_features(public_train, TARGETS)

public_train_build = public_train[public_train["Date"]<public_start_date].copy()
public_train_valid = public_train[public_train["Date"]>=public_start_date].copy()

#public_train_build = public_train_build[public_train_build["Date"]>train_first_date].copy()
#public_train_valid.loc[public_train_valid["Date"]>public_start_date, ['prev_ConfirmedCases', 'prev_Fatalities']] = 0 , 0

arima_forecast_dates = pd.DataFrame(public_train_valid["Date"].unique())
arima_forecast_dates.columns = ["Date"]

validation_ConfirmedCases = pd.DataFrame()
validation_Fatalities = pd.DataFrame()

row = 0

for i in public_train_build['Country_Region_State'].unique():
    print(i)
    print(row)
#    i = 'Angola-none'
    data = public_train_build[public_train_build['Country_Region_State']==i]
    data_c = data.copy()
#    if data_c.loc[data_c['ConfirmedCases']>0,:].shape[0]>0:
#        data_c=data_c.loc[data_c['ConfirmedCases']>0,:]
    #data_arima=data_c['ConfirmedCases'].to_list()
    data_arima=data_c['ConfirmedCases'].astype('int32').to_list()
    if len(data_arima)==1:
        data_arima.append(data_arima[0])
    #model = ARIMA(data_arima, order=(0,1,1))
    model = SARIMAX(data_arima, order=(1,1,0), seasonal_order=(1,1,0,12), measurement_error=True)
    model_fit = model.fit(disp=0)
    #forecast=pd.DataFrame(model_fit.forecast(steps=cv_days)[0])
    forecast=pd.DataFrame(model_fit.predict(steps=cv_days)[0])
    forecast.columns = ["pred_ConfirmedCases"]    
    forecast = pd.concat([forecast, arima_forecast_dates], axis=1)         
    forecast['Country_Region_State'] = i    
    
    validation_ConfirmedCases = validation_ConfirmedCases.append(forecast[['Country_Region_State','Date','pred_ConfirmedCases']])  
    
#    if data.loc[data['Fatalities']>0,:].shape[0]>0:
#        data=data.loc[data['Fatalities']>0,:]
    #data_arima = data['Fatalities'].to_list()
    data_arima = data['Fatalities'].astype('int32').to_list()
    
    if len(data_arima)==1:
        data_arima.append(data_arima[0])
    #model = ARIMA(data_arima, order=(0,1,1))
    model = SARIMAX(data_arima, order=(1,1,0), seasonal_order=(1,1,0,12), measurement_error=True)
    model_fit = model.fit(disp=0)
    forecast=pd.DataFrame(model_fit.forecast(steps=cv_days)[0])
    forecast.columns = ["pred_Fatalities"]
    forecast = pd.concat([forecast, arima_forecast_dates], axis=1)         
    forecast['Country_Region_State'] = i      
       
    validation_Fatalities = validation_Fatalities.append(forecast[['Country_Region_State','Date','pred_Fatalities']])     

    row = row + 1

validations = pd.merge(validation_ConfirmedCases, validation_Fatalities, on=['Country_Region_State','Date'], how="inner")

x_valid = pd.merge(x_valid, validations[["Country_Region_State","Date","pred_ConfirmedCases","pred_Fatalities"]], on = ["Country_Region_State","Date"], how="inner")

evaluation_daily(x_valid)
print(evaluate(x_valid))


# private train
# use all train data for better forecast at private LB date range

#private_train = lag_features(private_train, TARGETS)

private_train_build = private_train[private_train["Date"]<= train["Date"].max()].copy()
private_train_valid = private_train[private_train["Date"]> train["Date"].max()].copy()

#private_train_build = private_train_build[private_train_build["Date"]>train_first_date].copy()
#private_train_valid.loc[private_train_valid["Date"]>private_train_valid["Date"].min(), ['prev_ConfirmedCases', 'prev_Fatalities']] = 0 , 0


from fbprophet import Prophet

test_ConfirmedCases = pd.DataFrame()
test_Fatalities = pd.DataFrame()

row = 0

for i in private_train_build['Country_Region_State'].unique():
    print(i)
    print(row)
#    i = 'Afghanistan-none'
    data = private_train_build[private_train_build['Country_Region_State']==i]
    
    data_prophet = data[['Date','ConfirmedCases']].copy()
    data_prophet.reset_index(drop=True, inplace=True)
    data_prophet.columns =['ds','y']          
    m = Prophet(daily_seasonality=True)
    m.fit(data_prophet) 
    future = m.make_future_dataframe(periods=private_train_valid["Date"].nunique(), freq='D')
    forecast = m.predict(future)
    forecast = forecast.tail(cv_days)
    forecast.reset_index(drop=True,inplace=True)
    
    forecast = forecast[['ds','yhat']]
    forecast.columns = ['Date','pred_ConfirmedCases']        
    forecast['Country_Region_State'] = i    
    
    test_ConfirmedCases = test_ConfirmedCases.append(forecast[['Country_Region_State','Date','pred_ConfirmedCases']])  

    data_prophet = data[['Date','Fatalities']].copy()
    data_prophet.reset_index(drop=True, inplace=True)
    data_prophet.columns =['ds','y']          
    m = Prophet(daily_seasonality=True)
    m.fit(data_prophet) 
    future = m.make_future_dataframe(periods=private_train_valid["Date"].nunique(), freq='D')
    forecast = m.predict(future)
    forecast = forecast.tail(cv_days)
    forecast.reset_index(drop=True,inplace=True)
    
    forecast = forecast[['ds','yhat']]
    forecast.columns = ['Date','pred_Fatalities']        
    forecast['Country_Region_State'] = i    
    
    test_Fatalities = test_Fatalities.append(forecast[['Country_Region_State','Date','pred_Fatalities']])     

    row = row + 1

testing = pd.merge(test_ConfirmedCases, test_Fatalities, on=['Country_Region_State','Date'], how="inner")

test_full = validations.append(testing, sort=False)

evaluation_daily(x_valid)
print(evaluate(x_valid))












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










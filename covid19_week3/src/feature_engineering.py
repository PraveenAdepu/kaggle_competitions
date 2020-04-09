"""
feature_engineering.py

Notes:
    feature_engineering is partly automated and partly use case specific
    use case requirements
    feature store - best practice is to use feature store from data engineering pipeline
        
    re-usable percentage - 30%       
                
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import time

"""
Prav
    file paths to be sourced from config file
"""

def dataset_extract_features_from_date(dataset,date_feature):
    """
    Extract common date features from date
        
    Args:
        dataset: pandas dataframe to derive new features
        date_feature: date feature name from dataset
        
    
    Returns:
        dataset: pandas dataframe with all original features
                 new date features from this logic
    
    Raises:
        None
    """       
    dataset['dayofmonth'] = dataset[date_feature].dt.day
    dataset['dayofyear'] = dataset[date_feature].dt.dayofyear 
    dataset['dayofweek'] = dataset[date_feature].dt.dayofweek
    dataset['month'] = dataset[date_feature].dt.month
    dataset['year'] = dataset[date_feature].dt.year
    dataset['weekofyear'] = dataset[date_feature].dt.weekofyear
    dataset['is_month_start'] = (dataset[date_feature].dt.is_month_start).astype(int)
    dataset['is_month_end'] = (dataset[date_feature].dt.is_month_end).astype(int)
    return dataset


'''
define dataset, date_feature

date_feature = "date"
dataset = dataset_extract_features_from_date(dataset, date_feature)

'''  

def create_sales_agg_monthwise_features(df, gpby_cols, target_col, agg_funcs):
    '''
    Creates various sales agg features with given agg functions  
    '''
    gpby = df.groupby(gpby_cols)
    newdf = df[gpby_cols].drop_duplicates().reset_index(drop=True)
    for agg_name, agg_func in agg_funcs.items():
        aggdf = gpby[target_col].agg(agg_func).reset_index()
        aggdf.rename(columns={target_col:target_col+'_'+agg_name}, inplace=True)
        newdf = newdf.merge(aggdf, on=gpby_cols, how='left')
    return newdf

# Creating sales lag features
def extract_lag_features(df, gpby_cols, target_col, lags):
    gpby = df.groupby(gpby_cols)
    for i in lags:
        df['_'.join([target_col, 'lag', str(i)])] = \
                gpby[target_col].shift(i).values #+ np.random.normal(scale=1.6, size=(len(df),))
    return df

# Creating sales rolling mean features
"""  
def extract_rmean_features(df, gpby_cols, target_col, windows, min_periods=2, 
                             shift=1, win_type=None):
    gpby = df.groupby(gpby_cols)
    for w in windows:
        df['_'.join([target_col, 'rmean', str(w)])] = \
            gpby[target_col].shift(shift).rolling(window=w, 
                                                  min_periods=min_periods,
                                                  win_type=win_type).mean().values #+\
            #np.random.normal(scale=1.6, size=(len(df),))
    return df
"""
def extract_rmean_features(df, gpby_cols, target_col, windows, min_periods=2, 
                             shift=[1], win_type=None):
    gpby = df.groupby(gpby_cols)
    for s in shift:
        for w in windows:
            df['_'.join([target_col, 'shift',str(s) ,'rmean', str(w)])] = \
                gpby[target_col].shift(s).rolling(window=w, 
                                                      min_periods=min_periods,
                                                      win_type=win_type).mean().values #+\
                #np.random.normal(scale=1.6, size=(len(df),))
    return df

# Creating sales rolling median features
"""
def extract_rmed_features(df, gpby_cols, target_col, windows, min_periods=2, 
                            shift=1, win_type=None):
    gpby = df.groupby(gpby_cols)
    for w in windows:
        df['_'.join([target_col, 'rmed', str(w)])] = \
            gpby[target_col].shift(shift).rolling(window=w, 
                                                  min_periods=min_periods,
                                                  win_type=win_type).median().values #+\
            #np.random.normal(scale=1.6, size=(len(df),))
    return df
"""
def extract_rmed_features(df, gpby_cols, target_col, windows, min_periods=2, 
                            shift=[1], win_type=None):
    gpby = df.groupby(gpby_cols)
    for s in shift:
        for w in windows:
            df['_'.join([target_col, 'shift',str(s) ,'rmed', str(w)])] = \
                gpby[target_col].shift(s).rolling(w).median().values #+\
                #np.random.normal(scale=1.6, size=(len(df),))
    return df

def extract_rstd_features(df, gpby_cols, target_col, windows, min_periods=2, 
                            shift=[1], win_type=None):
    gpby = df.groupby(gpby_cols)
    for s in shift:
        for w in windows:
            df['_'.join([target_col, 'shift',str(s) ,'rstd', str(w)])] = \
                gpby[target_col].shift(s).rolling(w).std().values #+\
                #np.random.normal(scale=1.6, size=(len(df),))
    return df

def extract_rmax_features(df, gpby_cols, target_col, windows, min_periods=2, 
                            shift=[1], win_type=None):
    gpby = df.groupby(gpby_cols)
    for s in shift:
        for w in windows:
            df['_'.join([target_col, 'shift',str(s) ,'rmax', str(w)])] = \
                gpby[target_col].shift(s).rolling(w).max().values #+\
                #np.random.normal(scale=1.6, size=(len(df),))
    return df

# Creating sales exponentially weighted mean features
def extract_ewm_features(df, gpby_cols, target_col, alpha=[0.9], shift=[1]):
    gpby = df.groupby(gpby_cols)
    for a in alpha:
        for s in shift:
            df['_'.join([target_col, 'lag', str(s), 'ewm', str(a)])] = \
                gpby[target_col].shift(s).ewm(alpha=a).mean().values
    return df
# Creating sales lag features
def extract_lag_features_by_day(df, gpby_cols, target_col, lags):
    gpby = df.groupby(gpby_cols)
    for i in lags:
        df['_'.join([target_col, 'days_lag', str(i)])] = \
                gpby[target_col].shift(i).values #+ np.random.normal(scale=1.6, size=(len(df),))
    return df

def extract_rmean_features_by_day(df, gpby_cols, target_col, windows, min_periods=2, 
                             shift=[1], win_type=None):
    gpby = df.groupby(gpby_cols)
    for s in shift:
        for w in windows:
            df['_'.join([target_col, 'days_shift',str(s) ,'rmean', str(w)])] = \
                gpby[target_col].shift(s).rolling(window=w, 
                                                      min_periods=min_periods,
                                                      win_type=win_type).mean().values
                
    return df

def extract_rmed_features_by_day(df, gpby_cols, target_col, windows, min_periods=2, 
                            shift=[1], win_type=None):
    gpby = df.groupby(gpby_cols)
    for s in shift:
        for w in windows:
            df['_'.join([target_col, 'days_shift',str(s) ,'rmed', str(w)])] = \
                gpby[target_col].shift(s).rolling(w).median().values                
    return df

def extract_rmax_features_by_day(df, gpby_cols, target_col, windows, min_periods=2, 
                            shift=[1], win_type=None):
    gpby = df.groupby(gpby_cols)
    for s in shift:
        for w in windows:
            df['_'.join([target_col, 'days_shift',str(s) ,'rmax', str(w)])] = \
                gpby[target_col].shift(s).rolling(w).max().values 
    return df
def extract_rmin_features_by_day(df, gpby_cols, target_col, windows, min_periods=2, 
                            shift=[1], win_type=None):
    gpby = df.groupby(gpby_cols)
    for s in shift:
        for w in windows:
            df['_'.join([target_col, 'days_shift',str(s) ,'rmin', str(w)])] = \
                gpby[target_col].shift(s).rolling(w).min().values 
    return df

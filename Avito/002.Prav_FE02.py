# -*- coding: utf-8 -*-
"""
Created on Tue May 15 23:17:43 2018

@author: SriPrav
"""

import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
#from matplotlib_venn import venn2, venn2_circles
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import scipy
import lightgbm as lgb

sns.set()
%matplotlib inline

inDir = 'C:/Users/SriPrav/Documents/R/48Avito'

used_cols = ['item_id', 'user_id']

train = pd.read_csv(inDir +'/input/train.csv', usecols=used_cols)
train_active = pd.read_csv(inDir +'/input/train_active.csv', usecols=used_cols)

test = pd.read_csv(inDir +'/input/test.csv', usecols=used_cols)
test_active = pd.read_csv(inDir +'/input/test_active.csv', usecols=used_cols)

train_periods = pd.read_csv(inDir +'/input/periods_train.csv', parse_dates=['date_from', 'date_to'])
test_periods = pd.read_csv(inDir +'/input/periods_test.csv', parse_dates=['date_from', 'date_to'])

train.head()

all_samples = pd.concat([
    train,
    train_active,
    test,
    test_active
]).reset_index(drop=True)
all_samples.drop_duplicates(['item_id'], inplace=True)

del train_active
del test_active
gc.collect()

all_periods = pd.concat([
    train_periods,
    test_periods
])

del train_periods
del test_periods
gc.collect()

all_periods.head()

all_periods['days_up'] = (all_periods['date_to'] - all_periods['date_from']).dt.days

gp = all_periods.groupby(['item_id'])[['days_up']]

gp_df = pd.DataFrame()
gp_df['days_up_sum'] = gp.sum()['days_up']
gp_df['days_up_max'] = gp.max()['days_up']   # Prav
gp_df['days_up_min'] = gp.min()['days_up']   # Prav
gp_df['days_up_mean'] = gp.mean()['days_up'] # Prav
gp_df['times_put_up'] = gp.count()['days_up']
gp_df.reset_index(inplace=True)
gp_df.rename(index=str, columns={'index': 'item_id'})

gp_df.head()

all_periods.drop_duplicates(['item_id'], inplace=True)
all_periods = all_periods.merge(gp_df, on='item_id', how='left')
all_periods.head()

del gp
del gp_df
gc.collect()

all_periods = all_periods.merge(all_samples, on='item_id', how='left')
all_periods.head()


#gp = all_periods.groupby(['user_id'])[['days_up_sum', 'times_put_up']].mean().reset_index() \
#    .rename(index=str, columns={
#        'days_up_sum': 'avg_days_up_user',
#        'times_put_up': 'avg_times_up_user'
#    })
#gp.head()

###########################################################################################################
gp_mean = all_periods.groupby(['user_id'])[['days_up_sum', 'times_put_up','days_up_max','days_up_min','days_up_mean']].mean().reset_index().rename(index=str, columns={
        'days_up_sum': 'days_up_sum_user_mean',
        'times_put_up': 'times_put_up_user_mean',
        'days_up_max': 'days_up_max_user_mean',
        'days_up_min': 'days_up_min_user_mean',
        'days_up_mean': 'days_up_mean_user_mean'
    })
    
gp_mean.head()

gp_max = all_periods.groupby(['user_id'])[['days_up_sum', 'times_put_up','days_up_max','days_up_min','days_up_mean']].max().reset_index().rename(index=str, columns={
        'days_up_sum': 'days_up_sum_user_max',
        'times_put_up': 'times_put_up_user_max',
        'days_up_max': 'days_up_max_user_max',
        'days_up_min': 'days_up_min_user_max',
        'days_up_mean': 'days_up_mean_user_max'
    })
    
gp_max.head()

gp_min = all_periods.groupby(['user_id'])[['days_up_sum', 'times_put_up','days_up_max','days_up_min','days_up_mean']].min().reset_index().rename(index=str, columns={
        'days_up_sum': 'days_up_sum_user_min',
        'times_put_up': 'times_put_up_user_min',
        'days_up_max': 'days_up_max_user_min',
        'days_up_min': 'days_up_min_user_min',
        'days_up_mean': 'days_up_mean_user_min'
    })
    
gp_min.head()

gp = gp_mean.merge(gp_max, on='user_id', how='left')

gp = gp.merge(gp_min, on='user_id', how='left')

##########################################################################################################
n_user_items = all_samples.groupby(['user_id'])[['item_id']].count().reset_index() \
    .rename(index=str, columns={
        'item_id': 'n_user_items'
    })
gp = gp.merge(n_user_items, on='user_id', how='outer') # left FE_02

gp.head()

gp.to_csv(inDir+'/input/traintest_FE_022.csv', index=False)

del all_samples
del all_periods
del train
del test

gc.collect()

#####################################################




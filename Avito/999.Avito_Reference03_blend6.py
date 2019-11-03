# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:30:27 2018

@author: SriPrav
"""

import pandas as pd
import numpy as np
import seaborn as sns 

b1 = pd.read_csv('../input/bow-meta-text-and-dense-features-lgbm-clone2/lgsub.csv').rename(columns={'deal_probability':'dp1'})
b2 = pd.read_csv('../input/xgb-text2vec-tfidf-0-2243-0-2241-ori/xgb_tfidf0.21907.csv').rename(columns={'deal_probability':'dp2'})
b3 = pd.read_csv('../input/aggregated-features-v2-lb-0p2232/submission v2 LB 0p2232.csv').rename(columns={'deal_probability':'dp3'})

b1 = pd.merge(b1, b2, how='left', on='item_id')
b1 = pd.merge(b1, b3, how='left', on='item_id')

b1['deal_probability'] = (b1['dp1'] * 0.20) + (b1['dp2'] * 0.30) + (b1['dp3'] * 0.50)   # v08 LB 0.2216

b1[['item_id','deal_probability']].to_csv('blend 06.csv', index=False)

print('correlation between models outputs')

blend_results = pd.concat([ b1['dp1'], b1['dp2'], b1['dp3'] ],axis=1)

print(blend_results.corr())
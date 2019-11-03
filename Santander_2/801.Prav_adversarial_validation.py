# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 19:32:18 2018

@author: SriPrav
"""

#!/usr/bin/env python

"train a classifier to distinguish between train and test"
"save train examples in order of similarity to test (ascending)"

import numpy as np
import pandas as pd

from sklearn import cross_validation as CV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy

from time import ctime

inDir = 'C:/Users/SriPrav/Documents/R/50Santander'

train = pd.read_csv(inDir+'/input/train.csv.zip', compression='zip', header=0, sep=',')

test = pd.read_csv(inDir+'/input/test.csv.zip', compression='zip', header=0, sep=',')




test['target'] = 0		# dummy for preserving column order when concatenating


train['is_test'] = 0
test['is_test'] = 1

orig_train = train.copy()
#assert( np.all( orig_train.columns == test.columns ))

train_all = pd.concat(( orig_train, test ))
train_all.reset_index( inplace = True, drop = True )

x = train_all.drop( [ 'ID','is_test', 'target' ], axis = 1 )
y = train_all.is_test

#

print("cross-validating...")

n_estimators = 100
clf = RF( n_estimators = n_estimators, n_jobs = -1 )

predictions = np.zeros( y.shape )

cv = CV.StratifiedKFold( y, n_folds = 5, shuffle = True, random_state = 20180512 )

for f, ( train_i, test_i ) in enumerate( cv ):

	print("# fold {}, {}".format( f + 1, ctime()))

	x_train = x.iloc[train_i]
	x_test = x.iloc[test_i]
	y_train = y.iloc[train_i]
	y_test = y.iloc[test_i]
	
	clf.fit( x_train, y_train )	

	p = clf.predict_proba( x_test )[:,1]
	
	auc = AUC( y_test, p )
	print("# AUC: {:.2%}\n".format( auc )	)
	
	predictions[ test_i ] = p

# fold 1, Mon Jun 25 19:39:24 2018
# AUC: 82.71%

# fold 2, Mon Jun 25 19:39:48 2018
# AUC: 82.99%

# fold 3, Mon Jun 25 19:40:10 2018
# AUC: 82.20%

# fold 4, Mon Jun 25 19:40:31 2018
# AUC: 83.54%

# fold 5, Mon Jun 25 19:40:53 2018
# AUC: 83.00%

train_all['p'] = predictions
	
i = predictions.argsort()
train_all_sorted = train_all.iloc[i]

"""
print "predictions distribution for test"
train_sorted.loc[ train_sorted.is_test == 1, 'p' ].hist()
p_test_mean = train_sorted.loc[ train_sorted.is_test == 1, 'p' ].mean()
p_test_std = train_sorted.loc[ train_sorted.is_test == 1, 'p' ].std()
print "# mean: {}, std: {}".format( p_test_mean, p_test_std )
# mean: 0.404749669062, std: 0.109116404564
"""

train_all_sorted = train_all_sorted.loc[ train_all_sorted.is_test == 0 ]
#assert( train_all_sorted.target.sum() == orig_train.target.sum())

"""
print "predictions distribution for train"
p_train_mean = train_sorted.p.mean()
p_train_std = train_sorted.p.std()
print "# mean: {}, std: {}".format( p_train_mean, p_train_std )
# mean: 0.293768613822, std: 0.113601453932
"""

train_all_sorted.drop( 'is_test', axis = 1, inplace = True )


validations = train_all_sorted[["ID","p"]].copy()

validations.head()

validations = validations.sort_values(by='p', ascending=False)

validations['CVindices'] = np.where(validations['p']>=0.9, 1, 2)

validations.to_csv( inDir+'/input/Prav_adversarial_validation.csv', index = False )
import numpy as np
import dicom
import glob
from matplotlib import pyplot as plt
import os
import cv2
import mxnet as mx
import pandas as pd
from sklearn import cross_validation
import xgboost as xgb
from scipy.stats import gmean
import numpy as np
import keras as k
import keras.layers as l


stage1_labels = pd.read_csv('C:/Users/SriPrav/Documents/R/19DSB2017/input/sources/stage1_labels.csv')
Prav_CVindices_5folds = pd.read_csv('C:/Users/SriPrav/Documents/R/19DSB2017/CVSchema/Prav_CVindices_5folds.csv')
stage1_labels_CVindices = pd.merge(stage1_labels, Prav_CVindices_5folds, on=['id', 'cancer'], how='left')
train_csv_table = pd.read_csv('C:/Users/SriPrav/Documents/R/19DSB2017/input/sources/stage1_labels.csv')

test_df = pd.read_csv('C:/Users/SriPrav/Documents/R/19DSB2017/input/sources/stage1_sample_submission.csv')
test_x = np.array([np.mean(np.load('C:/Users/SriPrav/Documents/R/19DSB2017/input/sources/stage1/stage1/%s.npy' % str(id)), axis=0) for id in test_df['id'].tolist()])
    
def get_model(size):
    m = k.models.Sequential()
    m.add(l.Dense(128, input_dim=size))
    m.add(l.Activation('relu'))
    m.add(l.Dense(32))
    m.add(l.Activation('relu'))
    m.add(l.Dense(1))
    m.add(l.Activation('sigmoid'))
    
    m.compile(loss='binary_crossentropy', optimizer='adam')
    return m

def train_nnet(i):
    #train_patients, valid_patients, train_y, valid_y = get_train_CV_fold(stage1_labels_CVindices, i)
    X_build = stage1_labels_CVindices[stage1_labels_CVindices['CVindices'] != i]
    X_val   = stage1_labels_CVindices[stage1_labels_CVindices['CVindices'] == i]
    train_list = X_build['id'].values
    valid_list = X_val['id'].values
    train_y    = X_build['cancer'].values
    valid_y    = X_val['cancer'].values
    print('Train patients: {}'.format(len(train_list)))
    print('Valid patients: {}'.format(len(valid_list)))
    pred_cv = pd.DataFrame({'id': X_val['id'], 'cancer': 0})

    for id in train_list:
        trn_x = np.array([np.mean(np.load('C:/Users/SriPrav/Documents/R/19DSB2017/input/sources/stage1/stage1/%s.npy' % str(id)), axis=0) for id in X_build['id'].tolist()])
    trn_y = X_build['cancer'].as_matrix()
    for id in valid_list:
        val_x = np.array([np.mean(np.load('C:/Users/SriPrav/Documents/R/19DSB2017/input/sources/stage1/stage1/%s.npy' % str(id)), axis=0) for id in X_val['id'].tolist()])
    val_y = X_val['cancer'].as_matrix()
    
    m = get_model(trn_x.shape[1])
    m.fit(trn_x, trn_y, batch_size=32, nb_epoch=40, validation_data=(val_x, val_y), verbose=2)     
#    clf = xgb.XGBRegressor(max_depth=5,
#                               n_estimators=1500,
#                               min_child_weight=95,
#                               learning_rate=0.035,
#                               nthread=8,
#                               subsample=0.85,
#                               colsample_bytree=0.90,
#                               seed=2017)

#    clf = xgb.XGBRegressor(max_depth=10,
#                           n_estimators=1500,
#                           min_child_weight=3,
#                           learning_rate=0.01,
#                           nthread=8,
#                           subsample=0.70,
#                           colsample_bytree=0.70,
#                           seed=2017)

#    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)
    pred_valid = [p[0] for p in m.predict(val_x)] #m.predict(val_x)
    pred_cv['cancer'] = pred_valid
    sub_valfile = 'C:/Users/SriPrav/Documents/R/19DSB2017/submissions/Prav.nnet.fold' + str(i) + '.csv'
    pred_cv.to_csv(sub_valfile, index=False)    
    pred_test =[p[0] for p in m.predict(test_x)] #
    test_df['cancer'] = pred_test
    
    sub_file = 'C:/Users/SriPrav/Documents/R/19DSB2017/submissions/Prav.nnet.fold' + str(i) + '-test' + '.csv'
    test_df.to_csv(sub_file, index=False)
    del pred_cv

fulltrain_x = np.array([np.mean(np.load('C:/Users/SriPrav/Documents/R/19DSB2017/input/sources/stage1/stage1/%s.npy' % str(id)), axis=0) for id in stage1_labels_CVindices['id'].tolist()])
fulltrain_y = stage1_labels_CVindices['cancer'].as_matrix()
print('Full Train patients: {}'.format(len(stage1_labels_CVindices['id'].values)))
test_df['cancer'] = 0
test_df['cancer'].head()
 
  
def fulltrain_nnet(bags):
    test_df['cancer'] = 0
    runseed = 2017 
    for i in range(1, bags+1):
        runseed = runseed + i
        m = get_model(fulltrain_x.shape[1])
        m.fit(fulltrain_x, fulltrain_y, batch_size=32, nb_epoch=50, validation_data=(fulltrain_x, fulltrain_y), verbose=2)
#        clf = xgb.XGBRegressor(max_depth=5,
#                               n_estimators=150,
#                               min_child_weight=95,
#                               learning_rate=0.035,
#                               nthread=8,
#                               subsample=0.85,
#                               colsample_bytree=0.90,
#                               seed=runseed)
#
#        clf.fit(fulltrain_x, fulltrain_y, eval_set=[(fulltrain_x, fulltrain_y)], verbose=True, eval_metric='logloss')     
        pred_test = [p[0] for p in m.predict(test_x)] #m.predict(test_x) 
        test_df['cancer'] += pred_test
        test_df['cancer'].head() 
    test_df['cancer'] = test_df['cancer']/folds
    sub_file = 'C:/Users/SriPrav/Documents/R/19DSB2017/submissions/Prav.nnet.full' + str(i) + '-bags' + '.csv'
    test_df.to_csv(sub_file, index=False)

    
folds = 5
bags = 5
if __name__ == '__main__':
    for i in range(1, folds+1):
        train_nnet(i)
    fulltrain_nnet(bags)
    
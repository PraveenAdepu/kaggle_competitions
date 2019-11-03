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


inDir = 'C:/Users/SriPrav/Documents/R/19DSB2017'

labels = inDir + "/input/sources/stage1_labels.csv"
FeatureExtraction_Folder = inDir + "/input/FeatureExtraction_01"
cvfolds = inDir + "/CVSchema/Prav_CVindices_5folds.csv"
sample_submission = inDir + "/input/sources/stage1_sample_submission.csv"
FeatureFiles = inDir + "/input/FeatureExtraction_01/%s.npy"


stage1_labels = pd.read_csv(labels)
Prav_CVindices_5folds = pd.read_csv(cvfolds)
stage1_labels_CVindices = pd.merge(stage1_labels, Prav_CVindices_5folds, on=['id', 'cancer'], how='left')
train_csv_table = pd.read_csv(labels)

test_df = pd.read_csv(sample_submission)
test_x = np.array([np.mean(np.load(FeatureFiles % str(id)), axis=0) for id in test_df['id'].tolist()])
    


def train_xgboost(i):
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
        trn_x = np.array([np.mean(np.load(FeatureFiles % str(id)), axis=0) for id in X_build['id'].tolist()])
    trn_y = X_build['cancer'].as_matrix()
    for id in valid_list:
        val_x = np.array([np.mean(np.load(FeatureFiles % str(id)), axis=0) for id in X_val['id'].tolist()])
    val_y = X_val['cancer'].as_matrix()
         
    clf = xgb.XGBRegressor(max_depth=5,
                               n_estimators=1500,
                               min_child_weight=95,
                               learning_rate=0.035,
                               nthread=8,
                               subsample=0.85,
                               colsample_bytree=0.90,
                               seed=2017)


    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=100)
    pred_valid = np.clip(clf.predict(val_x),0.001,1)
    pred_cv['cancer'] = pred_valid
    sub_valfile = 'C:/Users/SriPrav/Documents/R/19DSB2017/submissions/Prav.bl02.fold' + str(i) + '.csv'
    pred_cv.to_csv(sub_valfile, index=False)
    
    pred_test = np.clip(clf.predict(test_x),0.001,1)
    test_df['cancer'] = pred_test
    
    sub_file = 'C:/Users/SriPrav/Documents/R/19DSB2017/submissions/Prav.bl02.fold' + str(i) + '-test' + '.csv'
    test_df.to_csv(sub_file, index=False)
    del pred_cv

fulltrain_x = np.array([np.mean(np.load(FeatureFiles % str(id)), axis=0) for id in stage1_labels_CVindices['id'].tolist()])
fulltrain_y = stage1_labels_CVindices['cancer'].as_matrix()
print('Full Train patients: {}'.format(len(stage1_labels_CVindices['id'].values)))
test_df['cancer'] = 0
test_df['cancer'].head()

def fulltrain_xgboost(bags):
    runseed = 2017 
    for i in range(1, bags+1):
        runseed = runseed + i 
        clf = xgb.XGBRegressor(max_depth=5,
                               n_estimators=150,
                               min_child_weight=95,
                               learning_rate=0.035,
                               nthread=8,
                               subsample=0.85,
                               colsample_bytree=0.90,
                               seed=runseed)

        clf.fit(fulltrain_x, fulltrain_y, eval_set=[(fulltrain_x, fulltrain_y)], verbose=True, eval_metric='logloss')     
        pred_test = clf.predict(test_x)
        test_df['cancer'] += pred_test
        test_df['cancer'].head() 
    test_df['cancer'] = test_df['cancer']/folds
    sub_file = 'C:/Users/SriPrav/Documents/R/19DSB2017/submissions/Prav.bl02.full' + str(i) + '-bags' + '.csv'
    test_df.to_csv(sub_file, index=False)

folds = 5
if __name__ == '__main__':
    for i in range(1, folds+1):
        train_xgboost(i)
    fulltrain_xgboost(folds)
    
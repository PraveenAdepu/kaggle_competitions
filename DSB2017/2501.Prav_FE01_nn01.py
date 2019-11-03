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
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

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
    
MODEL_WEIGHTS_FILE = 'FE01_dsb_nn01_weights.h5'
   
#def get_model(size):
#    m = k.models.Sequential()
#    m.add(l.Dense(128, input_dim=size))
#    m.add(l.Activation('relu'))
#    m.add(l.Dense(32))
#    m.add(l.Activation('relu'))
#    m.add(l.Dense(1))
#    m.add(l.Activation('sigmoid'))
#    
#    m.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
#    return m
def nn_model(size):
    model = Sequential()
    
    model.add(Dense(800, input_dim = size, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
       
    model.add(Dense(400, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(200, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(50, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'he_normal', activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics=['accuracy'])
    return(model)
    
callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=1)]

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
        trn_x = np.array([np.mean(np.load(FeatureFiles % str(id)), axis=0) for id in X_build['id'].tolist()])
    trn_y = X_build['cancer'].as_matrix()
    for id in valid_list:
        val_x = np.array([np.mean(np.load(FeatureFiles % str(id)), axis=0) for id in X_val['id'].tolist()])
    val_y = X_val['cancer'].as_matrix()
    
#    m = nn_model(trn_x.shape[1])
#    callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=1)]
#    m.fit(trn_x, trn_y, batch_size=32, nb_epoch=15, validation_data=(val_x, val_y), verbose=2,callbacks=callbacks)     
    m = nn_model(trn_x.shape[1])
    m.fit(trn_x, trn_y, batch_size=32, nb_epoch=10, validation_data=(val_x, val_y), verbose=2)
    pred_valid = [p[0] for p in m.predict(val_x)] #m.predict(val_x)
    pred_cv['cancer'] = pred_valid
    sub_valfile = 'C:/Users/SriPrav/Documents/R/19DSB2017/submissions/Prav.FE01.nn01.fold' + str(i) + '.csv'
    pred_cv.to_csv(sub_valfile, index=False)    
    pred_test =[p[0] for p in m.predict(test_x)] #
    test_df['cancer'] = pred_test
    
    sub_file = 'C:/Users/SriPrav/Documents/R/19DSB2017/submissions/Prav.FE01.nn01.fold' + str(i) + '-test' + '.csv'
    test_df.to_csv(sub_file, index=False)
    #os.remove('C:/Users/SriPrav/Documents/R/19DSB2017/FE01_dsb_nn01_weights.h5')
    del pred_cv

fulltrain_x = np.array([np.mean(np.load(FeatureFiles % str(id)), axis=0) for id in stage1_labels_CVindices['id'].tolist()])
fulltrain_y = stage1_labels_CVindices['cancer'].as_matrix()
print('Full Train patients: {}'.format(len(stage1_labels_CVindices['id'].values)))
test_df['cancer'] = 0
test_df['cancer'].head()
 
  
def fulltrain_nnet(bags):
    test_df['cancer'] = 0
    runseed = 2017 
    for i in range(1, bags+1):
        runseed = runseed + i
        m = nn_model(fulltrain_x.shape[1])
        m.fit(fulltrain_x, fulltrain_y, batch_size=32, nb_epoch=10, validation_data=(fulltrain_x, fulltrain_y), verbose=2)
   
        pred_test = [p[0] for p in m.predict(test_x)] #m.predict(test_x) 
        test_df['cancer'] += pred_test
        test_df['cancer'].head() 
    test_df['cancer'] = test_df['cancer']/folds
    sub_file = 'C:/Users/SriPrav/Documents/R/19DSB2017/submissions/Prav.FE01.nn01.full' + str(i) + '-bags' + '.csv'
    test_df.to_csv(sub_file, index=False)

    
folds = 5
bags = 5
if __name__ == '__main__':
    for i in range(1, folds+1):
        train_nnet(i)
    fulltrain_nnet(bags)
    
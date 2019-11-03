''' 
Author: Danijel Kivaranovic 
Title: Neural network (Keras) with sparse data
'''

## import libraries
import numpy as np
np.random.seed(123)

import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
import itertools


#COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,' \
#               'cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,' \
#               'cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,' \
#               'cat4,cat14,cat38,cat24,cat82,cat25'.split(',')
#               
#ToDeleteColumns =   [  "cat87_cat40"  , "cat10_cat5"   , "cat10_cat4"   , "cat87_cat5"  
#                    , "cat28_cat25"  , "cat28_cat76"  , "cat90_cat73"  , "cat13_cat5"  , "cat11_cat5"  , "cat16_cat6"  , "cat7_cat81"   
#                    , "cat16_cat76"  , "cat1_cat16"   , "cat13_cat16"  , "cat79_cat89" , "cat87_cat90" , "cat7_cat38"   
#                    , "cat36_cat24"  , "cat28_cat14"  , "cat28_cat6"   , "cat7_cat28"  , "cat12_cat5"  , "cat87_cat4"  , "cat23_cat28"  
#                    , "cat13_cat9"   , "cat10_cat13"  , "cat28_cat38"  , "cat40_cat38" , "cat12_cat89" , "cat7_cat25"   
#                    , "cat16_cat36"  , "cat72_cat90"  , "cat89_cat82"  , "cat11_cat16" , "cat7_cat50"  , "cat57_cat23"  
#                    , "cat7_cat36"   , "cat13_cat90"  , "cat40_cat14"  , "cat90_cat50" , "cat7_cat9"   , "cat7_cat3"   , "cat16_cat23"  
#                    , "cat7_cat23"   , "cat76_cat4"   , "cat23_cat24"  , "cat81_cat14" , "cat38_cat24"         
#                    , "cat13_cat24"  , "cat36_cat14"  , "cat7_cat1"    , "cat11_cat28" , "cat72_cat14"  
#                    , "cat89_cat81"  , "cat12_cat90"  , "cat57_cat73"  , "cat24_cat25" , "cat57_cat9"  , "cat57_cat1"  , "cat16_cat24"  
#                    , "cat7_cat40"   , "cat89_cat9"   , "cat111_cat14" , "cat79_cat90" , "cat12_cat7"  , "cat89_cat111", "cat89_cat2"   
#                    , "cat90_cat28"  , "cat28_cat5"   , "cat16_cat28"  , "cat6_cat14"  , "cat16_cat73" , "cat89_cat103" 
#                    , "cat57_cat28"  , "cat89_cat38"  , "cat90_cat23"  , "cat57_cat36" , "cat7_cat6"   , "cat90_cat25" , "cat89_cat76"  
#                    , "cat7_cat24"   , "cat89_cat72"  , "cat57_cat14"  , "cat7_cat13"  , "cat14_cat38" , "cat90_cat40" , "cat11_cat14"  
#                    , "cat40_cat25"  , "cat90_cat24"  , "cat57_cat10"  , "cat40_cat4"  , "cat90_cat36" , "cat3_cat14"  , "cat90_cat6"   
#                    , "cat12_cat14"  , "cat14_cat24"  , "cat11_cat90"  , "cat10_cat89" , "cat80_cat14" , "cat87_cat14" , "cat89_cat50"  
#                    , "cat57_cat40"  , "cat10_cat16"  , "cat89_cat28"  , "cat57_cat2"  , "cat14_cat25" , "cat89_cat14" , "cat57_cat82"  
#                    , "cat89_cat6"   , "cat10_cat14"  , "cat7_cat90"   , "cat57_cat25" , "cat2_cat16"  , "cat89_cat40"  
#                    , "cat9_cat16"   , "cat89_cat25"  , "cat89_cat1"   , "cat7_cat14"  , "cat16_cat38" , "cat89_cat3"   
#                    , "cat3_cat4"    , "cat16_cat5"   , "cat89_cat24"  , "cat90_cat76" , "cat87_cat16" , "cat13_cat14"  
#                    , "cat23_cat14"  , "cat89_cat11"  , "cat5_cat14"  , "cat16_cat90"    
#                    , "cat90_cat5"   , "cat89_cat16"  , "cat90_cat14"  , "cat3_cat5"   , "cat89_cat36" , "cat89_cat13"  
#                    , "cat4_cat14"   , "cat7_cat4"    , "cat57_cat24"     
#                    , "cat7_cat5"    , "cat57_cat89"  , "cat89_cat4"   , "cat16_cat4"  , "cat76_cat14" , "cat57_cat4"   
#                    , "cat89_cat90"  , "cat89_cat5"     
#                    , "cat4_cat24"   , "cat16_cat14"  , "cat57_cat5" ]

## Batch generators ##################################################################################################################################

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

########################################################################################################################################################

## read data
train = pd.read_csv('C:/Users/SriPrav/Documents/R/14Allstate/input/train.csv')
test = pd.read_csv('C:/Users/SriPrav/Documents/R/14Allstate/input/test.csv')

#for comb in itertools.combinations(COMB_FEATURE, 2):
#    feat = comb[0] + "_" + comb[1]
#    train[feat] = train[comb[0]] + train[comb[1]]
#    print('Analyzing Columns:', feat)
#    
#for comb in itertools.combinations(COMB_FEATURE, 2):
#    feat = comb[0] + "_" + comb[1]
#    test[feat] = test[comb[0]] + test[comb[1]]
#    print('Analyzing Columns:', feat)

#trainFeatures = pd.read_csv('C:/Users/SriPrav/Documents/R/14Allstate/input/trainFeatures.csv')
#testFeatures = pd.read_csv('C:/Users/SriPrav/Documents/R/14Allstate/input/testFeatures.csv')
#
#train = pd.merge( train, trainFeatures, on=['id'] , how='left'  )
#test  = pd.merge( test, testFeatures, on=['id'] , how='left'  )
## set test loss to NaN
test['loss'] = np.nan

#train.drop([col for col in ToDeleteColumns if col in train], 
#        axis=1, inplace=True)
#test.drop([col for col in ToDeleteColumns if col in test], 
#        axis=1, inplace=True)
#
#train.columns
## response and IDs
y = np.log(train['loss'].values+200)
id_train = train['id'].values
id_test = test['id'].values

## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)

## Preprocessing and transforming to sparse data
sparse_data = []

f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data, tmp)

## neural net
def nn_model():
    model = Sequential()
    
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization()) 
    model.add(Dropout(0.4))
    
    model.add(Dense(200, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization()) 
    model.add(Dropout(0.2))
    
    model.add(Dense(50, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)

## cv-folds
nfolds = 5
folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)

## train models
i = 0
nbags = 1
nepochs = 100
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])


for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        model = nn_model()
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0], #samples_per_epoch=(batchsize * np.ceil(X_train.shape[0]/batchsize))
                                  verbose = 1,validation_data=(xte.todense(),yte)#,callbacks=[EarlyStopping(patience=5)]
                                  )
        pred += np.exp(model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0])-200
        pred_test += np.exp(model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0])-200
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(np.exp(yte)-200, pred)
    i += 1
    print('Fold ', i, '- MAE:', score)

print('Total - MAE:', mean_absolute_error(np.exp(y)-200, pred_oob))

## train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
df.to_csv('C:/Users/SriPrav/Documents/R/14Allstate/submissions/preds_oob_keras_gpu05.csv', index = False)

## test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv('C:/Users/SriPrav/Documents/R/14Allstate/submissions/submission_keras_gpu05.csv', index = False)


#Keras_03
#('Fold ', 1, '- MAE:', 1127.6323839971628)
#('Fold ', 2, '- MAE:', 1135.0933273852793)
#('Fold ', 3, '- MAE:', 1153.7295703550872)
#('Fold ', 4, '- MAE:', 1140.229600464836)
#('Fold ', 5, '- MAE:', 1143.0014918088336)
#('Total - MAE:', 1139.9372569784382)

xtr = xtrain[inTr]
ytr = y[inTr]
xte = xtrain[inTe]
yte = y[inTe]
    
pred_fulltest = np.zeros(xtest.shape[0])

print('Full model training')
for j in range(1,nbags+1):
    print('bag ', j , ' Processing')        
    model = nn_model()
    fit      = model.fit(xtrain, y, nb_epoch=nepochs, batch_size=128,  verbose=2)
    pred_fulltest += np.exp(model.predict(xtest)[:,0])-200
pred_fulltest/= nbags
pred_fulltest_df = pd.DataFrame({'id': id_test, 'loss': pred_fulltest})    
pred_fulltest_df.to_csv('C:/Users/padepu/Documents/R/14Allstate/submissions/prav.keras02_fullModel.full.csv', index = False)
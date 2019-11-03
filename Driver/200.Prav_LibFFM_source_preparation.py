# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 21:14:45 2017

@author: SriPrav
"""

import math
import numpy as np
import pandas as pd

inDir = 'C:\\Users\\SriPrav\\Documents\\R\\36Driver'
train = pd.read_csv(inDir+'/input/train.csv')
test = pd.read_csv(inDir+'/input/test.csv')


test.insert(1,'target',0)

trainfoldSource = pd.read_csv(inDir+'/input/Prav_5folds_CVindices.csv')
train = pd.merge(train, trainfoldSource, how='left',on="id")

test.insert(59,'CVindices',0)

print(train.shape)
print(test.shape)

x = pd.concat([train,test])
x = x.reset_index(drop=True)
unwanted = x.columns[x.columns.str.startswith('ps_calc_')]
x.drop(unwanted,inplace=True,axis=1)

features = x.columns[2:39]
categories = []
for c in features:
    trainno = len(x.loc[:train.shape[0],c].unique())
    testno = len(x.loc[train.shape[0]:,c].unique())
    print(c,trainno,testno)
    
x.loc[:,'ps_reg_03'] = pd.cut(x['ps_reg_03'], 50,labels=False)
x.loc[:,'ps_car_12'] = pd.cut(x['ps_car_12'], 50,labels=False)
x.loc[:,'ps_car_13'] = pd.cut(x['ps_car_13'], 50,labels=False)
x.loc[:,'ps_car_14'] =  pd.cut(x['ps_car_14'], 50,labels=False)
x.loc[:,'ps_car_15'] =  pd.cut(x['ps_car_15'], 50,labels=False)

test = x.loc[train.shape[0]:].copy()
train = x.loc[:train.shape[0]].copy()

#Always good to shuffle for SGD type optimizers
#train = train.sample(frac=1).reset_index(drop=True)

train.drop('id',inplace=True,axis=1)
test.drop('id',inplace=True,axis=1)

train.head()

test.head()

categories = train.columns[1:]
numerics = []

currentcode = len(numerics)
catdict = {}
catcodes = {}
for x in numerics:
    catdict[x] = 0
for x in categories:
    catdict[x] = 1

noofrows = train.shape[0]
noofcolumns = len(features)
with open("alltrainffm.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if((n%100000)==0):
            print('Row',n)
        datastring = ""
        datarow = train.iloc[r].to_dict()
        datastring += str(int(datarow['target']))


        for i, x in enumerate(catdict.keys()):
            if(catdict[x]==0):
                datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
            else:
                if(x not in catcodes):
                    catcodes[x] = {}
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode
                elif(datarow[x] not in catcodes[x]):
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode

                code = catcodes[x][datarow[x]]
                datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"
        datastring += '\n'
        text_file.write(datastring)
        
noofrows = test.shape[0]
noofcolumns = len(features)
with open("alltestffm.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if((n%100000)==0):
            print('Row',n)
        datastring = ""
        datarow = test.iloc[r].to_dict()
        datastring += str(int(datarow['target']))


        for i, x in enumerate(catdict.keys()):
            if(catdict[x]==0):
                datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
            else:
                if(x not in catcodes):
                    catcodes[x] = {}
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode
                elif(datarow[x] not in catcodes[x]):
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode

                code = catcodes[x][datarow[x]]
                datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"
        datastring += '\n'
        text_file.write(datastring)
        
# sub = pd.read_csv('../input/sample_submission.csv')
# outputs = pd.read_csv('output.txt',header=None)
# outputs.columns = ['target']
# sub.target = outputs.target.ravel()
# sub.to_csv('libffmsubmission.csv',index=False)

folds = 5
for fold in range(1, folds+1):
    print(fold)
    X_build = train[train['CVindices'] != fold]
    X_valid   = train[train['CVindices'] == fold]
    

    fold_file = inDir +'/input/Prav.LibFFM.fold' + str(fold) + '.csv'
    X_valid[["id","CVindices"]].to_csv(fold_file, index=False)
         
    X_build.drop('id',inplace=True,axis=1)
    X_build.drop('CVindices',inplace=True,axis=1)
    
    X_valid.drop('id',inplace=True,axis=1)
    X_valid.drop('CVindices',inplace=True,axis=1)
    
    X_build.head()
    
    categories = X_build.columns[1:]
    numerics = []
    
    currentcode = len(numerics)
    catdict = {}
    catcodes = {}
    for x in numerics:
        catdict[x] = 0
    for x in categories:
        catdict[x] = 1
    
    noofrows = X_build.shape[0]
    noofcolumns = len(features)
    trainfoldfile = "trainfold"+str(fold)+".txt"
    with open(trainfoldfile, "w") as text_file:
        for n, r in enumerate(range(noofrows)):
            if((n%100000)==0):
                print('Row',n)
            datastring = ""
            datarow = X_build.iloc[r].to_dict()
            datastring += str(int(datarow['target']))
    
    
            for i, x in enumerate(catdict.keys()):
                if(catdict[x]==0):
                    datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
                else:
                    if(x not in catcodes):
                        catcodes[x] = {}
                        currentcode +=1
                        catcodes[x][datarow[x]] = currentcode
                    elif(datarow[x] not in catcodes[x]):
                        currentcode +=1
                        catcodes[x][datarow[x]] = currentcode
    
                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"
            datastring += '\n'
            text_file.write(datastring)
    
    noofrows = X_valid.shape[0]        
    validfoldfile = "validfold"+str(fold)+".txt"
    with open(validfoldfile, "w") as text_file:
        for n, r in enumerate(range(noofrows)):
            if((n%100000)==0):
                print('Row',n)
            datastring = ""
            datarow = X_valid.iloc[r].to_dict()
            datastring += str(int(datarow['target']))
    
    
            for i, x in enumerate(catdict.keys()):
                if(catdict[x]==0):
                    datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
                else:
                    if(x not in catcodes):
                        catcodes[x] = {}
                        currentcode +=1
                        catcodes[x][datarow[x]] = currentcode
                    elif(datarow[x] not in catcodes[x]):
                        currentcode +=1
                        catcodes[x][datarow[x]] = currentcode
    
                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"
            datastring += '\n'
            text_file.write(datastring)
            
#############################################################################################################


test_file = inDir +'/input/Prav.LibFFM.test.csv'
test[["id","CVindices"]].to_csv(test_file, index=False)
     
train.drop('id',inplace=True,axis=1)
train.drop('CVindices',inplace=True,axis=1)

test.drop('id',inplace=True,axis=1)
test.drop('CVindices',inplace=True,axis=1)

train.head()

categories = train.columns[1:]
numerics = []

currentcode = len(numerics)
catdict = {}
catcodes = {}
for x in numerics:
    catdict[x] = 0
for x in categories:
    catdict[x] = 1

noofrows = train.shape[0]
noofcolumns = len(features)
trainfile = "train.txt"
with open(trainfile, "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if((n%100000)==0):
            print('Row',n)
        datastring = ""
        datarow = train.iloc[r].to_dict()
        datastring += str(int(datarow['target']))


        for i, x in enumerate(catdict.keys()):
            if(catdict[x]==0):
                datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
            else:
                if(x not in catcodes):
                    catcodes[x] = {}
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode
                elif(datarow[x] not in catcodes[x]):
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode

                code = catcodes[x][datarow[x]]
                datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"
        datastring += '\n'
        text_file.write(datastring)

noofrows = test.shape[0]        
testfile = "test.txt"
with open(testfile, "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if((n%100000)==0):
            print('Row',n)
        datastring = ""
        datarow = test.iloc[r].to_dict()
        datastring += str(int(datarow['target']))


        for i, x in enumerate(catdict.keys()):
            if(catdict[x]==0):
                datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
            else:
                if(x not in catcodes):
                    catcodes[x] = {}
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode
                elif(datarow[x] not in catcodes[x]):
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode

                code = catcodes[x][datarow[x]]
                datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"
        datastring += '\n'
        text_file.write(datastring)
        
#############################################################################################################################################
        
#cd /d C:\Users\SriPrav\Documents\R\36Driver\input\libffm
#
#ffm-train -l 0.0001 -k 15 -t 30 -r 0.01 -s 15 --auto-stop -p validfold1.txt trainfold1.txt model
#ffm-predict validfold1.txt model validfold1_preds.txt
#
#ffm-train -l 0.0001 -k 15 -t 30 -r 0.01 -s 15 --auto-stop -p validfold2.txt trainfold2.txt model
#ffm-predict validfold2.txt model validfold2_preds.txt
#
#ffm-train -l 0.0001 -k 15 -t 30 -r 0.01 -s 15 --auto-stop -p validfold3.txt trainfold3.txt model
#ffm-predict validfold3.txt model validfold3_preds.txt
#
#ffm-train -l 0.0001 -k 15 -t 30 -r 0.01 -s 15 --auto-stop -p validfold4.txt trainfold4.txt model
#ffm-predict validfold4.txt model validfold4_preds.txt
#
#ffm-train -l 0.0001 -k 15 -t 30 -r 0.01 -s 15 --auto-stop -p validfold5.txt trainfold5.txt model
#ffm-predict validfold5.txt model validfold5_preds.txt
#
#ffm-train -l 0.0001 -k 15 -t 7 -r 0.01 -s 15 train.txt model
#ffm-predict test.txt model test_preds.txt

#############################################################################################################################################

# convert all pred.txt files into pred.csv files manually for now
ModelName = "libffm"
folds = 5
for fold in range(1, folds+1):
    validid_file = inDir+"/input/Prav.LibFFM.fold"+str(fold)+".csv"
    validpreds_file = inDir+"/input/libffm/validfold"+str(fold)+"_preds.csv"
    validids = pd.read_csv(validid_file)
    validpreds = pd.read_csv(validpreds_file, header=None)
    
    validpreds.columns = ['target']  
    
    del validids['CVindices']    
    
    valid_predictions = pd.concat([validids,validpreds], axis = 1) 
    
    sub_valfile = inDir + '/submissions/Prav.'+ str(ModelName)+'.fold' + str(fold) + '.csv'
    valid_predictions[["id","target"]].to_csv(sub_valfile, index=False)
        
        
testid_file = inDir+"/input/Prav.LibFFM.test.csv"
testpreds_file = inDir+"/input/libffm/test_preds.csv"
testids = pd.read_csv(testid_file)
testpreds = pd.read_csv(testpreds_file, header=None)

testpreds.columns = ['target']  

del testids['CVindices']    

test_predictions = pd.concat([testids,testpreds], axis = 1) 

test_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.full' + '.csv'
test_predictions[["id","target"]].to_csv(test_file, index=False)       
        
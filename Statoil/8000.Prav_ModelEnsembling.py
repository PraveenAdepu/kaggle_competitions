# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:37:14 2017

@author: SriPrav
"""

import pandas as pd

inDir = 'C:/Users/SriPrav/Documents/R/35Statoil'

fold1 = pd.read_csv(inDir + "/submissions/Prav.vgg03.fold1-test.csv")
fold2 = pd.read_csv(inDir + "/submissions/Prav.vgg03.fold2-test.csv")
fold3 = pd.read_csv(inDir + "/submissions/Prav.vgg03.fold3-test.csv")
fold4 = pd.read_csv(inDir + "/submissions/Prav.vgg03.fold4-test.csv")

test = pd.merge(fold1, fold2, on="id",how="left")
test = pd.merge(test, fold3, on="id",how="left")
test = pd.merge(test, fold4, on="id",how="left")

test["is_iceberg"] = test.mean(axis=1)

test[["id","is_iceberg"]].to_csv(inDir+"/submissions/Prav.vgg03.folds1-4.mean.csv", index=False)

fold1 = pd.read_csv(inDir + "/submissions/Prav.vgg04.fold1-test.csv")
fold2 = pd.read_csv(inDir + "/submissions/Prav.vgg04.fold2-test.csv")
fold3 = pd.read_csv(inDir + "/submissions/Prav.vgg04.fold3-test.csv")
fold4 = pd.read_csv(inDir + "/submissions/Prav.vgg04.fold4-test.csv")

test = pd.merge(fold1, fold2, on="id",how="left")
test = pd.merge(test, fold3, on="id",how="left")
test = pd.merge(test, fold4, on="id",how="left")

test["is_iceberg"] = test.mean(axis=1)

test[["id","is_iceberg"]].to_csv(inDir+"/submissions/Prav.vgg04.folds1-4.mean.csv", index=False)

vgg01_model = pd.read_csv(inDir + "/submissions/Prav.vgg03.folds1-4.mean.csv")
vgg02_model = pd.read_csv(inDir + "/submissions/Prav.vgg04.folds1-4.mean.csv")

models = pd.merge(vgg01_model, vgg02_model, on="id",how="left")

models[["is_iceberg_x","is_iceberg_y"]].corr()

models["is_iceberg"] = models.mean(axis=1)

models[["id","is_iceberg"]].to_csv(inDir+"/submissions/Prav.vgg03.vgg04.mean.csv", index=False)

################################################################################################################################

model_01 = pd.read_csv(inDir + "/submissions/Prav.vgg03.vgg04.mean.csv")
model_02 = pd.read_csv(inDir + "/submissions/200_ens_densenet.csv")

models = pd.merge(model_01, model_02, on="id",how="left")

models[["is_iceberg_x","is_iceberg_y"]].corr()

models["is_iceberg"] = models.mean(axis=1)

models[["id","is_iceberg"]].to_csv(inDir+"/submissions/Prav.vgg0304_refDense.mean.csv", index=False)

################################################################################################################################

fold1 = pd.read_csv(inDir + "/submissions/Prav.vgg19_01.fold1-test.csv")
fold2 = pd.read_csv(inDir + "/submissions/Prav.vgg19_01.fold2-test.csv")
fold3 = pd.read_csv(inDir + "/submissions/Prav.vgg19_01.fold3-test.csv")
fold4 = pd.read_csv(inDir + "/submissions/Prav.vgg19_01.fold4-test.csv")

test = pd.merge(fold1, fold2, on="id",how="left")
test = pd.merge(test, fold3, on="id",how="left")
test = pd.merge(test, fold4, on="id",how="left")

test["is_iceberg"] = test.mean(axis=1)

test[["id","is_iceberg"]].to_csv(inDir+"/submissions/Prav.vgg19_01.folds1-4.mean.csv", index=False)

################################################################################################################################

model_01 = pd.read_csv(inDir + "/submissions/Prav.vgg03.vgg04.mean.csv")
model_02 = pd.read_csv(inDir + "/submissions/200_ens_densenet.csv")
model_03 = pd.read_csv(inDir + "/submissions/Prav.vgg19_01.folds1-4.mean.csv")

models = pd.merge(model_01, model_02, on="id",how="left")
models = pd.merge(models, model_03, on="id",how="left")

models[["is_iceberg_x","is_iceberg_y","is_iceberg"]].corr()

models.head()
models["is_iceberg"] = models.mean(axis=1)
models.head()
models[["id","is_iceberg"]].to_csv(inDir+"/submissions/Prav.vgg0304_refDense_vgg19.mean.csv", index=False)

################################################################################################################################

################################################################################################################################

fold1 = pd.read_csv(inDir + "/submissions/Prav.vgg19_02.fold1-test.csv")
fold2 = pd.read_csv(inDir + "/submissions/Prav.vgg19_02.fold2-test.csv")
fold3 = pd.read_csv(inDir + "/submissions/Prav.vgg19_02.fold3-test.csv")
fold4 = pd.read_csv(inDir + "/submissions/Prav.vgg19_02.fold4-test.csv")

test = pd.merge(fold1, fold2, on="id",how="left")
test = pd.merge(test, fold3, on="id",how="left")
test = pd.merge(test, fold4, on="id",how="left")

test["is_iceberg"] = test.mean(axis=1)

test[["id","is_iceberg"]].to_csv(inDir+"/submissions/Prav.vgg19_02.folds1-4.mean.csv", index=False)

################################################################################################################################


################################################################################################################################

fold1 = pd.read_csv(inDir + "/submissions/Prav.vgg19_03.fold1-test.csv")
fold2 = pd.read_csv(inDir + "/submissions/Prav.vgg19_03.fold2-test.csv")
fold3 = pd.read_csv(inDir + "/submissions/Prav.vgg19_03.fold3-test.csv")
fold4 = pd.read_csv(inDir + "/submissions/Prav.vgg19_03.fold4-test.csv")

test = pd.merge(fold1, fold2, on="id",how="left")
test = pd.merge(test, fold3, on="id",how="left")
test = pd.merge(test, fold4, on="id",how="left")

test["is_iceberg"] = test.mean(axis=1)

test[["id","is_iceberg"]].to_csv(inDir+"/submissions/Prav.vgg19_03.folds1-4.mean.csv", index=False)

################################################################################################################################

################################################################################################################################

model_01 = pd.read_csv(inDir + "/submissions/Prav.vgg19_02.folds1-4.mean.csv")
model_02 = pd.read_csv(inDir + "/submissions/Prav.vgg19_03.folds1-4.mean.csv")

models = pd.merge(model_01, model_02, on="id",how="left")

models[["is_iceberg_x","is_iceberg_y"]].corr()

models["is_iceberg"] = models.mean(axis=1)

models[["id","is_iceberg"]].to_csv(inDir+"/submissions/Prav.vgg19_0203.mean.csv", index=False)

################################################################################################################################

################################################################################################################################

model_01 = pd.read_csv(inDir + "/submissions/Prav.vgg03.vgg04.mean.csv")
model_02 = pd.read_csv(inDir + "/submissions/Prav.vgg19_0203.mean.csv")

models = pd.merge(model_01, model_02, on="id",how="left")

models[["is_iceberg_x","is_iceberg_y"]].corr()

models["is_iceberg"] = models.mean(axis=1)

models[["id","is_iceberg"]].to_csv(inDir+"/submissions/Prav.vgg16_0304_vgg19_0203.mean.csv", index=False)

################################################################################################################################

################################################################################################################################

model_01 = pd.read_csv(inDir + "/submissions/Prav.vgg0304_refDense.mean.csv")
model_02 = pd.read_csv(inDir + "/submissions/LB1541_final_ensemble.csv")

models = pd.merge(model_01, model_02, on="id",how="left")

models[["is_iceberg_x","is_iceberg_y"]].corr()

models["is_iceberg"] = models.mean(axis=1)

models[["id","is_iceberg"]].to_csv(inDir+"/submissions/Prav.vgg0304_refDense_ref2.mean.csv", index=False)

################################################################################################################################

################################################################################################################################

model_01 = pd.read_csv(inDir + "/submissions/Prav.vgg03.folds1-4.mean.csv")
model_02 = pd.read_csv(inDir + "/submissions/Prav.vgg04.folds1-4.mean.csv")
model_03 = pd.read_csv(inDir + "/submissions/Prav.vgg19_02.folds1-4.mean.csv")
model_04 = pd.read_csv(inDir + "/submissions/Prav.vgg19_03.folds1-4.mean.csv")

models = pd.merge(model_01, model_02, on="id",how="left")
models = pd.merge(models, model_03, on="id",how="left")
models = pd.merge(models, model_04, on="id",how="left")

models[["is_iceberg_x","is_iceberg_y","is_iceberg_x","is_iceberg_y"]].corr()

models.head()
models["is_iceberg"] = models.mean(axis=1)
models.head()
models[["id","is_iceberg"]].to_csv(inDir+"/submissions/Prav.vgg16_0304_vgg19_0203.mean.csv", index=False)

################################################################################################################################

################################################################################################################################

model_01 = pd.read_csv(inDir + "/submissions/Prav.vgg16_0304_vgg19_0203.mean.csv")
model_02 = pd.read_csv(inDir + "/submissions/200_ens_densenet.csv")
model_03 = pd.read_csv(inDir + "/submissions/LB1541_final_ensemble.csv")
model_04 = pd.read_csv(inDir + "/submissions/vggbnw_fcn_en.csv")

models = pd.merge(model_01, model_02, on="id",how="left")
models = pd.merge(models, model_03, on="id",how="left")
models = pd.merge(models, model_04, on="id",how="left")

models[["is_iceberg_x","is_iceberg_y","is_iceberg_x","is_iceberg_y"]].corr()

models.head()
models["is_iceberg"] = models.mean(axis=1)
models.head()
models[["id","is_iceberg"]].to_csv(inDir+"/submissions/Prav.vgg16_0304_vgg19_0203.mean_3ref_mean.csv", index=False)

################################################################################################################################
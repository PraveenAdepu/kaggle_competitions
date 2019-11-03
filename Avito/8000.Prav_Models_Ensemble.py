# -*- coding: utf-8 -*-
"""
Created on Tue May 15 22:48:17 2018

@author: SriPrav
"""

import numpy as np 
import pandas as pd 

inDir = 'C:/Users/SriPrav/Documents/R/48Avito'


Model01 = pd.read_csv(inDir+'/submissions/Prav.lgbm003.full-test_clip.csv') # LB : 0.2287
Model02 = pd.read_csv(inDir+'/submissions/Prav_Ref_AggFeatures.csv') # LB : 0.2232

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

ModelName = 'Ensemble01'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)

#######################################################################################################################   

SingleModel = pd.read_csv(inDir+'/submissions/Prav.L2_xgb02.full.csv')
SingleModel['deal_probability'] = SingleModel['deal_probability'].clip(0.0, 1.0)

SingleModel[["item_id","deal_probability"]].to_csv(inDir+'/submissions/Prav.L2_xgb02.full_clip.csv', index=False)
#######################################################################################################################
 
Model01 = pd.read_csv(inDir+'/submissions/Prav.xgb1.full_clip.csv') # LB : 0.2258
Model02 = pd.read_csv(inDir+'/submissions/Prav_Ref_AggFeatures.csv') # LB : 0.2232

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

ModelName = 'Ensemble03'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)

#80bf58082ad3  4
#67a9944a7373  3

#######################################################################################################################   

SingleModel = pd.read_csv(inDir+'/submissions/Prav.xgb2.full.csv')
SingleModel['deal_probability'] = SingleModel['deal_probability'].clip(0.0, 1.0)

SingleModel[["item_id","deal_probability"]].to_csv(inDir+'/submissions/Prav.xgb2.full_clip.csv', index=False)
#######################################################################################################################
 
Model01 = pd.read_csv(inDir+'/submissions/Prav.xgb2.full_clip.csv') # LB : 0.2258
Model02 = pd.read_csv(inDir+'/submissions/Prav_Ref_AggFeatures.csv') # LB : 0.2232

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

ModelName = 'Ensemble04'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################
Model01 = pd.read_csv(inDir+'/submissions/Prav.xgb2.full_clip.csv') # LB : 0.2258
Model02 = pd.read_csv(inDir+'/submissions/blend06.csv') # LB : 0.2216

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

ModelName = 'Ensemble05'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################

Model01 = pd.read_csv(inDir+'/submissions/Prav.Ensemble04.csv') # LB : 0.2258
Model02 = pd.read_csv(inDir+'/submissions/blend06.csv') # LB : 0.2216

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

Ensemble.head()

ModelName = 'Ensemble06'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################

#######################################################################################################################   

SingleModel = pd.read_csv(inDir+'/submissions/Prav.lgbm005.full.csv')
SingleModel['deal_probability'] = SingleModel['deal_probability'].clip(0.0, 1.0)

SingleModel[["item_id","deal_probability"]].to_csv(inDir+'/submissions/Prav.lgbm005.full_clip.csv', index=False)
#######################################################################################################################

Model01 = pd.read_csv(inDir+'/submissions/Prav.Ensemble04.csv') # LB : 0.2258
Model02 = pd.read_csv(inDir+'/submissions/blend06.csv') # LB : 0.2216

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

Ensemble.head()

ModelName = 'Ensemble06'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################

#######################################################################################################################
 
Model01 = pd.read_csv(inDir+'/submissions/Prav.xgb2.full_clip.csv') # LB : 0.2258
Model02 = pd.read_csv(inDir+'/submissions/Prav_Ref_AggFeatures02.csv') # LB : 0.2232

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

ModelName = 'Ensemble07'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################

Model01 = pd.read_csv(inDir+'/submissions/Prav.Ensemble07.csv') # LB : 0.2258
Model02 = pd.read_csv(inDir+'/submissions/blend06.csv') # LB : 0.2216

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

Ensemble.head()

ModelName = 'Ensemble07_blend'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################
SingleModel = pd.read_csv(inDir+'/submissions/Prav.xgb3.full.csv')
SingleModel['deal_probability'] = SingleModel['deal_probability'].clip(0.0, 1.0)

SingleModel[["item_id","deal_probability"]].to_csv(inDir+'/submissions/Prav.xgb3.full_clip.csv', index=False)
#######################################################################################################################

#######################################################################################################################
 
Model01 = pd.read_csv(inDir+'/submissions/Prav.xgb3.full_clip.csv') # LB : 0.2258
Model02 = pd.read_csv(inDir+'/submissions/Prav_Ref_AggFeatures02.csv') # LB : 0.2232

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

ModelName = 'Ensemble08'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################

Model01 = pd.read_csv(inDir+'/submissions/Prav.nn05.full.csv') # LB : 0.2258
Model02 = pd.read_csv(inDir+'/submissions/Prav.nn06.full.csv') # LB : 0.2216

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

Ensemble.head()

ModelName = 'nn0506'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################

#######################################################################################################################
 
Model01 = pd.read_csv(inDir+'/submissions/Prav.xgb3.full_clip.csv') # LB : 0.2242
Model02 = pd.read_csv(inDir+'/submissions/Avito_Shanth_RNN_AVERAGE.csv') # LB : 0.227 Avito_Shanth_RNN_AVERAGE

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

ModelName = 'Ensemble09'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################

Model01 = pd.read_csv(inDir+'/submissions/Prav.Ensemble09.csv') # LB : 0.2258
Model02 = pd.read_csv(inDir+'/submissions/Prav_Ref_AggFeatures02.csv') # LB : 0.2232

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

ModelName = 'Ensemble09_Agg'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################

Model01 = pd.read_csv(inDir+'/submissions/Prav.Ensemble09_Agg.csv') # LB : 0.2258
Model02 = pd.read_csv(inDir+'/submissions/blend06.csv') # LB : 0.2216

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

Ensemble.head()

ModelName = 'Ensemble09_blend2'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################

Model01 = pd.read_csv(inDir+'/submissions/Prav.nn07.full.csv') # LB : 0.2258
Model02 = pd.read_csv(inDir+'/submissions/Prav.nn08.full.csv') # LB : 0.2216

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

Ensemble.head()

ModelName = 'nn0506'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################
#####################################################################################################################

Model01 = pd.read_csv(inDir+'/submissions/Prav.L2_xgb02.full_clip.csv')    # LB : 0.2224
Model02 = pd.read_csv(inDir+'/submissions/Prav_Ref_AggFeatures02.csv') # LB : 0.2232

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

ModelName = 'L2_xgb02_Agg'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################

Model01 = pd.read_csv(inDir+'/submissions/Prav.L2_xgb02_Agg.csv') # LB : 0.2258
Model02 = pd.read_csv(inDir+'/submissions/blend06.csv') # LB : 0.2216

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

Ensemble.head()

ModelName = 'L2_xgb02_Agg_blend2'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################
#####################################################################################################################

Model01 = pd.read_csv(inDir+'/submissions/blend06.csv')    # LB : 0.2224
Model02 = pd.read_csv(inDir+'/submissions/Prav.L2_xgb02.full_clip.csv') # LB : 0.2232

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

ModelName = 'L2_xgb02_blend'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################

#####################################################################################################################

Model01 = pd.read_csv(inDir+'/submissions/best_public_blend.csv')    # LB : 0.2204
Model02 = pd.read_csv(inDir+'/submissions/Prav.L2_xgb02_blend.csv') # LB : 0.2209

Ensemble = pd.merge(Model01, Model02, how = 'inner', on = 'item_id')

Ensemble[["deal_probability_x","deal_probability_y"]].corr()

Ensemble["deal_probability"] = Ensemble["deal_probability_x"] * 0.5 + Ensemble["deal_probability_y"] * 0.5

ModelName = 'L2_xgb02_blend_public'
sub_file = inDir +'/submissions/Prav.'+ str(ModelName)+'.csv'
Ensemble['deal_probability'] = Ensemble['deal_probability'].clip(0.0, 1.0)
Ensemble[["item_id","deal_probability"]].to_csv(sub_file, index=False)
#####################################################################################################################



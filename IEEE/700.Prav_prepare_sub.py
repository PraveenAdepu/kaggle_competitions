# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:57:53 2018

@author: SriPrav
"""

import os

import numpy as np
random_state = 2017
np.random.seed(random_state)

import pandas as pd


inDir = 'C:/Users/SriPrav/Documents/R/43IEEE'

model01_fold01 = pd.read_csv(inDir + '/submissions/Prav.InceptionResnet_04.fold1-test.csv')
model01_fold02 = pd.read_csv(inDir + '/submissions/Prav.InceptionResnet_03.fold1-test.csv')
model01_fold03 = pd.read_csv(inDir + '/submissions/Prav.InceptionResnet_03.fold1-test.csv')
model01_fold04 = pd.read_csv(inDir + '/submissions/Prav.Resnet_06.fold1-test.csv')
model01_fold05 = pd.read_csv(inDir + '/submissions/Prav.Resnet_06.fold2-test.csv')
model01_fold06 = pd.read_csv(inDir + '/submissions/Prav.Resnet_06.fold3-test.csv')

model01 = pd.concat([model01_fold01, model01_fold02 ,model01_fold03 ,model01_fold04,model01_fold05,model01_fold06] )

model01.head()
model01[model01['fname']=='img_0002a04_manip.tif']
model01_agg = model01.groupby(['fname']).agg({'HTC-1-M7': np.mean ,
                                              'iPhone-4s': np.mean,
                                              'iPhone-6': np.mean ,
                                              'LG-Nexus-5x': np.mean,
                                              'Motorola-Droid-Maxx': np.mean ,
                                              'Motorola-Nexus-6': np.mean,
                                              'Motorola-X': np.mean ,
                                              'Samsung-Galaxy-Note3': np.mean,
                                              'Samsung-Galaxy-S4': np.mean ,
                                              'Sony-NEX-7': np.mean,
                                                                                   
                                              })

#model01_agg.columns = ["_".join(x) for x in model01_agg.columns.ravel()]
model01_agg.reset_index(level=model01_agg.index.names, inplace=True)
#artemp = artemp.rename(columns={'visit_datetime': 'visit_date'})


model01_agg['camera'] = model01_agg[['HTC-1-M7','iPhone-4s','iPhone-6','LG-Nexus-5x','Motorola-Droid-Maxx','Motorola-Nexus-6','Motorola-X','Samsung-Galaxy-Note3','Samsung-Galaxy-S4','Sony-NEX-7']].idxmax(axis=1)

ModelName= 'IncRes_04_fold1'

sub_valfile = inDir+'/submissions/Prav.'+ModelName+'.fold11.csv'    
model01_agg[['fname','camera']].to_csv(sub_valfile, index=False)
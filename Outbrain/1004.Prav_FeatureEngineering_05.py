# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 14:24:17 2016

@author: SriPrav
"""

import sys
import csv
csv.field_size_limit(2147483647)
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
import pandas as pd



# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


##############################################################################
# parameters #################################################################
##############################################################################

# A, paths
data_path = "C:/Users/SriPrav/Documents/R/13Outbrain/input/"
out_path = "C:/Users/SriPrav/Documents/R/13Outbrain/submissions/"
events = data_path+'events.csv'               # path to training file
page_views = data_path+'page_views.csv'                 # path to testing file
submission = out_path+'Prav_sub_proba_04.csv'  # path of to be outputted submission file
trainLeak = data_path+'trainingSet_Leak.csv'
testLeak = data_path+'testingSet_Leak.csv'

page_views = pd.read_csv(page_views)



page_views["day"] = page_views.timestamp // (3600 * 24 * 1000)

page_views.drop('timestamp', axis=1, inplace=True)

events = pd.read_csv(events)
events["day"]     = events.timestamp // (3600 * 24 * 1000)



events = pd.merge(events, page_views, how='left', on=['uuid', 'document_id', 'platform','geo_location','day'])
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 21:50:36 2016

@author: SriPrav
"""
import sys
import csv
csv.field_size_limit(2147483647)
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
import pandas as pd
import numpy as np
import hashlib
# A, paths
#data_path = "C:/Users/SriPrav/Documents/R/13Outbrain/input/"
#out_path = "C:/Users/SriPrav/Documents/R/13Outbrain/submissions/"
#train = data_path+'trainingSet_20161215.csv'               # path to training file
#test = data_path+'testingSet_20161215.csv'                 # path to testing file
#submission = data_path+'train_libffm.csv'  # path of to be outputted submission file
#

#merge_dat = pd.read_csv(train)
#nr_bins = 2 ** 22
#
#def hashstr(str):
#    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1
#    


# [1] "display_id"                      "ad_id"                           "clicked"                         "uuid"                           
# [5] "event_document_id"               "platform"                        "event_source_id"                 "event_publisher_id"             
# [9] "event_category_id"               "event_entity_id"                 "event_topic_id"                  "document_id"                    
#[13] "campaign_id"                     "advertiser_id"                   "source_id"                       "publisher_id"                   
#[17] "category_id"                     "entity_id"                       "topic_id"                        "location1"                      
#[21] "location2"                       "location3"                       "hour"                            "day"                            
#[25] "minutes"                         "event_publish_dateToDate"        "publish_dateToDate"              "event_publish_dateTopublishdate"
#[29] "leak"                            "traffic_source"                  "weekday"                         "weekflag"                       
#[33] "event_LastCat_id"                "event_Lasttopic_id"              "LastCat_id"                      "Lasttopic_id"                   
#[37] "event_Catconf"                   "event_Entconf"                   "event_topconf"                   "Catconf"                        
#[41] "Entconf"                         "topconf" 
    
def convert(src_path, dst_path, is_train):
    with open(dst_path, 'w') as f:
        for row in csv.DictReader(open(src_path)):
            i = 1
            w = 1
            weight = 1.0
            num = 1
            feats = []
               
            for field in fields:
                #v = hashstr(field+'-'+row[field])
                v = row[field]
                #print v
               
                #feats.append('{i}:{v}:{w}'.format(i=i, v=v ,w=row[field]))
                feats.append('{i}:{v}:1'.format(i=i, v=v ))
                i += 1
                target = '{0}:{1}:{2}'.format(row['clicked'],weight,row['display_id'])
            #print i
            #print num
 
            if is_train == True:
                f.write('{0} {1}\n'.format(target, ' '.join(feats)))
               
            num += 1
            if num % 1000000 == 0:
                print num

def testconvert(src_path, dst_path, is_train):
    with open(dst_path, 'w') as f:
        for row in csv.DictReader(open(src_path)):
            i = 1
            w = 1
            num = 1
            feats = []
               
            for field in fields:
                v = row[field]
                #v = hashstr(field+'-'+row[field])
                #print v
               
                #feats.append('{i}:{v}:{w}'.format(i=i, v=v ,w=row[field]))
                feats.append('{i}:{v}:1'.format(i=i, v=v ))
                i += 1
                target = '{0}:{1}:{2}'.format(0,weight,row['display_id'])
            #print i
            #print num
 
            if is_train == True:
                f.write('{0} {1}\n'.format(target, ' '.join(feats)))
               
            num += 1
            if num % 1000000 == 0:
                print num



fields = [u'ad_id',u'uuid',u'event_document_id',u'platform'
          ,u'event_source_id',u'event_publisher_id' ,u'event_category_id', u'event_topic_id' #,u'event_entity_id'                  
          ,u'document_id',u'campaign_id',u'advertiser_id',u'source_id',u'publisher_id'         
          ,u'location1',u'location2',u'location3' 
          #,u'day'
          ,u'event_Catconf',u'event_topconf'#,u'event_Entconf'
           ,u'hour'
           , u'category_id',u'topic_id' # ,u'entity_id'
           ,u'Catconf',u'topconf'#,u'Entconf'
         #,u'LastCat_id'
         #,u'Lasttopic_id'
          #,u'weekday'
          ,u'weekflag'
          #,u'minutes'
          ,u'event_publish_dateToDate',u'publish_dateToDate',u'event_publish_dateTopublishdate'
          ,u'traffic_source'
          ,u'leak']
          
convert('C:/Users/SriPrav/Documents/R/13Outbrain/input/trainingSet30_train.csv', 'C:/Users/SriPrav/Documents/R/13Outbrain/input/libffm2/trainingSet25_train_ffm.txt', True)
convert('C:/Users/SriPrav/Documents/R/13Outbrain/input/trainingSet30_valid.csv', 'C:/Users/SriPrav/Documents/R/13Outbrain/input/libffm2/trainingSet25_valid_ffm.txt', True)




convert('C:/Users/SriPrav/Documents/R/13Outbrain/input/trainingSet20_train.csv', 'C:/Users/SriPrav/Documents/R/13Outbrain/input/trainingSet202_train_ffm.txt', True)
testconvert('C:/Users/SriPrav/Documents/R/13Outbrain/input/trainingSet20_valid.csv', 'C:/Users/SriPrav/Documents/R/13Outbrain/input/trainingSet2021_valid_ffm.txt', True)

        
convert('C:/Users/SriPrav/Documents/R/13Outbrain/input/trainingSet12.csv'   , 'C:/Users/SriPrav/Documents/R/13Outbrain/input/trainingSet13_ffm.txt', True)
testconvert('C:/Users/SriPrav/Documents/R/13Outbrain/input/testingSet12.csv', 'C:/Users/SriPrav/Documents/R/13Outbrain/input/testingSet13_ffm.txt', True)

# ffm-train -p trainingSet_20161215_valid2_fold5_ffm.txt -l 0.001 -k 16 -t 30 -r 0.05 -s 30 --auto-stop trainingSet_20161215_train2_fold1to4_ffm.txt validationmodel
# ffm-predict trainingSet_20161215_valid_fold5_ffm.txt validationmodel prav_ffm_fold5.csv
# 30 0.42620 0.42860
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm2>ffm-train -p trainingSet25_valid_ffm.txt -l 0.00002 -k 25 -t 30 -r 0.05 -s 20 --map --auto-stop trainingSet25_train_ffm.txt validationmodel25



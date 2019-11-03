# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 20:09:11 2016

@author: SriPrav
"""

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
nr_bins = 2 ** 20

def hashstr(str):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1
    

fields = [u'ad_id',u'uuid',u'event_document_id',u'platform'#,u'event_source_id',u'event_publisher_id',u'event_category_id'#,u'event_entity_id'
          #, u'event_topic_id'          
          ,u'document_id',u'campaign_id',u'advertiser_id',u'source_id',u'publisher_id'#, u'category_id',u'entity_id',u'topic_id'
          ,u'location1',u'location2',u'location3',u'hour',u'day',u'minutes'
          #,u'event_publish_dateToDate'
          ,u'publish_dateToDate'
          #,u'event_publish_dateTopublishdate'
          ,u'leak']

#[1] "display_id"                      "ad_id"                           "clicked"                         "uuid"                           
# [5] "event_document_id"               "platform"                        "event_source_id"                 "event_publisher_id"             
# [9] "event_category_id"               "event_entity_id"                 "event_topic_id"                  "document_id"                    
#[13] "campaign_id"                     "advertiser_id"                   "source_id"                       "publisher_id"                   
#[17] "category_id"                     "entity_id"                       "topic_id"                        "location1"                      
#[21] "location2"                       "location3"                       "hour"                            "day"                            
#[25] "minutes"                         "event_publish_dateToDate"        "publish_dateToDate"              "event_publish_dateTopublishdate"
#[29] "leak"
    

def convert(src_path, dst_path, is_train):
    with open(dst_path, 'w') as f:
        for row in csv.DictReader(open(src_path)):
            i = 1
            w = 1
            num = 1
            feats = []
               
            for field in fields:
                if field == 'publish_dateToDate':
                    v = hashstr(field+'-'+str(int(float(row[field])/30)))
                    feats.append('{i}:{v}:1'.format(i=i, v=v ))
                else:
                    v = hashstr(field+'-'+row[field])
                    
                #print v
               
                #feats.append('{i}:{v}:{w}'.format(i=i, v=v ,w=row[field]))
                    feats.append('{i}:{v}:1'.format(i=i, v=v ))
                i += 1
            #print i
            #print num
 
            if is_train == True:
                f.write('{0} {1}\n'.format(row['clicked'], ' '.join(feats)))
               
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
                if field == 'publish_dateToDate':
                    v = hashstr(field)
                    feats.append('{i}:{v}:{w}'.format(i=i, v=v ,w=row[field]))
                else:
                    v = hashstr(field+'-'+row[field])
                    
                #print v
               
                #feats.append('{i}:{v}:{w}'.format(i=i, v=v ,w=row[field]))
                    feats.append('{i}:{v}:1'.format(i=i, v=v ))
                i += 1
            #print i
            #print num
 
            if is_train == True:
                f.write('{0} {1}\n'.format(0, ' '.join(feats)))
               
            num += 1
            if num % 1000000 == 0:
                print num

convert('C:/Users/SriPrav/Documents/R/13Outbrain/input/trainingSet05_fold1to4.csv', 'C:/Users/SriPrav/Documents/R/13Outbrain/input/trainingSet052_fold1to4_ffm.txt', True)
convert('C:/Users/SriPrav/Documents/R/13Outbrain/input/trainingSet05_fold5.csv', 'C:/Users/SriPrav/Documents/R/13Outbrain/input/trainingSet052_fold5_ffm.txt', True)

convert('C:/Users/SriPrav/Documents/R/13Outbrain/input/trainingSet5_20161215.csv'   , 'C:/Users/SriPrav/Documents/R/13Outbrain/input/trainingSet52_20161215_ffm.txt', True)
testconvert('C:/Users/SriPrav/Documents/R/13Outbrain/input/testingSet5_20161215.csv', 'C:/Users/SriPrav/Documents/R/13Outbrain/input/testingSet52_20161215_ffm.txt', True)

# ffm-train -p trainingSet_20161215_valid2_fold5_ffm.txt -l 0.001 -k 16 -t 30 -r 0.05 -s 30 --auto-stop trainingSet_20161215_train2_fold1to4_ffm.txt validationmodel
# ffm-predict trainingSet_20161215_valid_fold5_ffm.txt validationmodel prav_ffm_fold5.csv
# 30 0.42620 0.42860
#logloss = 0.42860
#iter   tr_logloss   va_logloss
#   1      0.44486      0.43816
#   2      0.43658      0.43544
#   3      0.43452      0.43400
#   4      0.43330      0.43321
#   5      0.43245      0.43264
#   6      0.43178      0.43218
#   7      0.43124      0.43159
#   8      0.43079      0.43132
#   9      0.43039      0.43105
#  10      0.43004      0.43087
#  11      0.42972      0.43060
#  12      0.42943      0.43036
#  13      0.42917      0.43021
#  14      0.42893      0.43006
#  15      0.42870      0.42993
#  16      0.42848      0.42976
#  17      0.42827      0.42966
#  18      0.42808      0.42958
#  19      0.42789      0.42943
#  20      0.42771      0.42939
#  21      0.42754      0.42926
#  22      0.42738      0.42923
#  23      0.42721      0.42902
#  24      0.42706      0.42902
#  25      0.42691      0.42900
#  26      0.42676      0.42888
#  27      0.42662      0.42885
#  28      0.42648      0.42873
#  29      0.42634      0.42863
#  30      0.42620      0.42860
#####################################################################################################################################################################
#iter   tr_logloss   va_logloss
#   1      0.43359      0.42636
#   2      0.42459      0.42352
#   3      0.42241      0.42208
#   4      0.42113      0.42107
#   5      0.42024      0.42037
#   6      0.41956      0.41990
#   7      0.41901      0.41950
#   8      0.41854      0.41923
#   9      0.41815      0.41889
#  10      0.41780      0.41866
#  11      0.41749      0.41842
#  12      0.41722      0.41823
#  13      0.41696      0.41806
#  14      0.41673      0.41790
#  15      0.41651      0.41778
#  16      0.41630      0.41772
#  17      0.41611      0.41755
#  18      0.41592      0.41753
#  19      0.41575      0.41730
#  20      0.41558      0.41728
#  21      0.41542      0.41723
#  22      0.41527      0.41711
#  23      0.41512      0.41696
#  24      0.41498      0.41702
#Auto-stop. Use model at 23th iteration.

# ffm-train  -l 0.001 -k 16 -t 25 -r 0.05 -s 30 trainingSet2_20161215_ffm.txt fulltrainmodel
# ffm-predict testingSet2_20161215_ffm.txt fulltrainmodel prav_fullmodel2_ffm.csv

#iter   tr_logloss
#   1      0.43203
#   2      0.42366
#   3      0.42162
#   4      0.42043
#   5      0.41960
#   6      0.41896
#   7      0.41845
#   8      0.41803
#   9      0.41766
#  10      0.41734
#  11      0.41705
#  12      0.41680
#  13      0.41656
#  14      0.41634
#  15      0.41614
#  16      0.41595
#  17      0.41577
#  18      0.41560
#  19      0.41544
#  20      0.41528
#  21      0.41513
#  22      0.41499
#  23      0.41485
#  24      0.41472
#  25      0.41459
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict testingSet2_20161215_ffm.txt fulltrainmodel prav_fullmodel2_ffm.csv
#logloss = 0.27077

#####################################################################################################################################
# cd C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm
# ffm-train -p trainingSet03_fold5_ffm.txt -l 0.001 -k 16 -t 30 -r 0.05 -s 30 --auto-stop trainingSet03_fold1to4_ffm.txt validationmodel03
# ffm-predict trainingSet_20161215_valid_fold5_ffm.txt validationmodel prav_ffm_fold5.csv

# ffm-train  -l 0.001 -k 16 -t 25 -r 0.05 -s 30 trainingSet3_20161215_ffm.txt fulltrainmodel
# ffm-predict testingSet3_20161215_ffm.txt fulltrainmodel prav_fullmodel3_ffm.csv
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet03_fold5_ffm.txt -l 0.001 -k 16 -t 30 -r 0.05 -s 30 --auto-stop trainingSet03_fold1to4_ffm.txt validationmodel03
#iter   tr_logloss   va_logloss
#   1      0.43136      0.42719
#   2      0.42579      0.42470
#   3      0.42381      0.42331
#   4      0.42255      0.42233
#   5      0.42163      0.42159
#   6      0.42090      0.42118
#   7      0.42031      0.42056
#   8      0.41980      0.42017
#   9      0.41936      0.41989
#  10      0.41896      0.41959
#  11      0.41861      0.41929
#  12      0.41828      0.41917
#  13      0.41799      0.41891
#  14      0.41771      0.41875
#  15      0.41745      0.41867
#  16      0.41721      0.41843
#  17      0.41698      0.41829
#  18      0.41676      0.41817
#  19      0.41654      0.41808
#  20      0.41634      0.41799
#  21      0.41615      0.41782
#  22      0.41596      0.41768
#  23      0.41577      0.41768
#  24      0.41560      0.41753
#  25      0.41542      0.41757
#Auto-stop. Use model at 24th iteration.

# ffm-train  -l 0.001 -k 16 -t 27 -r 0.05 -s 30 trainingSet3_20161215_ffm.txt fulltrainmodel
# ffm-predict testingSet3_20161215_ffm.txt fulltrainmodel prav_fullmodel3_ffm.csv

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train  -l 0.001 -k 16 -t 27 -r 0.05 -s 30 trainingSet3_20161215_ffm.txt fulltrainmodel
#iter   tr_logloss
#   1      0.43046
#   2      0.42499
#   3      0.42305
#   4      0.42184
#   5      0.42096
#   6      0.42026
#   7      0.41970
#   8      0.41922
#   9      0.41880
#  10      0.41843
#  11      0.41810
#  12      0.41780
#  13      0.41752
#  14      0.41726
#  15      0.41702
#  16      0.41680
#  17      0.41658
#  18      0.41638
#  19      0.41618
#  20      0.41599
#  21      0.41582
#  22      0.41564
#  23      0.41547
#  24      0.41531
#  25      0.41515
#  26      0.41499
#  27      0.41484
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict testingSet3_20161215_ffm.txt fulltrainmodel prav_fullmodel3_ffm.csv
#logloss = 0.27721

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet04_fold5_ffm.txt -l 0.001 -k 16 -t 30 -r 0.05 -s 30 --auto-stop trainingSet04_fold1to4_ffm.txt validationmodel04
#iter   tr_logloss   va_logloss
#   1      0.42637      0.42264
#   2      0.42158      0.42083
#   3      0.42023      0.41987
#   4      0.41942      0.41934
#   5      0.41884      0.41880
#   6      0.41838      0.41860
#   7      0.41801      0.41821
#   8      0.41769      0.41803
#   9      0.41742      0.41784
#  10      0.41716      0.41768
#  11      0.41694      0.41743
#  12      0.41674      0.41736
#  13      0.41655      0.41722
#  14      0.41637      0.41712
#  15      0.41621      0.41705
#  16      0.41605      0.41691
#  17      0.41590      0.41682
#  18      0.41576      0.41679
#  19      0.41562      0.41664
#  20      0.41549      0.41661
#  21      0.41536      0.41653
#  22      0.41525      0.41642
#  23      0.41512      0.41639
#  24      0.41501      0.41633
#  25      0.41490      0.41634
#Auto-stop. Use model at 24th iteration.

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet04_fold5_ffm.txt -l 0.001 -k 16 -t 30 -r 0.05 -s 30 --auto-stop trainingSet04_fold1to4_ffm.txt validationmodel03

#ffm-train -p trainingSet052_fold5_ffm.txt -l 0.001 -k 16 -t 30 -r 0.05 -s 12 --auto-stop trainingSet052_fold1to4_ffm.txt validationmodel05
#ffm-predict trainingSet05_fold5_ffm.txt validationmodel05 trainingSet05_fold5_ffm.csv

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet05_fold5_ffm.txt -l 0.001 -k 16 -t 30 -r 0.05 -s 30 --auto-stop trainingSet05_fold1to4_ffm.txt validationmodel05
#iter   tr_logloss   va_logloss
#   1      0.43035      0.42650
#   2      0.42553      0.42492
#   3      0.42430      0.42389
#   4      0.42355      0.42344
#   5      0.42301      0.42293
#   6      0.42259      0.42273
#   7      0.42225      0.42221
#   8      0.42196      0.42218
#   9      0.42172      0.42187
#  10      0.42150      0.42180
#  11      0.42131      0.42164
#  12      0.42114      0.42152
#  13      0.42098      0.42136
#  14      0.42084      0.42129
#  15      0.42070      0.42113
#  16      0.42058      0.42098
#  17      0.42046      0.42096
#  18      0.42035      0.42093
#  19      0.42025      0.42085
#  20      0.42015      0.42078
#  21      0.42006      0.42071
#  22      0.41997      0.42070
#  23      0.41988      0.42058
#  24      0.41980      0.42060
#Auto-stop. Use model at 23th iteration.

# ffm-train  -l 0.001 -k 16 -t 27 -r 0.05 -s 30 trainingSet5_20161215_ffm.txt fulltrainmodel5
# ffm-predict testingSet5_20161215_ffm.txt fulltrainmodel5 prav_fullmodel5_ffm.csv




#def convert(src_path, dst_path, is_train):
#    with open(dst_path, 'w') as f:
#        for row in csv.DictReader(open(src_path)):
#            i = 1
#            w = 1
#            num = 1
#            feats = []
#               
#            for field in fields:
#                if field == 'publish_dateToDate':
#                    v = hashstr(field)
#                    feats.append('{i}:{v}:{w}'.format(i=i, v=v ,w=row[field]))
#                else:
#                    v = hashstr(field+'-'+row[field])
#                    
#                #print v
#               
#                #feats.append('{i}:{v}:{w}'.format(i=i, v=v ,w=row[field]))
#                    feats.append('{i}:{v}:1'.format(i=i, v=v ))
#                i += 1
#            #print i
#            #print num
# 
#            if is_train == True:
#                f.write('{0} {1}\n'.format(row['clicked'], ' '.join(feats)))
#               
#            num += 1
#            if num % 1000000 == 0:
#                print num
#
#def testconvert(src_path, dst_path, is_train):
#    with open(dst_path, 'w') as f:
#        for row in csv.DictReader(open(src_path)):
#            i = 1
#            w = 1
#            num = 1
#            feats = []
#               
#            for field in fields:
#                if field == 'publish_dateToDate':
#                    v = hashstr(field)
#                    feats.append('{i}:{v}:{w}'.format(i=i, v=v ,w=row[field]))
#                else:
#                    v = hashstr(field+'-'+row[field])
#                    
#                #print v
#               
#                #feats.append('{i}:{v}:{w}'.format(i=i, v=v ,w=row[field]))
#                    feats.append('{i}:{v}:1'.format(i=i, v=v ))
#                i += 1
#            #print i
#            #print num
# 
#            if is_train == True:
#                f.write('{0} {1}\n'.format(0, ' '.join(feats)))
#               
#            num += 1
#            if num % 1000000 == 0:
#                print num

# ffm-train -p trainingSet053_fold5_ffm.txt -l 0.001 -k 16 -t 30 -r 0.05 -s 20 --auto-stop trainingSet053_fold1to4_ffm.txt validationmodel053
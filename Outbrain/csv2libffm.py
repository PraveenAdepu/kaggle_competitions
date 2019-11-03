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
nr_bins = 2 ** 22

def hashstr(str):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1
    

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
            num = 1
            feats = []
               
            for field in fields:
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
          #,u'event_publish_dateToDate',u'publish_dateToDate',u'event_publish_dateTopublishdate'
         
          ,u'traffic_source'#,u'adCount'
           ,u'user_next_document_id',u'user_next_publisher_id'
          ,u'leak']
          

convert('C:/Users/SriPrav/Documents/R/13Outbrain/input/training40_train.csv', 'C:/Users/SriPrav/Documents/R/13Outbrain/input/libffm/trainingSet40_train_ffm.txt', True)
#convert('C:/Users/SriPrav/Documents/R/13Outbrain/input/training40_valid.csv', 'C:/Users/SriPrav/Documents/R/13Outbrain/input/libffm/trainingSet40_valid_ffm.txt', True)

convert('C:/Users/SriPrav/Documents/R/13Outbrain/input/training40.csv', 'C:/Users/SriPrav/Documents/R/13Outbrain/input/libffm/trainingSet40_ffm.txt', True)
#testconvert('C:/Users/SriPrav/Documents/R/13Outbrain/input/testing40.csv', 'C:/Users/SriPrav/Documents/R/13Outbrain/input/libffm/testingSet40_ffm.txt', True)



#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet40_valid_ffm.txt -l 0.00002 -k 9 -t 7 -r 0.05 -s 20 --auto-stop trainingSet40_train_ffm.txt validationmodel40
#  ffm-predict trainingSet40_valid_ffm.txt validationmodel40 prav_validationmodel40_ffm.csv

#ffm-train  -l 0.00002 -k 9 -t 8 -r 0.05 -s 23 trainingSet40_ffm.txt fulltrainmodel40

#  ffm-predict testingSet40_ffm.txt fulltrainmodel40 prav_fullmodel40_ffm.csv




#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet40_valid_ffm.txt -l 0.00002 -k 9 -t 7 -r 0.05 -s 20 --auto-stop trainingSet40_train_ffm.txt validationmodel40
#iter   tr_logloss   va_logloss
#   1      0.40812      0.40778
#   2      0.39903      0.40470
#   3      0.39611      0.40305
#   4      0.39417      0.40194
#   5      0.39266      0.40114
#   6      0.39138      0.40050
#   7      0.39023      0.40007
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict trainingSet40_valid_ffm.txt validationmodel40 prav_validationmodel40_ffm.csv
#logloss = 0.40007








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

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet052_fold5_ffm.txt -l 0.001 -k 16 -t 30 -r 0.05 -s 12 --auto-stop trainingSet052_fold1to4_ffm.txt validationmodel05
#iter   tr_logloss   va_logloss
#   1      0.43002      0.42583
#   2      0.42447      0.42335
#   3      0.42246      0.42210
#   4      0.42116      0.42106
#   5      0.42021      0.42032
#   6      0.41945      0.41980
#   7      0.41883      0.41928
#   8      0.41829      0.41894
#   9      0.41782      0.41861
#  10      0.41740      0.41832
#  11      0.41702      0.41805
#  12      0.41666      0.41778
#  13      0.41634      0.41760
#  14      0.41603      0.41753
#  15      0.41574      0.41725
#  16      0.41547      0.41705
#  17      0.41520      0.41695
#  18      0.41495      0.41679
#  19      0.41471      0.41678
#  20      0.41447      0.41665
#  21      0.41424      0.41642
#  22      0.41402      0.41635
#  23      0.41380      0.41624
#  24      0.41358      0.41619
#  25      0.41337      0.41612
#  26      0.41316      0.41606
#  27      0.41295      0.41586
#  28      0.41275      0.41589
#Auto-stop. Use model at 27th iteration.

# ffm-train  -l 0.001 -k 16 -t 30 -r 0.05 -s 12 trainingSet52_20161215_ffm.txt fulltrainmodel52
# ffm-predict testingSet52_20161215_ffm.txt fulltrainmodel52 prav_fullmodel52_ffm.csv

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train  -l 0.001 -k 16 -t 30 -r 0.05 -s 12 trainingSet52_20161215_ffm.txt fulltrainmodel52
#iter   tr_logloss
#   1      0.42913
#   2      0.42366
#   3      0.42170
#   4      0.42045
#   5      0.41954
#   6      0.41881
#   7      0.41822
#   8      0.41771
#   9      0.41727
#  10      0.41688
#  11      0.41652
#  12      0.41619
#  13      0.41588
#  14      0.41559
#  15      0.41533
#  16      0.41507
#  17      0.41483
#  18      0.41459
#  19      0.41436
#  20      0.41414
#  21      0.41393
#  22      0.41372
#  23      0.41352
#  24      0.41333
#  25      0.41313
#  26      0.41294
#  27      0.41275
#  28      0.41256
#  29      0.41238
#  30      0.41219
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict testingSet52_20161215_ffm.txt fulltrainmodel52 prav_fullmodel52_ffm.csv
#logloss = 0.27912

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet053_fold5_ffm.txt -l 0.001 -k 16 -t 30 -r 0.05 -s 20 --auto-stop trainingSet053_fold1to4_ffm.txt validationmodel053

#iter   tr_logloss   va_logloss
#   1      0.42491      0.42127
#   2      0.42022      0.41941
#   3      0.41887      0.41862
#   4      0.41806      0.41813
#   5      0.41746      0.41762
#   6      0.41699      0.41724
#   7      0.41661      0.41681
#   8      0.41626      0.41667
#   9      0.41597      0.41646
#  10      0.41571      0.41623
#  11      0.41546      0.41622
#  12      0.41524      0.41600
#  13      0.41503      0.41585
#  14      0.41483      0.41578
#  15      0.41464      0.41568
#  16      0.41447      0.41551
#  17      0.41430      0.41544
#  18      0.41414      0.41540
#  19      0.41398      0.41529
#  20      0.41383      0.41526
#  21      0.41367      0.41511
#  22      0.41353      0.41504
#  23      0.41339      0.41500
#  24      0.41325      0.41495
#  25      0.41312      0.41484
#  26      0.41298      0.41491
#Auto-stop. Use model at 25th iteration.

# ffm-train  -l 0.001 -k 16 -t 28 -r 0.05 -s 25 trainingSet53_20161215_ffm.txt fulltrainmodel53
# ffm-predict testingSet53_20161215_ffm.txt fulltrainmodel53 prav_fullmodel53_ffm.csv

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train  -l 0.001 -k 16 -t 28 -r 0.05 -s 25 trainingSet53_20161215_ffm.txt fulltrainmodel53
#iter   tr_logloss
#   1      0.42412
#   2      0.41967
#   3      0.41840
#   4      0.41762
#   5      0.41705
#   6      0.41661
#   7      0.41624
#   8      0.41592
#   9      0.41564
#  10      0.41538
#  11      0.41515
#  12      0.41494
#  13      0.41475
#  14      0.41456
#  15      0.41439
#  16      0.41422
#  17      0.41406
#  18      0.41391
#  19      0.41376
#  20      0.41362
#  21      0.41348
#  22      0.41334
#  23      0.41322
#  24      0.41309
#  25      0.41296
#  26      0.41284
#  27      0.41272
#  28      0.41260
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict testingSet53_20161215_ffm.txt fulltrainmodel53 prav_fullmodel53_ffm.csv
#logloss = 0.27350

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet054_fold5_ffm.txt -l 0.001 -k 18 -t 30 -r 0.05 -s 24 --auto-stop trainingSet054_fold1to4_ffm.txt validationmodel054


#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet053_fold5_ffm.txt -l 0.0002 -k 16 -t 30 -r 0.05 -s 24 --auto-stop trainingSet053_fold1to4_ffm.txt validationmodel053
#iter   tr_logloss   va_logloss
#   1      0.41751      0.41310
#   2      0.41162      0.41077
#   3      0.40968      0.40953
#   4      0.40841      0.40862
#   5      0.40742      0.40802
#   6      0.40661      0.40748
#   7      0.40589      0.40707
#   8      0.40524      0.40669
#   9      0.40464      0.40641
#  10      0.40407      0.40610
#  11      0.40353      0.40590
#  12      0.40299      0.40568
#  13      0.40247      0.40548
#  14      0.40194      0.40535
#  15      0.40142      0.40517
#  16      0.40089      0.40503
#  17      0.40035      0.40491
#  18      0.39979      0.40481
#  19      0.39922      0.40471
#  20      0.39862      0.40464
#  21      0.39800      0.40456
#  22      0.39735      0.40447
#  23      0.39666      0.40442
#  24      0.39594      0.40438
#  25      0.39517      0.40437
#  26      0.39436      0.40434
#  27      0.39349      0.40432
#  28      0.39257      0.40434
#Auto-stop. Use model at 27th iteration.

# ffm-train  -l 0.00002 -k 16 -t 12 -r 0.05 -s 25 trainingSet53_20161215_ffm.txt fulltrainmodel53
# ffm-predict testingSet53_20161215_ffm.txt fulltrainmodel53 prav_fullmodel532_ffm.csv

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train  -l 0.0002 -k 16 -t 30 -r 0.05 -s 25 trainingSet53_20161215_ffm.txt fulltrainmodel53
#iter   tr_logloss
#   1      0.41655
#   2      0.41088
#   3      0.40901
#   4      0.40778
#   5      0.40683
#   6      0.40605
#   7      0.40536
#   8      0.40473
#   9      0.40416
#  10      0.40361
#  11      0.40308
#  12      0.40257
#  13      0.40207
#  14      0.40157
#  15      0.40107
#  16      0.40056
#  17      0.40004
#  18      0.39951
#  19      0.39896
#  20      0.39839
#  21      0.39780
#  22      0.39717
#  23      0.39652
#  24      0.39582
#  25      0.39509
#  26      0.39431
#  27      0.39349
#  28      0.39260
#  29      0.39167
#  30      0.39067
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict testingSet53_20161215_ffm.txt fulltrainmodel53 prav_fullmodel531_ffm.csv
#logloss = 0.27597

#ffm-train -p trainingSet053_fold5_ffm.txt -l 0.00002 -k 16 -t 30 -r 0.05 -s 24 --auto-stop trainingSet053_fold1to4_ffm.txt validationmodel053

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet053_fold5_ffm.txt -l 0.00002 -k 16 -t 30 -r 0.05 -s 24 --auto-stop trainingSet053_fold1to4_ffm.txt validationmodel053
#iter   tr_logloss   va_logloss
#   1      0.41386      0.40956
#   2      0.40779      0.40711
#   3      0.40544      0.40582
#   4      0.40376      0.40496
#   5      0.40235      0.40436
#   6      0.40108      0.40392
#   7      0.39986      0.40358
#   8      0.39864      0.40332
#   9      0.39739      0.40315
#  10      0.39604      0.40303
#  11      0.39457      0.40298
#  12      0.39293      0.40301
#Auto-stop. Use model at 11th iteration.
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train  -l 0.00002 -k 16 -t 12 -r 0.05 -s 25 trainingSet53_20161215_ffm.txt fulltrainmodel53
#iter   tr_logloss
#   1      0.41292
#   2      0.40702
#   3      0.40474
#   4      0.40311
#   5      0.40175
#   6      0.40053
#   7      0.39935
#   8      0.39817
#   9      0.39693
#  10      0.39561
#  11      0.39415
#  12      0.39251
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict testingSet53_20161215_ffm.txt fulltrainmodel53 prav_fullmodel532_ffm.csv
#logloss = 0.28249

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet11_fold5_ffm.txt -l 0.00002 -k 18 -t 30 -r 0.05 -s 24 --auto-stop trainingSet11_fold1to4_ffm.txt validationmodel11
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train  -l 0.00002 -k 18 -t 13 -r 0.05 -s 25 trainingSet11_ffm.txt fulltrainmodel11
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict testingSet11_ffm.txt fulltrainmodel11 prav_fullmodel11_ffm.csv

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict testingSet11_ffm.txt fulltrainmodel11 prav_fullmodel11_ffm.csv
#logloss = 0.28058
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet12_fold5_ffm.txt -l 0.00002 -k 20 -t 30 -r 0.05 -s 24 --auto-stop trainingSet12_fold1to4_ffm.txt validationmodel12
#iter   tr_logloss   va_logloss
#   1      0.41397      0.40935
#   2      0.40746      0.40677
#   3      0.40503      0.40544
#   4      0.40332      0.40457
#   5      0.40191      0.40396
#   6      0.40066      0.40351
#   7      0.39949      0.40316
#   8      0.39836      0.40291
#   9      0.39724      0.40273
#  10      0.39610      0.40260
#  11      0.39491      0.40250
#  12      0.39365      0.40246
#  13      0.39229      0.40247
#Auto-stop. Use model at 12th iteration.

#ffm-train -p trainingSet122_fold5_ffm.txt -l 0.00002 -k 24 -t 30 -r 0.05 -s 24 --auto-stop trainingSet122_fold1to4_ffm.txt validationmodel12

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet12_fold5_ffm.txt -l 0.00002 -k 20 -t 30 -r 0.05 -s 24 --auto-stop trainingSet12_fold1to4_ffm.txt validationmodel12
#iter   tr_logloss   va_logloss
#   1      0.41394      0.40927
#   2      0.40746      0.40674
#   3      0.40505      0.40543
#   4      0.40335      0.40454
#   5      0.40195      0.40392
#   6      0.40071      0.40346
#   7      0.39955      0.40309
#   8      0.39842      0.40282
#   9      0.39728      0.40262
#  10      0.39611      0.40247
#  11      0.39488      0.40235
#  12      0.39355      0.40229
#  13      0.39207      0.40227
#  14      0.39040      0.40229
#Auto-stop. Use model at 13th iteration.

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet122_fold5_ffm.txt -l 0.00002 -k 24 -t 30 -r 0.05 -s 24 --auto-stop trainingSet122_fold1to4_ffm.txt validationmodel12
#iter   tr_logloss   va_logloss
#   1      0.41414      0.40952
#   2      0.40774      0.40700
#   3      0.40539      0.40565
#   4      0.40372      0.40473
#   5      0.40235      0.40406
#   6      0.40112      0.40357
#   7      0.39996      0.40315
#   8      0.39882      0.40284
#   9      0.39768      0.40260
#  10      0.39648      0.40241
#  11      0.39523      0.40227
#  12      0.39388      0.40217
#  13      0.39239      0.40211
#  14      0.39073      0.40209
#  15      0.38884      0.40212
#Auto-stop. Use model at 14th iteration.

# ffm-predict trainingSet122_fold5_ffm.txt validationmodel12 prav_validationmodel12_ffm.csv

# ffm-train  -l 0.00002 -k 24 -t 16 -r 0.05 -s 25 trainingSet12_ffm.txt fulltrainmodel12
# ffm-predict testingSet12_ffm.txt fulltrainmodel12 prav_fullmodel12_ffm.csv

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train  -l 0.00002 -k 24 -t 16 -r 0.05 -s 25 trainingSet12_ffm.txt fulltrainmodel12
#iter   tr_logloss
#   1      0.41309
#   2      0.40686
#   3      0.40455
#   4      0.40292
#   5      0.40158
#   6      0.40037
#   7      0.39923
#   8      0.39811
#   9      0.39697
#  10      0.39580
#  11      0.39456
#  12      0.39322
#  13      0.39176
#  14      0.39015
#  15      0.38832
#  16      0.38621
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict testingSet12_ffm.txt fulltrainmodel12 prav_fullmodel12_ffm.csv
#logloss = 0.28255

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet123_fold5_ffm.txt -l 0.00002 -k 27 -t 30 -r 0.05 -s 24 --auto-stop trainingSet123_fold1to4_ffm.txt validationmodel13

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet13_fold5_ffm.txt -l 0.00002 -k 27 -t 30 -r 0.05 -s 24 --auto-stop trainingSet13_fold1to4_ffm.txt validationmodel13
#iter   tr_logloss   va_logloss
#   1      0.41455      0.40991
#   2      0.40806      0.40729
#   3      0.40569      0.40591
#   4      0.40404      0.40497
#   5      0.40269      0.40429
#   6      0.40150      0.40376
#   7      0.40039      0.40333
#   8      0.39931      0.40299
#   9      0.39823      0.40272
#  10      0.39714      0.40250
#  11      0.39600      0.40231
#  12      0.39479      0.40218
#  13      0.39349      0.40208
#  14      0.39207      0.40203
#  15      0.39047      0.40199
#  16      0.38866      0.40199
#Auto-stop. Use model at 15th iteration.
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train  -l 0.00002 -k 25 -t 18 -r 0.05 -s 25 trainingSet20_ffm.txt fulltrainmodel20

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train  -l 0.00002 -k 27 -t 17 -r 0.05 -s 25 trainingSet13_ffm.txt fulltrainmodel13
#iter   tr_logloss
#   1      0.41350
#   2      0.40718
#   3      0.40486
#   4      0.40325
#   5      0.40193
#   6      0.40076
#   7      0.39967
#   8      0.39862
#   9      0.39756
#  10      0.39648
#  11      0.39536
#  12      0.39417
#  13      0.39289
#  14      0.39149
#  15      0.38993
#  16      0.38818
#  17      0.38617
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict testingSet13_ffm.txt fulltrainmodel13 prav_fullmodel13_ffm.csv
#logloss = 0.28096
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet20_train_ffm.txt -l 0.00002 -k 24 -t 30 -r 0.05 -s 24 --auto-stop trainingSet20_valid_ffm.txt validationmodel20

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet202_train_ffm.txt -l 0.00002 -k 25 -t 30 -r 0.05 -s 24 --auto-stop trainingSet202_valid_ffm.txt validationmodel202
#iter   tr_logloss   va_logloss
#   1      0.42034      0.41471
#   2      0.41286      0.41160
#   3      0.41002      0.40998
#   4      0.40805      0.40888
#   5      0.40643      0.40809
#   6      0.40500      0.40748
#   7      0.40365      0.40700
#   8      0.40236      0.40664
#   9      0.40107      0.40632
#  10      0.39977      0.40610
#  11      0.39845      0.40591
#  12      0.39706      0.40580
#  13      0.39561      0.40571
#  14      0.39408      0.40568
#  15      0.39244      0.40570
#Auto-stop. Use model at 14th iteration.
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict trainingSet202_valid_ffm.txt validationmodel202 prav_validationmodel202_ffm.csv
#logloss = 0.39208
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet202_train_ffm.txt -l 0.00002 -k 25 -t 30 -r 0.05 -s 24 --auto-stop trainingSet202_valid_ffm.txt validationmodel202
#iter   tr_logloss   va_logloss
#   1      0.42089      0.41561
#   2      0.41379      0.41287
#   3      0.41112      0.41138
#   4      0.40922      0.41032
#   5      0.40766      0.40954
#   6      0.40628      0.40894
#   7      0.40499      0.40847
#   8      0.40374      0.40806
#   9      0.40251      0.40772
#  10      0.40126      0.40749
#  11      0.39998      0.40724
#  12      0.39865      0.40708
#  13      0.39725      0.40693
#  14      0.39578      0.40685
#  15      0.39420      0.40680
#  16      0.39251      0.40679
#  17      0.39069      0.40683
#Auto-stop. Use model at 16th iteration.

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict trainingSet2021_valid_ffm.txt validationmodel202 prav_validationmodel2021_ffm.csv
#logloss = 0.39037
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict trainingSet122_fold5_ffm.txt validationmodel12 prav_validationmodel12_ffm.csv
#logloss = 0.64999
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict trainingSet2021_valid_ffm.txt validationmodel202 prav_validationmodel2021_ffm.csv
#logloss = 0.26120
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train  -l 0.00002 -k 25 -t 18 -r 0.05 -s 25 trainingSet20_ffm.txt fulltrainmodel20

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train  -l 0.00002 -k 25 -t 18 -r 0.05 -s 25 trainingSet20_ffm.txt fulltrainmodel20
#iter   tr_logloss
#   1      0.41415
#   2      0.40816
#   3      0.40587
#   4      0.40423
#   5      0.40287
#   6      0.40163
#   7      0.40046
#   8      0.39931
#   9      0.39815
#  10      0.39697
#  11      0.39573
#  12      0.39441
#  13      0.39300
#  14      0.39145
#  15      0.38975
#  16      0.38783
#  17      0.38566
#  18      0.38315
#
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict testingSet20_ffm.txt fulltrainmodel20 prav_fulltrain20_ffm.csv

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet30_valid_ffm.txt -l 0.00002 -k 8 -t 30 -r 0.05 -s 24 --auto-stop trainingSet30_train_ffm.txt validationmodel30
#iter   tr_logloss   va_logloss
#   1      0.42184      0.41627
#   2      0.41438      0.41358
#   3      0.41167      0.41215
#   4      0.40975      0.41110
#   5      0.40816      0.41036
#   6      0.40675      0.40979
#   7      0.40545      0.40932
#   8      0.40420      0.40897
#   9      0.40298      0.40866
#  10      0.40177      0.40844
#  11      0.40055      0.40825
#  12      0.39930      0.40812
#  13      0.39804      0.40801
#  14      0.39673      0.40797
#  15      0.39537      0.40796
#  16      0.39396      0.40799
#Auto-stop. Use model at 15th iteration.

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict trainingSet301_valid_ffm.txt validationmodel301 prav_validation30_ffm.csv

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet202_valid_ffm.txt -l 0.00002 -k 25 -t 30 -r 0.05 -s 24 --auto-stop trainingSet202_train_ffm.txt validationmodel202

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet202_valid_ffm.txt -l 0.00002 -k 25 -t 30 -r 0.05 -s 24 --auto-stop trainingSet202_train_ffm.txt validationmodel20222
#iter   tr_logloss   va_logloss
#   1      0.41451      0.41575
#   2      0.40832      0.41347
#   3      0.40596      0.41225
#   4      0.40428      0.41150
#   5      0.40289      0.41091
#   6      0.40164      0.41049
#   7      0.40045      0.41005
#   8      0.39927      0.40979
#   9      0.39809      0.40955
#  10      0.39688      0.40939
#  11      0.39562      0.40924
#  12      0.39428      0.40915
#  13      0.39286      0.40911
#  14      0.39132      0.40910
#  15      0.38964      0.40913
#Auto-stop. Use model at 14th iteration.
#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-predict trainingSet202_valid_ffm.txt validationmodel20222 prav_validation20222_ffm.csv

#C:\Users\SriPrav\Documents\R\13Outbrain\input\libffm>ffm-train -p trainingSet21_valid_ffm.txt -l 0.00002 -k 9 -t 30 -r 0.05 -s 12 --auto-stop trainingSet21_train_ffm.txt validationmodel21


# ffm-predict trainingSet21_valid_ffm.txt validationmodel21 prav_validation21_ffm.csv

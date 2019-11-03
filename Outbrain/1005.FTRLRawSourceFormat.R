training <- fread("./input/trainingSet20_train.csv")

head(training)
sampletraining <-   training[sample.int(nrow(training)) ]
head(sampletraining)
write_csv(sampletraining, "./input/trainingSet20_train_FTRL.csv")

# del row['display_id']
# del row['event_entity_id'] 
# del row['entity_id']
# del row['day']
# del row['minutes']
# del row['event_publish_dateToDate']
# del row['publish_dateToDate']
# del row['event_publish_dateTopublishdate']
# del row['traffic_source']
# del row['weekday']     
# del row['event_LastCat_id']
# del row['event_Lasttopic_id']
# del row['LastCat_id']
# del row['Lasttopic_id']
# del row['event_Entconf']
# del row['Entconf']


# [1] "display_id"                      "event_entity_id"                 "entity_id"                      
# [5] "day"                             "minutes"                         "event_publish_dateToDate"        "publish_dateToDate"             
# [9] "event_publish_dateTopublishdate" "traffic_source"                  "weekday"                         "event_LastCat_id"               
# [13] "event_Lasttopic_id"              "LastCat_id"                      "Lasttopic_id"                    "event_Entconf"                  
# [17] "Entconf"  


fields = c('ad_id','uuid','event_document_id','platform'
          ,'event_source_id','event_publisher_id' ,'event_category_id', 'event_topic_id'                 
          ,'document_id','campaign_id','advertiser_id','source_id','publisher_id'         
          ,'location1','location2','location3'
          ,'event_Catconf','event_topconf'
          ,'hour'
          , 'category_id','topic_id'
          ,'Catconf','topconf'
          ,'weekflag'
          ,'leak')

delcols <- setdiff(names(sampletraining), fields)
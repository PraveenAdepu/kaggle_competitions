# Sys.time()
# load("Outbrain_Baseline01_20161216.RData"); gc()
# Sys.time()

head(trainfile)

length(unique(trainfile$campaign_id))

rm(document_category_features, document_entities_features, document_topics_features); gc()
names(train_test)
#########################################################################################################################
ReqNames <- c( "display_id"       , "ad_id"          ,   "clicked"       ,    "uuid"         ,     "document_id"  ,     "platform"     ,     "campaign_id",      
               "advertiser_id"    , "source_id"       ,  "publisher_id"  ,    "location1"    ,     "location2"     ,    "location3"    ,     "hour" ,            
               "day"              , "minutes")


tReqNames <- c( "display_id"       , "ad_id"               ,    "uuid"         ,     "document_id"  ,     "platform"     ,     "campaign_id",      
               "advertiser_id"    , "source_id"       ,  "publisher_id"  ,    "location1"    ,     "location2"     ,    "location3"    ,     "hour" ,            
               "day"              , "minutes")

training <- trainfile[, ReqNames]
testing  <- testfile[, tReqNames]

#########################################################################################################################  
trainLeak <- read_csv("./input/trainingSet_Leak.csv")
testLeak <- read_csv("./input/testingSet_Leak.csv")

trainLeak$leak <- gsub("\\[|\\'|\\]","", trainLeak$leak)
testLeak$leak <- gsub("\\[|\\'|\\]","", testLeak$leak)

unique(trainLeak$leak)
unique(testLeak$leak)

training <- left_join(training, trainLeak, by = c("display_id","ad_id"))
testing  <- left_join(testing, testLeak, by = c("display_id","ad_id"))

unique(training$leak)
unique(testing$leak)

#########################################################################################################################  

#########################################################################################################################

Prav_CVindices_5folds  <- read_csv("./CVSchema/Prav_CVindices_5folds.csv") 
names(Prav_CVindices_5folds)
Prav_CVindices_5folds$day <- NULL

#########################################################################################################################
# train -- 87,141,731
training <- left_join(training, Prav_CVindices_5folds, by = "display_id")

names(training)
i = 5

X_build <- subset(training, CVindices != i, select = -c( CVindices))
X_val   <- subset(training, CVindices == i, select = -c( CVindices)) 

write.csv(X_build,  './input/trainingSet_20161215_train2_fold1to4.csv', row.names=FALSE, quote = FALSE)
write.csv(X_val, './input/trainingSet_20161215_valid2_fold5.csv', row.names=FALSE, quote = FALSE)
#########################################################################################################################

training$CVindices <- NULL

gc()

names(training)

write.csv(training,  './input/trainingSet2_20161215.csv', row.names=FALSE, quote = FALSE)
write.csv(testing,  './input/testingSet2_20161215.csv', row.names=FALSE, quote = FALSE)


#########################################################################################################################


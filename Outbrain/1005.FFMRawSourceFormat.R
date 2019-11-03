
training <- read_csv("./input/trainingSet12.csv")
#87,141,731

# sapply(training, class)

# display_id         "integer"   
# ad_id              "integer"  
# clicked            "numeric"                         
# uuid               "character"             
# event_document_id  "integer" 
# platform           "numeric"                
# event_source_id    "numeric"             
# event_publisher_id "numeric"              
# event_category_id  "numeric"                
# event_entity_id    "character" 
# event_topic_id     "numeric"                   
# document_id        "integer"                   
# campaign_id        "integer"                  
# advertiser_id      "integer"                     
# source_id          "numeric"  
# publisher_id       "numeric"                    
# category_id        "numeric"                      
# entity_id          "character"                       
# topic_id           "numeric"                      
# location1          "character" 
# location2          "character"                      
# location3          "integer"                          
# hour               "numeric"                            
# day                "integer"                       
# minutes            "numeric"  
# event_publish_dateToDate         "numeric"            
# publish_dateToDate               "numeric" 
# event_publish_dateTopublishdate  "numeric"                          
# leak                             "character"                  
# traffic_source                   "integer"  
# weekday            "character"                     
# weekflag           "character"              
# event_LastCat_id   "numeric"            
# event_Lasttopic_id "numeric"                      
# LastCat_id         "numeric" 
# Lasttopic_id       "numeric"                 
# event_Catconf      "numeric"                
# event_Entconf      "numeric"                 
# event_topconf      "numeric"                         
# Catconf            "numeric"  
# Entconf            "numeric"                        
# topconf            "numeric"

# convert to interger:    as.integer( factor( confidence ) )
# convert rounded to integer:   as.integer( factor( round(confidence, digits=2) ) )


training$uuid               <- as.integer( as.factor( training$uuid ) )
training$platform           <- as.integer( as.factor( training$platform ) )
training$event_source_id    <- as.integer( as.factor( training$event_source_id ) )
training$event_publisher_id <- as.integer( as.factor( training$event_publisher_id ) )
training$event_category_id  <- as.integer( as.factor( training$event_category_id ) )
training$event_entity_id    <- as.integer( as.factor( training$event_entity_id ) )
training$event_topic_id     <- as.integer( as.factor(training$event_topic_id                  ))
training$entity_id          <- as.integer( as.factor( training$entity_id ) )
training$location1          <- as.integer( as.factor( training$location1 ) )
training$location2          <- as.integer( as.factor( training$location2 ) )
training$leak               <- as.integer( as.factor( training$leak ) )
training$weekday            <- as.integer( as.factor( training$weekday ) )
training$weekflag           <- as.integer( as.factor( training$weekflag ) )


training$source_id                          <- as.integer( as.factor(training$source_id                       ))
training$publisher_id                       <- as.integer( as.factor(training$publisher_id                    ))
training$category_id                        <- as.integer( as.factor(training$category_id                     ))
training$entity_id                          <- as.integer( as.factor(training$entity_id                       ))   
training$topic_id                           <- as.integer( as.factor(training$topic_id                        ))
training$location1                          <- as.integer( as.factor(training$location1                       ))
training$location2                          <- as.integer( as.factor(training$location2                       ))  
training$hour                               <- as.integer( as.factor(training$hour                            ))      
training$minutes                            <- as.integer( as.factor(training$minutes                         ))
training$event_publish_dateToDate           <- as.integer( as.factor(training$event_publish_dateToDate        ))    
training$publish_dateToDate                 <- as.integer( as.factor(training$publish_dateToDate              ))
training$event_publish_dateTopublishdate    <- as.integer( as.factor(training$event_publish_dateTopublishdate ))                  
training$leak                               <- as.integer( as.factor(training$leak                            ))            
training$weekday                            <- as.integer( as.factor(training$weekday                         ))
training$weekflag                           <- as.integer( as.factor(training$weekflag                        ))
training$event_LastCat_id                   <- as.integer( as.factor(training$event_LastCat_id                ))
training$event_Lasttopic_id                 <- as.integer( as.factor(training$event_Lasttopic_id              ))
training$LastCat_id                         <- as.integer( as.factor(training$LastCat_id                      ))
training$Lasttopic_id                       <- as.integer( as.factor(training$Lasttopic_id                    ))
training$event_Catconf                      <- as.integer( as.factor(training$event_Catconf                   ))
training$event_Entconf                      <- as.integer( as.factor(training$event_Entconf                   ))
training$event_topconf                      <- as.integer( as.factor(training$event_topconf                   ))  
training$Catconf                            <- as.integer( as.factor(training$Catconf                         ))
training$Entconf                            <- as.integer( as.factor(training$Entconf                         )) 
training$topconf                            <- as.integer( as.factor(training$topconf                         ))



write_csv(training,"./input/trainingSet30.csv")


Prav_CVindices_5folds  <- fread("./CVSchema/splits.csv") 

head(Prav_CVindices_5folds)

unique(Prav_CVindices_5folds$is_train)

table(Prav_CVindices_5folds$is_train)

#training <- read_csv("./input/trainingSet12.csv")

training <- left_join(training, Prav_CVindices_5folds, by = "display_id")

#63,502,376
X_build  <- training[training$is_train == 1,]
#23,639,355
X_valid  <- training[training$is_train == 0,]

X_build$is_train <- NULL
X_valid$is_train <- NULL

write_csv(X_build, "./input/trainingSet30_train.csv")
write_csv(X_valid, "./input/trainingSet30_valid.csv")


############################################################################################################################

training <- read_csv("./input/trainingSet12.csv")
testing  <- read_csv("./input/testingSet12.csv")

names(training)

training$clicked1 <- training$clicked
training$clicked  <- NULL
names(training)[names(training)=="clicked1"] <- "clicked"
names(testing)
testing$clicked      <- -1

training$is_trainSet <- 1
testing$is_trainSet  <- 0

train_test <- rbind(training, testing)

rm(training, testing); gc()

train_test$uuid               <- as.integer( as.factor( train_test$uuid ) )
train_test$platform           <- as.integer( as.factor( train_test$platform ) )
train_test$event_source_id    <- as.integer( as.factor( train_test$event_source_id ) )
train_test$event_publisher_id <- as.integer( as.factor( train_test$event_publisher_id ) )
train_test$event_category_id  <- as.integer( as.factor( train_test$event_category_id ) )
train_test$event_entity_id    <- as.integer( as.factor( train_test$event_entity_id ) )
train_test$event_topic_id     <- as.integer( as.factor(train_test$event_topic_id                  ))
train_test$entity_id          <- as.integer( as.factor( train_test$entity_id ) )
train_test$location1          <- as.integer( as.factor( train_test$location1 ) )
train_test$location2          <- as.integer( as.factor( train_test$location2 ) )
train_test$leak               <- as.integer( as.factor( train_test$leak ) )
train_test$weekday            <- as.integer( as.factor( train_test$weekday ) )
train_test$weekflag           <- as.integer( as.factor( train_test$weekflag ) )


train_test$source_id                          <- as.integer( as.factor(train_test$source_id                       ))
train_test$publisher_id                       <- as.integer( as.factor(train_test$publisher_id                    ))
train_test$category_id                        <- as.integer( as.factor(train_test$category_id                     ))
train_test$entity_id                          <- as.integer( as.factor(train_test$entity_id                       ))   
train_test$topic_id                           <- as.integer( as.factor(train_test$topic_id                        ))
train_test$location1                          <- as.integer( as.factor(train_test$location1                       ))
train_test$location2                          <- as.integer( as.factor(train_test$location2                       ))  
train_test$hour                               <- as.integer( as.factor(train_test$hour                            ))      
train_test$minutes                            <- as.integer( as.factor(train_test$minutes                         ))
train_test$event_publish_dateToDate           <- as.integer( as.factor(train_test$event_publish_dateToDate        ))    
train_test$publish_dateToDate                 <- as.integer( as.factor(train_test$publish_dateToDate              ))
train_test$event_publish_dateTopublishdate    <- as.integer( as.factor(train_test$event_publish_dateTopublishdate ))                  
train_test$leak                               <- as.integer( as.factor(train_test$leak                            ))            
train_test$weekday                            <- as.integer( as.factor(train_test$weekday                         ))
train_test$weekflag                           <- as.integer( as.factor(train_test$weekflag                        ))
train_test$event_LastCat_id                   <- as.integer( as.factor(train_test$event_LastCat_id                ))
train_test$event_Lasttopic_id                 <- as.integer( as.factor(train_test$event_Lasttopic_id              ))
train_test$LastCat_id                         <- as.integer( as.factor(train_test$LastCat_id                      ))
train_test$Lasttopic_id                       <- as.integer( as.factor(train_test$Lasttopic_id                    ))
train_test$event_Catconf                      <- as.integer( as.factor(train_test$event_Catconf                   ))
train_test$event_Entconf                      <- as.integer( as.factor(train_test$event_Entconf                   ))
train_test$event_topconf                      <- as.integer( as.factor(train_test$event_topconf                   ))  
train_test$Catconf                            <- as.integer( as.factor(train_test$Catconf                         ))
train_test$Entconf                            <- as.integer( as.factor(train_test$Entconf                         )) 
train_test$topconf                            <- as.integer( as.factor(train_test$topconf                         ))


train  <- train_test[train_test$is_trainSet == 1,]

test  <- train_test[train_test$is_trainSet == 0,]

train$is_trainSet <- NULL
test$is_trainSet  <- NULL

test$clicked <- NULL

gc()

write_csv(train,"./input/trainingSet301.csv")
write_csv(test,"./input/testingSet301.csv")


names(testfile)

#####################################################################################################################################

X_train <- read_csv("./input/trainingSet30_train.csv")
X_valid <- read_csv("./input/trainingSet30_valid.csv")
trainadCount  <- read_csv("./input/trainadCountFeatures.csv")

#63,502,376
#23,639,355

X_train <- left_join(X_train, trainadCount, by = c("display_id","ad_id"))
X_valid <- left_join(X_valid, trainadCount, by = c("display_id","ad_id"))

length(unique(X_train$adCount))

write_csv(X_train,"./input/training30_train.csv")
write_csv(X_valid,"./input/training30_valid.csv")

#######################################################################################################################################
training <- read_csv("./input/trainingSet301.csv")
testing <- read_csv("./input/testingSet301.csv")
trainadCount  <- read_csv("./input/trainadCountFeatures.csv")
testadCount  <- read_csv("./input/testadCountFeatures.csv")

training <- left_join(training, trainadCount, by = c("display_id","ad_id"))
testing <- left_join(testing, testadCount, by = c("display_id","ad_id"))

write_csv(training,"./input/training30.csv")
write_csv(testing,"./input/testing30.csv")

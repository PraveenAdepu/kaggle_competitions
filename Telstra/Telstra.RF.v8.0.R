# v2.0 In Progress
# 01. Libraries

require(caret)
require(corrplot)
#require(Rtsne)
require(xgboost)
require(stats)
require(knitr)
require(ggplot2)
knitr::opts_chunk$set(cache=TRUE)
require(DiagrammeR)
require(plyr)
require(dplyr)
require(sqldf)
require(reshape)
require(tidyr)
require(gbm)

#rm(list=ls())
setwd("C:/Users/padepu/Documents/R")

# 02. Set Seed
# you must know why I am using set.seed()
set.seed(546)

# 03. Import source files data
# Importing data into R
train       <- read.csv("./Telstra/train.csv"        , h=TRUE, sep=",")
test        <- read.csv("./Telstra/test.csv"         , h=TRUE, sep=",")
event       <- read.csv("./Telstra/event_type.csv"   , h=TRUE, sep=",")
log         <- read.csv("./Telstra/log_feature.csv"  , h=TRUE, sep=",")
resource    <- read.csv("./Telstra/resource_type.csv", h=TRUE, sep=",")
severity    <- read.csv("./Telstra/severity_type.csv", h=TRUE, sep=",")

# head(train,2)
# head(test ,2)

# 04. Set target variable to test data
test$fault_severity <- -1

df_all  <-  rbind(train,test)

# head(df_all,2)

# merging data
sessionsdata    <- merge(event,log,by="id"      ,all = T)  
sessionsdata    <- merge(sessionsdata,resource,by="id" ,all = T)
sessionsdata    <- merge(sessionsdata,severity,by="id" ,all = T)
df_all_combined <- merge(df_all,sessionsdata,by="id" ,all = T)

df_all_combinedForMovings <- df_all_combined

df_all_combinedForMovings$event_type <- as.integer(gsub("event_type ","",df_all_combinedForMovings$event_type))
df_all_combinedForMovings$log_feature <- as.integer(gsub("feature ","",df_all_combinedForMovings$log_feature))
df_all_combinedForMovings$resource_type <- as.integer(gsub("resource_type ","",df_all_combinedForMovings$resource_type))
df_all_combinedForMovings$severity_type <- as.integer(gsub("severity_type ","",df_all_combinedForMovings$severity_type))

# head(df_all_combinedForMovings,2)

Movings <- sqldf("SELECT id, location, MAX(event_type) Maxevent_type, 
                 MIN(event_type) Minevent_type,
                 (MAX(event_type) - MIN(event_type) ) Diffevent_type,
                 MAX(log_feature) Maxlog_feature, 
                 MIN(log_feature) Minlog_feature,
                 (MAX(log_feature) - MIN(log_feature) ) Difflog_feature,
                 MAX(resource_type) Maxresource_type, 
                 MIN(resource_type) Minresource_type,
                 (MAX(resource_type) - MIN(resource_type) ) Diffresource_type,
                 MAX(severity_type) Maxseverity_type, 
                 MIN(severity_type) Minseverity_type,
                 (MAX(severity_type) - MIN(severity_type) ) Diffseverity_type
                 
                 FROM df_all_combinedForMovings GROUP BY id, location")

Movings[is.na(Movings)] <- 0

# head(Movings,2)
# sqldf("select * from Movings  where id = 10024 ")


events       <- sqldf("SELECT id, location, fault_severity, event_type , AVG(volume) as [AvgEventVolume],MAX(volume) as [MaxEventVolume],   MIN(volume) as [MinEventVolume] FROM df_all_combined GROUP BY 1,2,3,4")

resources    <- sqldf("SELECT id, location, fault_severity, resource_type , AVG(volume) as [AvgResourceVolume] , MAX(volume) as [MaxResourceVolume], MIN(volume) as [MinResourceVolume]FROM df_all_combined GROUP BY 1,2,3,4")

Severities   <- sqldf("SELECT id, location, fault_severity, severity_type , AVG(volume) as [AvgSeverityVolume] , MAX(volume) as [MaxSeverityVolume] , MIN(volume) as [MinSeverityVolume]FROM df_all_combined GROUP BY 1,2,3,4")

logs         <- sqldf("SELECT id, location, fault_severity, log_feature ,   AVG(volume) as [AvglogVolume] FROM df_all_combined GROUP BY 1,2,3,4")

events       <- spread(events,   event_type ,  AvgEventVolume )
events[is.na(events)] <- 0
# names(events)

events$eventsCount <- rowSums(events[,6:58]>0)
# 
events$CountToTotalevents <- (events$eventsCount/53)

# sqldf("select * from events  where id = 10024 ")

resources    <- spread(resources,   resource_type ,  AvgResourceVolume )
resources[is.na(resources)] <- 0
#sqldf("select * from resources  where id = 10024 ")

# names(resources)

resources$resourcesCount <-rowSums(resources[,6:15]>0) 
# 
resources$CountToTotalresources <- (resources$resourcesCount/10)

Severities    <- spread(Severities,   severity_type ,  AvgSeverityVolume )
Severities[is.na(Severities)] <- 0

# sqldf("select * from Severities  where id = 10024 ")

logs    <- spread(logs,   log_feature ,  AvglogVolume )
logs[is.na(logs)] <- 0


# sqldf("select * from logs  where id = 10024 ")

# names(logs)
logs$logsCount <- rowSums(logs[,names(logs[-c(1:3)])])
# 
logs$logsNoCount <- rowSums(logs[,4:389]>0)

# sessionsdatas <- NULL
# df_all_combineds <- NULL
# merging data
sessionsdatas    <- merge(events,logs,by=(c("id" , "location" , "fault_severity"))    ,all = T)  
sessionsdatas    <- merge(sessionsdatas,resources,by=(c("id" , "location" , "fault_severity")) ,all = T)
sessionsdatas    <- merge(sessionsdatas,Severities,by=(c("id" , "location" , "fault_severity")) ,all = T)
df_all_combineds <- merge(df_all,sessionsdatas,by=(c("id" , "location" , "fault_severity")) ,all = T)

df_all_combineds <- merge(df_all_combineds,Movings,by=(c("id" , "location" )) ,all = T)
# sqldf("select * from df_all_combineds  where id = 10024 ")
# 
# event$eventCount <- 1
# 
# events    <- spread(event,   event_type ,  eventCount )
# 
# 
# events$eventsCount <- rowSums(events[,names(events[-c(1)])])
# 
# events$CountToTotalevents <- (events$eventsCount/53)
# 
# # sqldf("select * from events where id = 14121")
# 
# 
# # names(logs)
# 
# logs$logsCount <- rowSums(logs[,names(logs[-c(1)])])
# 
# logs$logsNoCount <- rowSums(logs[,2:386]>0)
# 
# logs$CountToTotallogs <- (logs$logsNoCount/386)
# 
# # sqldf("select * from logs where id = 14121")
# 
# resource$resourceCount <- 1
# 
# 
# 
# # names(resources[-c(1)])
# 
# resources$resourcesCount <- rowSums(resources[,names(resources[-c(1)])])
# 
# resources$CountToTotalresources <- (resources$resourcesCount/10)
# 
# # sqldf("select * from resources where id = 14121")
# 
# 
# severity$severityCount <- 1
# 
# 
# 
# sqldf("select * from severitys where id = 14121")
# 
# sessionsdata    <- merge(events,logs,by="id"      ,all = T)  
# sessionsdata    <- merge(sessionsdata,resources,by="id" ,all = T)
# sessionsdata    <- merge(sessionsdata,severitys,by="id" ,all = T)
# 
# 
# 
# 
# # head(train,2)
# # head(test ,2)
# # 
# # dim(train)
# # dim(test)
# 
# # 05. Union train and test datasets
# df_all  <-  rbind(train,test)
# 
# # head(df_all)
# 
# df_all_combined <- merge(df_all, sessionsdata , by ="id", all = T)
# 
# # sqldf("select fault_severity, cluster, count(*) as Count from df_all_combined group by fault_severity, cluster ")
# 
# # unique(df_all_combined$fault_severity)
# 
# # Cl <- kmeans(df_all_combined[c(4:464)], 3, nstart=100)
# # 
# # df_all_combined <- cbind(df_all_combined, Cluster = Cl$cluster)
# 
# #Cl$cluster
# 
df_all_combineds$location <- as.numeric(gsub("location",'',df_all_combineds$location))

Fulltrain  <- df_all_combineds[which(df_all_combineds$fault_severity > -1), ]
Fulltest   <- df_all_combineds[which(df_all_combineds$fault_severity <  0), ]

Fulltest$fault_severity  <- NULL 

featureNames <- names(Fulltrain[-c(1,3)])

TestfeatureNames <- names(Fulltest[-c(1)])

Fulltrain$fault_severity <- ifelse(Fulltrain$fault_severity==0,'Zero', ifelse(Fulltrain$fault_severity==1,'One', 'Two'))

Fulltrain$fault_severity  <- as.factor(Fulltrain$fault_severity)

# 
# # check for zero variance
# zero.var = nearZeroVar(Fulltrain[c(2,4:453)], saveMetrics=TRUE)
# zero.var[zero.var[,"zeroVar"] == 0, ]
# nzv <- zero.var[zero.var[,"zeroVar"] + zero.var[,"nzv"] > 0, ] 
# zero.var
# filter(zero.var, nzv$zeroVar == FALSE)
# badCols <- nearZeroVar(Fulltrain[c(2,4:453)])
# print(paste("Fraction of nearZeroVar columns:", round(length(badCols)/length(Fulltrain[c(2,4:453)]),4)))
# 
# # remove those "bad" columns from the training and cross-validation sets
# 
# train <- train[, -badCols]
# test <- test[, -badCols]
# 
# # corrPlot
# featurePlot(totaltrain[c(2,458,460:462)], outcome.org, "strip")
# 
# head(train.matrix)

require(randomForest)

rfCtrl <- trainControl(method="cv",	        # use repeated 10fold cross validation
                        #repeats=5,
                        number = 2,           # do 5 repititions of 10-fold cv
                        summaryFunction=multiClassSummary,	# Use AUC to pick the best model
                        classProbs=TRUE,
                        savePredictions = TRUE,
                        verboseIter=TRUE)

rfGrid <- expand.grid( mtry = 400)


#set.seed(546)
# registerDoParallel(4)		                        # Registrer a parallel backend for train
# getDoParWorkers()

featureNames <- names(Fulltrain[-c(1)])

system.time(rf.tune <- train(fault_severity~.,
                              data = Fulltrain[,featureNames],
                              method = "rf",
                              metric = "mlogloss",
                              trControl = rfCtrl,
                              tuneGrid=rfGrid,
                              ntree = 600,
                              importance = TRUE,
                              verbose=TRUE))

summary(rf.tune)

rf.tune$results

head(rf.tune$pred)

featureNames <- names(Fulltrain[-c(1,3)])

# Training error rate
confusionMatrix(predict(rf.tune, Fulltrain[,featureNames]), Fulltrain$fault_severity)
# Accuracy : 0.6931

# predictedClasses <- predict(svm.c, Fulltest[,TestfeatureNames] )
predictedProbs   <- predict(rf.tune, Fulltest[,TestfeatureNames], type = "prob")

head(predictedProbs)


# head(prediction)
# sqldf("select * from prediction where id = 1442")
# names(prediction)

prediction <- cbind( id = Fulltest$id ,  predict_0 = predictedProbs[,3] , predict_1 = predictedProbs[,1], predict_2 = predictedProbs[,2] )

write.csv(prediction, "submission26.csv", quote=FALSE, row.names = FALSE)

#write.table(prediction,file="submission21.csv", append=TRUE,sep=",",col.names=TRUE,row.names=FALSE)


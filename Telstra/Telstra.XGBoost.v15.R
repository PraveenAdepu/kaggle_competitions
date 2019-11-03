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

# rm(list=ls())
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

head(event)



# 04. Set target variable to test data
test$fault_severity <- -1

df_all  <-  rbind(train,test)

head(log,2)

FeatureFreqs <- sqldf("select log_feature, count(*) as featureFreq from log group by log_feature")

head(FeatureFreqs,2)

logFeatureFreqs <- merge(log,FeatureFreqs, by= "log_feature", all.x = TRUE)
logFeatureFreqs$volume <- NULL
logFeatureFreqs$log_feature <- NULL

logFeatureFreqs <- sqldf("select id , sum(featureFreq) as TotalfeatureFreq 
                         , MAX(featureFreq) as MaxfeatureFreq
                         , MIN(featureFreq) as MinfeatureFreq
                         , AVG(featureFreq) as AvgfeatureFreq 
                         from logFeatureFreqs group by id ")
logFeatureFreqs$DifffeatureFreq <- logFeatureFreqs$MaxfeatureFreq - logFeatureFreqs$MinfeatureFreq 
# logFeatureFreqs$TotalfeatureFreq <- log(logFeatureFreqs$TotalfeatureFreq + 1)
# logFeatureFreqs$MaxfeatureFreq <- log(logFeatureFreqs$MaxfeatureFreq + 1)
# logFeatureFreqs$MinfeatureFreq <- log(logFeatureFreqs$MinfeatureFreq + 1)
# logFeatureFreqs$AvgfeatureFreq <- log(logFeatureFreqs$AvgfeatureFreq + 1)
# 
# logFeatureFreqs$NormfeatureFreq <- logFeatureFreqs$MaxfeatureFreq - logFeatureFreqs$MinfeatureFreq

head(logFeatureFreqs,2)
# head(df_all,2)

head(event,2)

EventFreqs <- sqldf("select event_type, count(*) as eventFreq from event group by event_type")

head(EventFreqs,2)

eventTypeFreqs <- merge(event,EventFreqs, by= "event_type", all.x = TRUE)

eventTypeFreqs$event_type <- NULL

eventTypeFreqs <- sqldf("select id , sum(eventFreq) as TotaleventFreq 
                         , MAX(eventFreq) as MaxeventFreq
                         , MIN(eventFreq) as MineventFreq
                         , AVG(eventFreq) as AvgeventFreq
                         from eventTypeFreqs group by id ")

eventTypeFreqs$DiffeventFreq <- eventTypeFreqs$MaxeventFreq - eventTypeFreqs$MineventFreq 

head(eventTypeFreqs,2)

head(resource,2)

ResourceFreqs <- sqldf("select resource_type, count(*) as resourceFreq from resource group by resource_type")

head(ResourceFreqs,2)

resourceTypeFreqs <- merge(resource,ResourceFreqs, by= "resource_type", all.x = TRUE)

resourceTypeFreqs$resource_type <- NULL

resourceTypeFreqs <- sqldf("select id , sum(resourceFreq) as TotalresourceFreq 
                         , MAX(resourceFreq) as MaxresourceFreq
                         , MIN(resourceFreq) as MinresourceFreq
                         , AVG(resourceFreq) as AvgresourceFreq
                         from resourceTypeFreqs group by id ")

resourceTypeFreqs$DiffresourceFreq <- resourceTypeFreqs$MaxresourceFreq - resourceTypeFreqs$MinresourceFreq 

head(resourceTypeFreqs)

# merging data
Moves                     <- merge(event,log,by="id"      ,all = T)  
Moves                     <- merge(Moves,resource,by="id" ,all = T)
Moves                     <- merge(Moves,severity,by="id" ,all = T)
df_all_combinedForMovings <- merge(df_all,Moves,by="id"   ,all = T)


df_all_combinedForMovings$location      <- as.integer(gsub("location ","",df_all_combinedForMovings$location))
df_all_combinedForMovings$event_type    <- as.integer(gsub("event_type ","",df_all_combinedForMovings$event_type))
df_all_combinedForMovings$log_feature   <- as.integer(gsub("feature ","",df_all_combinedForMovings$log_feature))
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
                 AVG(event_type) Meanevent_Type,
                avg(resource_type) + avg(event_type) TotalEventResource,
                avg(resource_type) + avg(log_feature) TotalLogResource,
                avg(event_type) + avg(log_feature) TotalEventLog
--,
              --  ((id/location) * MIN(resource_type)) as  Totallogfeature
--,
               -- COUNT(*) + Max(resource_type) + MIN(resource_type) MixFeature,
--,
              --  avg(resource_type) + avg(event_type) + avg(log_feature) TotalAllTest
                 -- ,
                 
                 
                 -- AVG(resource_type) Avgresource_type,
                 -- AVG(log_feature) Avglog_feature,
                 -- MAX(volume) as MaxVolume,
                 -- MIN(volume) as MinVolume,
                 -- stdev(log_feature) as Stdlog_feature,
                 -- stdev(volume) as StdVolume
                 -- ,
                 --   AVG(log_feature) Meanlog_feature
                 --   (( MAX(volume) - MIN(volume) )/stdev(volume))  as NormVolume
                 --,
                 -- MIN(volume) as MinVolume,
                 -- stdev(volume) as StdVolume
                 --  MAX(severity_type) Maxseverity_type, 
                 -- MIN(severity_type) Minseverity_type,
                 -- (MAX(severity_type) - MIN(severity_type) ) Diffseverity_type,
                 -- COUNT(*) AS RowCount
                 
                 FROM df_all_combinedForMovings GROUP BY id, location")

Movings[is.na(Movings)] <- 0

#Movings$Totallogfeature <- log(Movins$TotalVolume)

M <- cor(Movings)
corrplot.mixed(M)



head(Movings)
event$eventtype <- 1 #as.integer(gsub("event_type ","",event$event_type))
head(event)
events       <- spread(event,   event_type ,  eventtype )
head(events)
sqldf("select * from events where id = 10024")
events[is.na(events)] <- 0
sqldf("select * from events where id = 10024")
# dim(events)
# names(events[,2:54])
events$eventsCount <- rowSums(events[,2:54]>0)

head(resource)
resource$resourcetype <- 1 # as.integer(gsub("resource_type ","",resource$resource_type))
head(resource)
resources       <- spread(resource,   resource_type ,  resourcetype )
head(resources)
sqldf("select * from resources where id = 10024")
resources[is.na(resources)] <- 0
sqldf("select * from resources where id = 10024")
# dim(resources)
# names(resources[,2:11])
resources$resourcesCount <-rowSums(resources[,2:11]>0) 

severitytypeInts <- severity

head(severity)
severity$severitytype <- 1  #as.integer(gsub("severity_type ","",severity$severity_type))
head(severity)
severities       <- spread(severity,   severity_type ,  severitytype )
head(severities)
sqldf("select * from severities where id = 10024")
severities[is.na(severities)] <- 0
sqldf("select * from severities where id = 10024")

severity$SeverityInt <- as.integer(gsub("severity_type ","",severity$severity_type))
severityhelper <- sqldf("SELECT id , SeverityInt as SeverityInt, case when severity_type in ('severity_type 1' , 'severity_type 2') then 1 else 0 end as severityhelper
                        FROM severity")

#head(severityhelper)

severitytypeInts$severity_type <- as.integer(gsub("severity_type ","",severitytypeInts$severity_type))

head(log)
logs       <- spread(log,   log_feature ,  volume )
head(logs)
sqldf("select * from logs where id = 10024")
logs[is.na(logs)] <- 0
sqldf("select * from logs where id = 10024")

# dim(logs)
# names(logs [,2:387])
logs$logsCount <-rowSums(logs[,2:387]>0) 
logs$logsVolume <- rowSums(logs[,2:387])


logseverities    <- merge(log,severitytypeInts ,by="id"      ,all = T) 

logseverities$severityXvolume <- logseverities$volume * logseverities$severity_type
head(logseverities)

logseverityvolume <- sqldf("SELECT id , SUM(severityXvolume) as severityXvolume from logseverities group by id")

head(logseverityvolume)

# head(train,2)
# head(test ,2)

# 04. Set target variable to test data
# test$fault_severity <- -1
# 
# df_all  <-  rbind(train,test)

# head(df_all,2)

# merging data
dim(events)
dim(logs)
sessionsdata    <- merge(events,logs,by="id"      ,all = T) 
dim(sessionsdata)
sessionsdata    <- merge(sessionsdata,resources,by="id" ,all = T)
dim(sessionsdata)
sessionsdata    <- merge(sessionsdata,severities,by="id" ,all = T)
dim(sessionsdata)

sessionsdata    <- merge(sessionsdata,severityhelper,by="id" ,all = T)
dim(sessionsdata)

sessionsdata    <- merge(sessionsdata,logseverityvolume,by="id" ,all = T)
dim(sessionsdata)

sessionsdata    <- merge(sessionsdata,logFeatureFreqs,by="id" ,all = T)
dim(sessionsdata)


sessionsdata    <- merge(sessionsdata,eventTypeFreqs,by="id" ,all = T)
dim(sessionsdata)

sessionsdata    <- merge(sessionsdata,resourceTypeFreqs,by="id" ,all = T)
dim(sessionsdata)



dim(df_all)
df_all_combined <- merge(df_all,sessionsdata,by="id" ,all = T)

dim(df_all_combined)

# logstats     <- sqldf("SELECT id,  SUM(volume) logsvolume, COUNT(*) as logsCount, AVG(volume) as MeanVolume, MAX(volume) Maxvolume, MIN(volume) Minvolume, 
#                       stdev(volume) Stdvolume   
#                       FROM log GROUP BY 1")


df_all_combined$location <- as.numeric(gsub("location",'',df_all_combined$location))

df_all_combined$Rows <- df_all_combined$eventsCount * df_all_combined$resourcesCount * df_all_combined$logsCount

df_all_combined$RowBins <- ifelse(df_all_combined$Rows > 100 , 1, 0)
# # 
df_all_combined$LocationBins <- ifelse(df_all_combined$location > 550 , 1, 0)

df_all_combined$RowsSum <- df_all_combined$eventsCount + df_all_combined$resourcesCount + df_all_combined$logsCount

#df_all_combined$VolumeBins <- ifelse(df_all_combined$logsVolume > 750 , 1, 0)
# 
#df_all_combined$Rows  <- NULL

#head(df_all_combined,2)
#head(Movings,2)

Locationfrequencies <- sqldf("select location , count(*) as LocationFreq from df_all_combined group by location ")

df_all_combined <- merge(df_all_combined,Locationfrequencies,by="location" ,all.x = T)

df_all_combined <- merge(df_all_combined, Movings, by=(c("id" , "location" )) ,all = T)


# df_all_combined %>% 
#   group_by(location) %>% 
#   mutate(locationTimeOrder = row_number(-id))
# 
# df_all_combined <- transform(df_all_combined, 
#                                           locationTimeOrder = ave(id, location, 
#                                            FUN = function(x) rank(-x, ties.method = "first")))
# 
# df_all_combined$locationTimeOrder
df_all_combined$allFreq <- (df_all_combined$LocationFreq + df_all_combined$TotalresourceFreq + df_all_combined$TotaleventFreq  ) 


#df_all_combined$MaxMeanEventDiff <- df_all_combined$Maxevent_type - df_all_combined$Meanevent_Type
#df_all_combined$MinMeanEventDiff <- df_all_combined$Meanevent_type - df_all_combined$Minevent_Type
# df_all_combined$LogsFeatureFreq <-     df_all_combined$TotalfeatureFreq/df_all_combined$logsCount

# df_all_combined$logsCountBin <- ifelse(df_all_combined$logsCount > 13 , 1 , 0)

# df_all_combined$VolumeToEvents    <- df_all_combined$logsVolume/df_all_combined$eventsCount
# df_all_combined$VolumeToResources <- df_all_combined$logsVolume/df_all_combined$resourcesCount

# df_all_combined$severityXvolume <- log(df_all_combined$severityXvolume)
# 
# Eventfrequencies <- sqldf("select event_type , count(*) as EventFreq from event group by event_type")
# 
# EventFreqCounts <- merge(event, Eventfrequencies, by="event_type",all.x = T)
# 
# EventFreqCounts$eventtype <- NULL
# 
# EventFreqCounts$eventtype <- NULL
# 
# EventFreqCounts <- sqldf("select id , sum(EventFreq) EventFreq from EventFreqCounts group by id")
# 
# df_all_combined <- merge(df_all_combined,EventFreqCounts,by="id" ,all.x = T)
# 
# df_all_combined$LocationToEventFreqRatio <- df_all_combined$EventFreq / df_all_combined$LocationFreq
# head(EventFreqCounts)
# summary(df_all_combined$LocationToEventFreqRatio)


# Locationfrequencies <- sqldf("select location , count(*) as LocationFreq from df_all_combined group by location ")

# sqldf("select id, fault_severity from df_all_combined where id IN(10024,1,10059)")

# set.seed(1234)
# fit.km <- kmeans(df_all_combined[-c(1,3)], 25, nstart=25)  

# df_all_combined <- cbind(df_all_combined, Cluster = fit.km$cluster)

# ct.km <- table(df_all_combined$fault_severity, fit.km$cluster)
# ct.km

# library(NbClust)
# set.seed(1234)
# nc <- NbClust(Fulltrain[-c(1,3)], min.nc=2, max.nc=15, method="kmeans")
# table(nc$Best.n[1,])


Fulltrain  <- df_all_combined[which(df_all_combined$fault_severity > -1), ]
Fulltest   <- df_all_combined[which(df_all_combined$fault_severity <  0), ]

Fulltest$fault_severity  <- NULL




# #######################################################################################################
# # Data Visualisations
## 3D Scatterplot
# require(scatterplot3d)
# scatterplot3d(Fulltrain$location,Fulltrain$fault_severity,Fulltrain$id, main="3D Scatterplot")
# 
# require(rgl)
# plot3d(Fulltrain$location,Fulltrain$fault_severity,Fulltrain$Totallogvolume, col="red", size=3)
# 
# pairs(iris[1:4], main = "Anderson's Iris Data -- 3 species",
#       pch = 21, bg = c("red", "green3", "blue")[unclass(iris$Species)])
# 
# # Using formula
# 
# pairs(~Fulltrain$location + Fulltrain$fault_severity + Fulltrain$Maxevent_type, main = "Fault Severity",
#       pch = 21, bg = c("red", "green3", "blue")[unclass(Fulltrain$fault_severity)])

# M <- cor(Fulltrain[c(3,498)])
# corrplot.mixed(M)
# conputationally expensive
# require(GGally)
# ggpairs(Fulltrain[c(3,498)], colour="fault_severity", alpha=0.4)

# head(Fulltrain[c(1,2,3)])
# 
# filter(Fulltrain, location == "601")



########################################################################################################


# names(Fulltrain)

featureNames <- names(Fulltrain [-c(1,3)]) # ,57,444


train.matrix <- as.matrix(Fulltrain[,featureNames])#, Fulltrain$severity_type == 3))
test.matrix  <- as.matrix(Fulltest[,featureNames]) #, Fulltest$severity_type == 3))

target <- ifelse(Fulltrain$fault_severity==0,'Zero', ifelse(Fulltrain$fault_severity==1,'One', 'Two'))
#y <- recode(target,"'Zero'=0; 'One'=1; 'Two'=2")
classnames = unique(target)
#target = as.integer(colsplit(target,'_',names=c('x1','x2'))[,2])
target  <- as.factor(target)
# outcome.org = Fulltrain$fault_severity
# outcome = outcome.org 
# levels(outcome)
y = target
y = as.matrix(as.integer(target)-1)
cvtarget = y
cvtarget <- as.factor(cvtarget)
num.class = length(levels(target))

# xgboost parameters
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = num.class,    # number of classes 
              "eval_metric" = "mlogloss",    # evaluation metric 
              "nthread" = 4,   # number of threads to be used 
              "max_depth" = 8,    # maximum depth of tree # 6
              "eta" = 0.05,    # step size shrinkage  # 0.5
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 0.9,    # part of data instances to grow tree # 0.5
              "colsample_bytree" = 0.7,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 3  # minimum sum of instance weight needed in a child 
              
)

# set random seed, for reproducibility 
set.seed(1231)
# k-fold cross validation, with timing
nround.cv = 2000
system.time( bst.cv <- xgb.cv(param=param, data=train.matrix, label=y, early.stop.round=10,
                              nfold=10, nrounds=nround.cv, prediction=TRUE, verbose=TRUE) )
#tail(bst.cv$dt)
# index of minimum merror
min.merror.idx = which.min(bst.cv$dt[, test.mlogloss.mean+test.mlogloss.std]) 
min.merror.idx
bst.cv$dt[min.merror.idx ,]


xgb.ctrl <- trainControl(method = "repeatedcv", repeats = 2,number = 5, 
                         summaryFunction = multiClassSummary,
                         classProbs = TRUE,
                         allowParallel=T, 
                         verboseIter =  TRUE)

xgb.grid <- expand.grid(nrounds = c(min.merror.idx ), # try with 195, get best nround from XGBoost model then apply here for Caret grid
                        eta = c(0.05),
                        max_depth = c(8),
                        colsample_bytree = c(0.7),
                        #subsample = c(0.3,0.5,0.6),
                        min_child_weight = c(3),
                        gamma = 0
)
set.seed(45)
# set.seed(54)
xgb_tune <-train(x=train.matrix, y=target, # factor level string levels
                 #data=train,
                 method="xgbTree",
                 objective = "multi:softprob",
                 trControl=xgb.ctrl,
                 tuneGrid=xgb.grid,
                 verbose=1,
                 metric="logLoss",
                 nthread =3,
                 print.every.n=5
)


xgb_tune

# #0.4943869, 0.01010081 , 0.7795671 , 0.01015636
# #0.4947401, 0.009533379, 0.7818702 , 0.01117952
# xgb_tune$best
# xgb_tune$best$nrounds 
# xgb_tune$best$max_depth
# xgb_tune$best$eta
# xgb_tune$best$colsample_bytree
# xgb_tune$best$min_child_weight
# xgb_tune$best$gamma

# # # get the trained model
# model = xgb.dump(xgb_tune, with.stats=TRUE)
# # # get the feature real names
# names = dimnames(train.matrix)[[2]]
# # # compute feature importance matrix
# importance_matrix = xgb.importance(names, model=bst)
# print(importance_matrix)

# Training error rate
# confusionMatrix(predict(xgb_tune, Fulltrain[,featureNames]), target) ## Fact level target (string levels)

predictedProbs   <- predict(xgb_tune, test.matrix, type = "prob")

#head(predictedProbs)

#head(test.matrix)
# head(prediction)
# sqldf("select * from prediction where id = 11")
# names(prediction)

prediction <- cbind( id = Fulltest$id ,  predict_0 = predictedProbs[,3] , predict_1 = predictedProbs[,1], predict_2 = predictedProbs[,2] )

write.csv(prediction, "submission47.csv", quote=FALSE, row.names = FALSE)

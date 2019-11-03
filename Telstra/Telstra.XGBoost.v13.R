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

# head(df_all,2)

# merging data
Moves                     <- merge(event,log,by="id"      ,all = T)  
Moves                     <- merge(Moves,resource,by="id" ,all = T)
Moves                     <- merge(Moves,severity,by="id" ,all = T)
df_all_combinedForMovings <- merge(df_all,Moves,by="id" ,all = T)


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
                 AVG(event_type) Meanevent_Type
-- ,
--                 AVG(log_feature) Meanlog_feature
--                (( MAX(volume) - MIN(volume) )/stdev(volume))  as NormVolume
--,
               -- MIN(volume) as MinVolume,
               -- stdev(volume) as StdVolume
               --  MAX(severity_type) Maxseverity_type, 
                -- MIN(severity_type) Minseverity_type,
                -- (MAX(severity_type) - MIN(severity_type) ) Diffseverity_type,
                -- COUNT(*) AS RowCount
                 
                 FROM df_all_combinedForMovings GROUP BY id, location")

Movings[is.na(Movings)] <- 0

#Movins$TotalVolume <- log(Movins$TotalVolume)

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
# 
#df_all_combined$Rows  <- NULL

#head(df_all_combined,2)
#head(Movings,2)

Locationfrequencies <- sqldf("select location , count(*) as LocationFreq from df_all_combined group by location ")

df_all_combined <- merge(df_all_combined,Locationfrequencies,by="location" ,all.x = T)

df_all_combined <- merge(df_all_combined, Movings, by=(c("id" , "location" )) ,all = T)


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
Fulltrain  <- df_all_combined[which(df_all_combined$fault_severity > -1), ]
Fulltest   <- df_all_combined[which(df_all_combined$fault_severity <  0), ]

Fulltest$fault_severity  <- NULL 


# 
# # 3D Scatterplot
# require(scatterplot3d)
# scatterplot3d(Fulltrain$location,Fulltrain$fault_severity,Fulltrain$Maxevent_type, main="3D Scatterplot")
# 
# require(rgl)
# plot3d(Fulltrain$location,Fulltrain$fault_severity,Fulltrain$Maxevent_type, col="red", size=3)
# 
# pairs(iris[1:4], main = "Anderson's Iris Data -- 3 species",
#       pch = 21, bg = c("red", "green3", "blue")[unclass(iris$Species)])
# 
# # Using formula
# 
# pairs(~Fulltrain$location + Fulltrain$fault_severity + Fulltrain$Maxevent_type, main = "Fault Severity",
#       pch = 21, bg = c("red", "green3", "blue")[unclass(Fulltrain$fault_severity)])

########################################


Fulltrain$ZeroProb <- ifelse(Fulltrain$fault_severity ==0, 1, 0)
# sqldf("select id, fault_severity,ZeroProb from Fulltrain where id IN(10024,1,10059)")
# names(Fulltrain)

featureNames <- names(Fulltrain [-c(1,3,478)])

# train.matrix <- as.matrix(Fulltrain[,featureNames])#, Fulltrain$severity_type == 3))
# test.matrix  <- as.matrix(Fulltest[,featureNames]) #, Fulltest$severity_type == 3))


dtrain<-xgb.DMatrix(data=data.matrix(Fulltrain[,featureNames]),label=Fulltrain$ZeroProb)

#watchlist<-list(val=dval,train=dtrain)
param <- list(  "objective"        = "binary:logistic", 
                "booster"          = "gbtree",
                "eval_metric"      = "logloss",#"merror",#"auc",
                "nthread"          = 4,   # number of threads to be used 
                "max_depth"        = 8, #changed from default of 8
                "eta"              = 0.05,#0.023,#0.1, # 0.06, #0.01,
                "gamma"            = 0,    # minimum loss reduction 
                "subsample"        = 0.7, #0.5, # 0.7
                "colsample_bytree" = 0.7, #0.5, # 0.7
                "min_child_weight" = 3
                #num_parallel_tree   = 2
                # alpha = 0.0001, 
                # lambda = 1
)


nround.cv = 1800
system.time( bst.cv <- xgb.cv(param=param
                              , data=dtrain
                              , early.stop.round=10
                              , nfold=3
                              , nrounds=nround.cv
                              , prediction=TRUE
                              , verbose=TRUE
                              , maximize = FALSE
                              , print.every.n=5) )

#tail(bst.cv$dt)
# index of minimum merror
max.auc.idx = which.min(bst.cv$dt[, test.logloss.mean]) 
max.auc.idx
bst.cv$dt[max.auc.idx ,]

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = max.auc.idx , #1800, 
                    verbose             = TRUE,  #1
                    #early.stop.round    = 150,
                    #watchlist           = watchlist,
                    maximize            = FALSE
)



Fulltrain$ZerovsAllProb <- predict(clf, data.matrix(Fulltrain[,featureNames]))
Fulltest$ZerovsAllProb  <- predict(clf, data.matrix(Fulltest[,featureNames]))

# sqldf("select id, fault_severity,ZeroProb,ZerovsAllProb from Fulltrain where id IN(10024,1,10059)")

# names(Fulltrain)

#featureNames <- names(Fulltrain [-c(1,3,465,466)])

Fulltrain$ZeroProb <- ifelse(Fulltrain$fault_severity == 1, 1, 0)

dtrain<-xgb.DMatrix(data=data.matrix(Fulltrain[,featureNames]),label=Fulltrain$ZeroProb)

#watchlist<-list(val=dval,train=dtrain)
param <- list(  "objective"        = "binary:logistic", 
                "booster"          = "gbtree",
                "eval_metric"      = "logloss",#"merror",#"auc",
                "nthread"          = 4,   # number of threads to be used 
                "max_depth"        = 8, #changed from default of 8
                "eta"              = 0.05,#0.023,#0.1, # 0.06, #0.01,
                "gamma"            = 0,    # minimum loss reduction 
                "subsample"        = 0.7, #0.5, # 0.7
                "colsample_bytree" = 0.7, #0.5, # 0.7
                "min_child_weight" = 3
                #num_parallel_tree   = 2
                # alpha = 0.0001, 
                # lambda = 1
)


nround.cv = 1800
system.time( bst.cv <- xgb.cv(param=param
                              , data=dtrain
                              , early.stop.round=10
                              , nfold=3
                              , nrounds=nround.cv
                              , prediction=TRUE
                              , verbose=TRUE
                              , maximize = FALSE
                              , print.every.n=5) )

#tail(bst.cv$dt)
# index of minimum merror
max.auc.idx = which.min(bst.cv$dt[, test.logloss.mean]) 
max.auc.idx
bst.cv$dt[max.auc.idx ,]

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = max.auc.idx , #1800, 
                    verbose             = TRUE,  #1
                    #early.stop.round    = 150,
                    #watchlist           = watchlist,
                    maximize            = FALSE
)



Fulltrain$ZeroOnevsAllProb <- predict(clf, data.matrix(Fulltrain[,featureNames]))
Fulltest$ZeroOnevsAllProb  <- predict(clf, data.matrix(Fulltest[,featureNames]))


# sqldf("select id, fault_severity,ZeroProb,ZerovsAllProb,ZeroOnevsAllProb from Fulltrain where id IN(10024,1,10059)")

# names(Fulltrain)

#featureNames <- names(Fulltrain [-c(1,3,465,466)])

Fulltrain$ZeroProb <- ifelse(Fulltrain$fault_severity == 2, 1, 0)

dtrain<-xgb.DMatrix(data=data.matrix(Fulltrain[,featureNames]),label=Fulltrain$ZeroProb)

#watchlist<-list(val=dval,train=dtrain)
param <- list(  "objective"        = "binary:logistic", 
                "booster"          = "gbtree",
                "eval_metric"      = "logloss",#"merror",#"auc",
                "nthread"          = 4,   # number of threads to be used 
                "max_depth"        = 8, #changed from default of 8
                "eta"              = 0.05,#0.023,#0.1, # 0.06, #0.01,
                "gamma"            = 0,    # minimum loss reduction 
                "subsample"        = 0.7, #0.5, # 0.7
                "colsample_bytree" = 0.7, #0.5, # 0.7
                "min_child_weight" = 3
                #num_parallel_tree   = 2
                # alpha = 0.0001, 
                # lambda = 1
)


nround.cv = 1800
system.time( bst.cv <- xgb.cv(param=param
                              , data=dtrain
                              , early.stop.round=10
                              , nfold=3
                              , nrounds=nround.cv
                              , prediction=TRUE
                              , verbose=TRUE
                              , maximize = FALSE
                              , print.every.n=5) )

#tail(bst.cv$dt)
# index of minimum merror
max.auc.idx = which.min(bst.cv$dt[, test.logloss.mean]) 
max.auc.idx
bst.cv$dt[max.auc.idx ,]

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = max.auc.idx , #1800, 
                    verbose             = TRUE,  #1
                    #early.stop.round    = 150,
                    #watchlist           = watchlist,
                    maximize            = FALSE
)



Fulltrain$TwovsAllProb <- predict(clf, data.matrix(Fulltrain[,featureNames]))
Fulltest$TwovsAllProb  <- predict(clf, data.matrix(Fulltest[,featureNames]))

Fulltrain$ZeroProb     <- NULL
Fulltrain$AllProbSum <- (Fulltrain$ZerovsAllProb * 0.1) + (Fulltrain$ZeroOnevsAllProb * 1) + (Fulltrain$TwovsAllProb * 2)
Fulltest$AllProbSum  <- (Fulltest$ZerovsAllProb * 0.1) + (Fulltest$ZeroOnevsAllProb * 1) + (Fulltest$TwovsAllProb * 2)


########################################
# hist(log(Fulltrain$severityXvolume) , breaks=15, main="Breaks=15")
# hist(log(Fulltrain$severityXvolume))

# hist(log(Fulltrain$severityXvolume))
# sqldf("select id, fault_severity,ZeroProb,ZerovsAllProb,ZeroOnevsAllProb,TwovsAllProb  from Fulltrain where id IN(10024,1,10059)")
# sqldf("select id,ZerovsAllProb,ZeroOnevsAllProb,TwovsAllProb from Fulltest where id IN(10000)")
# names(Fulltrain)

featureNames <- names(Fulltrain [-c(1,3)]) # ,466,467,468


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
              "subsample" = 0.7,    # part of data instances to grow tree # 0.5
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

xgb.grid <- expand.grid(nrounds = c(min.merror.idx), # try with 195, get best nround from XGBoost model then apply here for Caret grid
                        eta = c(0.05),
                        max_depth = c(8),
                        colsample_bytree = c(0.7),
                        #subsample = c(0.3,0.5,0.6),
                        min_child_weight = c(3),
                        gamma = 0
)
set.seed(45)
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
#0.5034819
#0.4951167
#0.4947347
xgb_tune$best
xgb_tune$best$nrounds 
xgb_tune$best$max_depth
xgb_tune$best$eta
xgb_tune$best$colsample_bytree
xgb_tune$best$min_child_weight
xgb_tune$best$gamma

# Training error rate
confusionMatrix(predict(xgb_tune, Fulltrain[,featureNames]), target) ## Fact level target (string levels)
# Accuracy : 0.9252,  0.5478917  0.8919464  0.761549  0.5223564
# ZerovsAll : 0.4410035
# OnevsAll : 0.446633
# TwovsAll : 0.4794516
# AllProb : 0.3900977
# AllProbSum : 0.5027926
# AllProbSum weights : 0.4366963
# predictedClasses <- predict(svm.c, Fulltest[,TestfeatureNames] )
predictedProbs   <- predict(xgb_tune, test.matrix, type = "prob")

head(predictedProbs)

#head(test.matrix)
# head(prediction)
# sqldf("select * from prediction where id = 11")
# names(prediction)

prediction <- cbind( id = Fulltest$id ,  predict_0 = predictedProbs[,3] , predict_1 = predictedProbs[,1], predict_2 = predictedProbs[,2] )

write.csv(prediction, "submission42.csv", quote=FALSE, row.names = FALSE)


# 
# 
# target <- ifelse(Fulltrain$fault_severity==0,'Zero', ifelse(Fulltrain$fault_severity==1,'One', 'Two'))
# #y <- recode(target,"'Zero'=0; 'One'=1; 'Two'=2")
# classnames = unique(target)
# #target = as.integer(colsplit(target,'_',names=c('x1','x2'))[,2])
# target  <- as.factor(target)
# # outcome.org = Fulltrain$fault_severity
# # outcome = outcome.org 
# # levels(outcome)
# y = target
# y = as.matrix(as.integer(target)-1)
# num.class = length(levels(target))
# 
# # names(Fulltrain)
# # featurePlot(Fulltrain[c(2)], target, "strip")
# 
# 
# # 
# 
# 
# #test.mlogloss.mean test.mlogloss.std 0.556998  0.035416 Accuracy : 0.7439 
# # 1:            0.435771           0.002387           0.541668          0.033271
# #1:             0.337070           0.002493           0.506490          0.020044
# #1:             0.330594           0.001691           0.505799          0.021455
# #1:             0.321642           0.002057           0.503586          0.021882
# #1:             0.325798           0.001945           0.503112          0.020154
# #1:             0.32757            0.001977           0.502703          0.021816
# 
# # get CV's prediction decoding
# pred.cv = matrix(bst.cv$pred, nrow=length(bst.cv$pred)/num.class, ncol=num.class)
# pred.cv = max.col(pred.cv, "last")
# # confusion matrix
# confusionMatrix(factor(y+1), factor(pred.cv))
# 
# 
# # real model fit training, with full data
# system.time( bst <- xgboost(param=param, data=train.matrix, label=y, 
#                             nrounds=min.merror.idx, verbose=1) )
# #train-mlogloss:0.455049
# 
# # # get the trained model
# model = xgb.dump(bst, with.stats=TRUE)
# # # get the feature real names
# names = dimnames(train.matrix)[[2]]
# # # compute feature importance matrix
# importance_matrix = xgb.importance(names, model=bst)
# print(importance_matrix)
# 
# # # plot
# gp = xgb.plot.importance(importance_matrix)
# print(gp) 
# # 
# # tree = xgb.plot.tree(feature_names = names, model = bst, n_first_tree = 1)
# # print(tree)
# # 
# # # xgboost predict test data using the trained model
# predict <- predict(bst, test.matrix)  
# # #head(predict, 10) 
# 
# 
# 
# # decode prediction
# predict01 = matrix(predict, nrow=num.class, ncol=length(predict)/num.class , byrow=T)
# predict01 = t(predict01)
# # head(predict01)
# 
# colnames(predict01) = classnames
# 
# # head(prediction)
# # sqldf("select * from prediction where id = 1442")
# # names(prediction)
# 
# prediction <- cbind( id = Fulltest$id , severity_type = Fulltest$severity_type,  predict_0 = predict01[,2] , predict_1 = predict01[,1], predict_2 = predict01[,3] )
# 
# write.csv(prediction, "submission26.csv", quote=FALSE, row.names = FALSE)
# 
# #write.table(prediction,file="submission21.csv", append=TRUE,sep=",",col.names=TRUE,row.names=FALSE)



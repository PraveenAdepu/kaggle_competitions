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

head(severity)
severity$severitytype <- 1  #as.integer(gsub("severity_type ","",resource$severity_type))
head(severity)
severities       <- spread(severity,   severity_type ,  severitytype )
head(severities)
sqldf("select * from severities where id = 10024")
severities[is.na(severities)] <- 0
sqldf("select * from severities where id = 10024")

severityhelper <- sqldf("SELECT id , case when severity_type in ('severity_type 1' , 'severity_type 2') then 1 else 0 end as severityhelper FROM severity")

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



# head(train,2)
# head(test ,2)

# 04. Set target variable to test data
test$fault_severity <- -1

df_all  <-  rbind(train,test)

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

dim(df_all)
df_all_combined <- merge(df_all,sessionsdata,by="id" ,all = T)

dim(df_all_combined)

# logstats     <- sqldf("SELECT id,  SUM(volume) logsvolume, COUNT(*) as logsCount, AVG(volume) as MeanVolume, MAX(volume) Maxvolume, MIN(volume) Minvolume, 
#                       stdev(volume) Stdvolume   
#                       FROM log GROUP BY 1")


df_all_combined$location <- as.numeric(gsub("location",'',df_all_combined$location))

df_all_combined$Rows <- df_all_combined$eventsCount * df_all_combined$resourcesCount * df_all_combined$logsCount

 df_all_combined$RowBins <- ifelse(df_all_combined$Rows > 100 , 1, 0)
# 
 df_all_combined$LocationBins <- ifelse(df_all_combined$location > 550 , 1, 0)
# 
 df_all_combined$Rows  <- NULL



# sqldf("select id, fault_severity from df_all_combined where id IN(10024,1,10059)")
Fulltrain  <- df_all_combined[which(df_all_combined$fault_severity > -1), ]
Fulltest   <- df_all_combined[which(df_all_combined$fault_severity <  0), ]

Fulltest$fault_severity  <- NULL 

########################################


Fulltrain$ZeroProb <- ifelse(Fulltrain$fault_severity ==0, 1, 0)
# sqldf("select id, fault_severity,ZeroProb from Fulltrain where id IN(10024,1,10059)")
# names(Fulltrain)

featureNames <- names(Fulltrain [-c(1,2,3,465)])

# train.matrix <- as.matrix(Fulltrain[,featureNames])#, Fulltrain$severity_type == 3))
# test.matrix  <- as.matrix(Fulltest[,featureNames]) #, Fulltest$severity_type == 3))


dtrain<-xgb.DMatrix(data=data.matrix(Fulltrain[,featureNames]),label=Fulltrain$ZeroProb)

#watchlist<-list(val=dval,train=dtrain)
param <- list(  "objective"        = "binary:logistic", 
                "booster"          = "gbtree",
                "eval_metric"      = "logloss",#"merror",#"auc",
                "nthread"          = 4,   # number of threads to be used 
                "max_depth"        = 6, #changed from default of 8
                "eta"              = 0.023,#0.023,#0.1, # 0.06, #0.01,
                "gamma"            = 0,    # minimum loss reduction 
                "subsample"        = 0.83, #0.5, # 0.7
                "colsample_bytree" = 0.77, #0.5, # 0.7
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
                "max_depth"        = 6, #changed from default of 8
                "eta"              = 0.023,#0.023,#0.1, # 0.06, #0.01,
                "gamma"            = 0,    # minimum loss reduction 
                "subsample"        = 0.83, #0.5, # 0.7
                "colsample_bytree" = 0.77, #0.5, # 0.7
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
                "max_depth"        = 6, #changed from default of 8
                "eta"              = 0.023,#0.023,#0.1, # 0.06, #0.01,
                "gamma"            = 0,    # minimum loss reduction 
                "subsample"        = 0.83, #0.5, # 0.7
                "colsample_bytree" = 0.77, #0.5, # 0.7
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

########################################

# sqldf("select id, fault_severity,ZeroProb,ZerovsAllProb,ZeroOnevsAllProb,TwovsAllProb  from Fulltrain where id IN(10024,1,10059)")
# sqldf("select id,ZerovsAllProb,ZeroOnevsAllProb,TwovsAllProb from Fulltest where id IN(10000)")
# names(Fulltrain)
featureNames <- names(Fulltrain [-c(1,3)])


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
num.class = length(levels(target))

# names(Fulltrain)
# featurePlot(Fulltrain[c(2)], target, "strip")


# 

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
min.merror.idx = which.min(bst.cv$dt[, test.mlogloss.mean]) 
min.merror.idx
bst.cv$dt[min.merror.idx ,]
#test.mlogloss.mean test.mlogloss.std 0.556998  0.035416 Accuracy : 0.7439 
# 1:            0.435771           0.002387           0.541668          0.033271
#1:             0.337070           0.002493           0.506490          0.020044
#1:             0.330594           0.001691           0.505799          0.021455
#1:             0.321642           0.002057           0.503586          0.021882
#1:             0.325798           0.001945           0.503112          0.020154
#1:             0.32757            0.001977           0.502703          0.021816

# get CV's prediction decoding
pred.cv = matrix(bst.cv$pred, nrow=length(bst.cv$pred)/num.class, ncol=num.class)
pred.cv = max.col(pred.cv, "last")
# confusion matrix
confusionMatrix(factor(y+1), factor(pred.cv))


# real model fit training, with full data
system.time( bst <- xgboost(param=param, data=train.matrix, label=y, 
                            nrounds=min.merror.idx, verbose=1) )
#train-mlogloss:0.455049

# # get the trained model
model = xgb.dump(bst, with.stats=TRUE)
# # get the feature real names
names = dimnames(train.matrix)[[2]]
# # compute feature importance matrix
importance_matrix = xgb.importance(names, model=bst)
print(importance_matrix)

# # plot
gp = xgb.plot.importance(importance_matrix)
print(gp) 
# 
# tree = xgb.plot.tree(feature_names = names, model = bst, n_first_tree = 1)
# print(tree)
# 
# # xgboost predict test data using the trained model
predict <- predict(bst, test.matrix)  
# #head(predict, 10) 



# decode prediction
predict01 = matrix(predict, nrow=num.class, ncol=length(predict)/num.class , byrow=T)
predict01 = t(predict01)
# head(predict01)

colnames(predict01) = classnames

# head(prediction)
# sqldf("select * from prediction where id = 1442")
# names(prediction)

prediction <- cbind( id = Fulltest$id , severity_type = Fulltest$severity_type,  predict_0 = predict01[,2] , predict_1 = predict01[,1], predict_2 = predict01[,3] )

write.csv(prediction, "submission26.csv", quote=FALSE, row.names = FALSE)

#write.table(prediction,file="submission21.csv", append=TRUE,sep=",",col.names=TRUE,row.names=FALSE)


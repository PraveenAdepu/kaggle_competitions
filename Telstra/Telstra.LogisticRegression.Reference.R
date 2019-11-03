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


LogisticRegression <- function(X_train,y,X_test=data.frame(),cv=5,seed=123,metric="auc",importance=0)
  
{
  
  # defining evaluation metric
  
  score <- function(a,b,metric)
    
  {
    
    switch(metric,
           
           accuracy = sum(abs(a-b)<=0.5)/length(a),
           
           auc = auc(a,b),
           
           logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),
           
           mae = sum(abs(a-b))/length(a),
           
           precision = length(a[a==b])/length(a),
           
           rmse = sqrt(sum((a-b)^2)/length(a)),
           
           rmspe = sqrt(sum(((a-b)/a)^2)/length(a))) 
    
  }
  
  if (metric == "auc")
    
  {
    
    library(pROC)
    
  }
  
  cat("Preparing Data\n")
  
  X_train$order <- seq(1, nrow(X_train))
  
  X_train$result <- as.numeric(y)
  
  set.seed(seed)
  
  X_train$randomCV <- floor(runif(nrow(X_train), 1, (cv+1)))
  
  # cross validation
  
  cat(cv, "-fold Cross Validation\n", sep = "")
  
  for (i in 1:cv)
    
  {
    
    X_build <- subset(X_train, randomCV != i, select = -c(order, randomCV))
    
    X_val <- subset(X_train, randomCV == i) 
    
    # building model
    
    model_lr <- glm(result ~., data=X_build, family=binomial())
    
    # predicting on validation data
    
    pred_lr <- predict(model_lr, X_val, type="response")
    
    X_val <- cbind(X_val, pred_lr)
    
    # predicting on test data
    
    if (nrow(X_test) > 0)
      
    {
      
      pred_lr <- predict(model_lr, X_test, type="response")
      
    }
    
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_lr, metric), "\n", sep = "")
    
    # initializing outputs
    
    if (i == 1)
      
    {
      
      output <- X_val
      
      if (nrow(X_test) > 0)
        
      {
        
        X_test <- cbind(X_test, pred_lr)
        
      } 
      
    }
    
    # appending to outputs
    
    if (i > 1)
      
    {
      
      output <- rbind(output, X_val)
      
      if (nrow(X_test) > 0)
        
      {
        
        X_test$pred_lr <- (X_test$pred_lr * (i-1) + pred_lr)/i
        
      } 
      
    }
    
    gc()
    
  } 
  
  # final evaluation score
  
  output <- output[order(output$order),]
  
  cat("\nLogisticRegression ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_lr, metric), "\n", sep = "")
  
  # returning CV predictions and test data with predictions
  
  return(list("train"=output, "test"=X_test)) 
  
}


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

Movings <- sqldf("SELECT id, location , MAX(event_type) Maxevent_type, 
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
                 (MAX(severity_type) - MIN(severity_type) ) Diffseverity_type,
                 
                 COUNT(*) AS RowCount
                 
                 
                 FROM df_all_combinedForMovings GROUP BY id, location")

#dim(Movings)
#sqldf("SELECT * FROM Movings where id = 22")
#Movings$MineventThreshold    <- ifelse(Movings$Minevent_type <= 15, 1, 0)
#Movings$MaxresourceThreshold <- ifelse(Movings$Maxresource_type >= 7, 1, 0)
#Movings$MaxseverityThreshold <- ifelse( Movings$Maxseverity_type <= 2, 1, 0)
#Movings$SDTotalvolume <- sd(Movings$Totalvolume)
#Movings$Totalvolume <- NULL

#unique(Movings$MaxseverityThreshold)
#summary(Movings$Ratio)
Movings[is.na(Movings)] <- 0
#df_all_combinedForMovings$fault_severity

# TrainMovings <- sqldf("SELECT * FROM Movings where fault_severity > -1")

# plot(TrainMovings[c(5,8,9,10,12,13,14)])

# head(Movings,2)

# nums <- sapply(TrainMovings, is.numeric)
# 
#  M <- cor(TrainMovings[-c(1,2,15)])
#  M1 <- cor(TrainMovings[c(5,8)])
# # corrplot(M, method = "circle")
#  corrplot.mixed(M)
#  corrplot.mixed(M1)
# corrplot(M, order = "hclust", addrect = 5)

# sqldf("select * from Movings  where id = 10024 ")

#dim(events)

events       <- sqldf("SELECT id, location, fault_severity, event_type , 1 as [AvgEventVolume],MAX(volume) as [MaxEventVolume],   MIN(volume) as [MinEventVolume] , AVG(volume) as [MeanEventVolume] FROM df_all_combined GROUP BY 1,2,3,4")

resources    <- sqldf("SELECT id, location, fault_severity, resource_type , 1 as [AvgResourceVolume]  FROM df_all_combined GROUP BY 1,2,3,4")

Severities   <- sqldf("SELECT id, location, fault_severity, severity_type , 1 as [AvgSeverityVolume]  FROM df_all_combined GROUP BY 1,2,3,4")

logs         <- sqldf("SELECT id, location, fault_severity, log_feature ,   AVG(volume) as [AvglogVolume] FROM df_all_combined GROUP BY 1,2,3,4")


logevents    <- sqldf("SELECT id, location, fault_severity, 1||log_feature as [log_feature] ,   1 as [logEventVolume] FROM df_all_combined GROUP BY 1,2,3,4")


# eventstd     <- sqldf("SELECT id, location, fault_severity ,  stdev(volume) as [StdEventVolume] FROM df_all_combined GROUP BY 1,2,3")


# eventsvol       <- sqldf("SELECT id, location, fault_severity, event_type+'vol'  as [event_type] , AVG(volume) as [AvgEventVolume],MAX(volume) as [MaxEventVolume],   MIN(volume) as [MinEventVolume] FROM df_all_combined GROUP BY 1,2,3,4")

# head(df_all_combined)
# head(logevents)
# logstats     <- sqldf("SELECT id, location, fault_severity, 
#                       AVG(volume) as [AvglogVolume] ,
#                       MAX(volume) as [MaxlogVolume] ,
#                       MIN(volume) as [MinlogVolume],
#                       Median(volume) as [MedianlogVolume],
#                       (MAX(volume) - MIN(volume))/(case when stdev(volume) = 0 then 1 else stdev(volume) end  ) as AvgDiffVolume
#                      
#                       FROM df_all_combined GROUP BY 1,2,3")


events       <- spread(events,   event_type ,  AvgEventVolume )
events[is.na(events)] <- 0
# dim(events)

events$eventsCount <- rowSums(events[,7:59]>0)
# 
events$CountToTotalevents <- (events$eventsCount/53)

# sqldf("select distinct * from eventstd  where id = 22 ")

resources    <- spread(resources,   resource_type ,  AvgResourceVolume )
resources[is.na(resources)] <- 0
#sqldf("select * from resources  where id = 22 ")

# names(resources)

resources$resourcesCount <-rowSums(resources[,4:13]>0) 
# 
resources$CountToTotalresources <- (resources$resourcesCount/10)


Severities    <- spread(Severities,   severity_type ,  AvgSeverityVolume )
Severities[is.na(Severities)] <- 0

# sqldf("select * from Severities  where id = 22 ")

logs    <- spread(logs,   log_feature ,  AvglogVolume )
logs[is.na(logs)] <- 0


# sqldf("select * from logs  where id = 22 ")

# names(logs)
logs$logsCount <- rowSums(logs[,names(logs[-c(1:3)])])
# 
logs$logsNoCount <- rowSums(logs[,4:389]>0)

logevents    <- spread(logevents,   log_feature ,  logEventVolume )
logevents[is.na(logevents)] <- 0

# logstats[is.na(logstats)] <- 0
# 
# head(logstats)

# sessionsdatas <- NULL
# df_all_combineds <- NULL
# merging data
sessionsdatas    <- merge(events,logs,by=(c("id" , "location" , "fault_severity"))    ,all = T)  
sessionsdatas    <- merge(sessionsdatas,resources,by=(c("id" , "location" , "fault_severity")) ,all = T)
sessionsdatas    <- merge(sessionsdatas,Severities,by=(c("id" , "location" , "fault_severity")) ,all = T)

sessionsdatas    <- merge(sessionsdatas,logevents,by=(c("id" , "location" , "fault_severity")) ,all = T)

#dim(sessionsdatas)

# count(unique(sessionsdatas$id))
# 
# dim(sessionsdata)
# dim(df_all)

df_all_combineds <- merge(df_all,sessionsdatas,by=(c("id" , "location" , "fault_severity")) ,all = T)

# dim(df_all_combineds)
# 
# dim(Movings)

df_all_combineds <- merge(df_all_combineds,Movings,by=(c("id" , "location" )) ,all = T)

df_all_combineds$EventRatio <- df_all_combineds$eventsCount/df_all_combineds$RowCount
df_all_combineds$ResourceRatio <- df_all_combineds$resourcesCount/df_all_combineds$RowCount
df_all_combineds$LogsRatio <- df_all_combineds$logsCount/df_all_combineds$RowCount


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

# names(Fulltrain)
# names(Fulltest[c(2:456)])

sqldf("select id,location,RowCount,logsCount,MaxEventVolume,MinEventVolume,MeanEventVolume,
                logsNoCount,eventsCount,resourcesCount,
                Maxevent_type,
                Minevent_type,
                Diffevent_type,
                Maxlog_feature,
                Minlog_feature,
                Difflog_feature,
                Maxresource_type,
                Minresource_type,
                EventRatio,
                ResourceRatio,
                LogsRatio,
                LogTotalToResource,
                fault_severity
      from df_all_combineds  ")

# Fulltrain$lmtarget <- ifelse(Fulltrain$fault_severity== 0,1, 0)

featureNames <- c('location','RowCount','logsVolume','MaxlogVolume','MinlogVolume','MeanlogVolume',
           'logsCount','eventsCount','resourcesCount',
           'Maxevent_type',
           'Minevent_type',
           'Diffevent_type',
           'Maxlog_feature',
           'Minlog_feature',
           'Difflog_feature',
           'Maxresource_type',
           'Minresource_type',
           'Maxseverity_type'
           
           )

#featureNames <- names(Fulltrain[, names])

X_train <- Fulltrain[, featureNames]

X_test <- Fulltest[, featureNames]

target <- ifelse(Fulltrain$fault_severity== 0,1, 0)

model_lr <- LogisticRegression(X_train, target, X_test, cv=3, seed=235, metric="logloss")


Fulltrain$ZeroProb <- model_lr$train$pred_lr
Fulltest$ZeroProb <- model_lr$test$pred_lr



#featureNames <- names(Fulltrain[-c(1,3,480)])

X_train <- Fulltrain[, featureNames]

X_test <- Fulltest[, featureNames]

target <- ifelse(Fulltrain$fault_severity<= 1,1, 0)

model_lr <- LogisticRegression(X_train, target, X_test, cv=3, seed=235, metric="logloss")


Fulltrain$OneProb <- model_lr$train$pred_lr
Fulltest$OneProb <- model_lr$test$pred_lr
# 
# #featureNames <- names(Fulltrain [-c(1,3,480,481)])
# 
# X_train <- Fulltrain[, featureNames]
# 
# X_test <- Fulltest[, featureNames]
# 
# target <- ifelse(Fulltrain$fault_severity== 2,1, 0)
# 
# model_lr <- LogisticRegression(X_train, target, X_test, cv=3, seed=235, metric="logloss")
# 
# 
# Fulltrain$TwoProb <- model_lr$train$pred_lr
# Fulltest$TwoProb <- model_lr$test$pred_lr


Fulltrain$ZeroProb <- Fulltrain$ZeroProb * 1
Fulltest$ZeroProb <- Fulltest$ZeroProb   * 1

Fulltrain$OneProb <- Fulltrain$OneProb * 2
Fulltrain$OneProb <- Fulltrain$OneProb * 2


Fulltrain$TwoProb <- Fulltrain$TwoProb * 3
Fulltrain$TwoProb <- Fulltrain$TwoProb * 3

Fulltrain$allProb <- Fulltrain$ZeroProb + Fulltrain$OneProb 
Fulltest$allProb <- Fulltest$ZeroProb + Fulltest$OneProb 



# head(Fulltrain)

# Fulltrain$lmtarget <- NULL

# head(train.matrix,2)
# 
# sapply(train.matrix, class)

# mode(train.matrix) = "numeric"
# mode(test.matrix) = "numeric"

# head(train.matrix,2)


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

featureNames <- names(Fulltrain [-c(1,3,483)])

# names(Fulltrain)
train.matrix <- as.matrix(Fulltrain[,featureNames])#, Fulltrain$severity_type == 3))
test.matrix  <- as.matrix(Fulltest[,featureNames]) #, Fulltest$severity_type == 3))

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


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
sqldf("select * from resources  where id = 10024 ")

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

# names(Fulltrain)
# names(Fulltest[c(2:456)])

featureNames <- names(Fulltrain[-c(1,3)])

# names(Fulltrain)
train.matrix <- as.matrix(Fulltrain[,featureNames])#, Fulltrain$severity_type == 3))
test.matrix  <- as.matrix(Fulltest[,featureNames]) #, Fulltest$severity_type == 3))



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


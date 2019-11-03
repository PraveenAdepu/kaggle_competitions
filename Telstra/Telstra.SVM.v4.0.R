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
require(e1071)
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


# head(df_all_combined,2)
# sqldf("select * from df_all_combined  where id = 10024 ")


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
# sqldf("select * from resources  where id = 10024 ")

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

df_all_combineds$location <- as.numeric(gsub("location",'',df_all_combineds$location))

Fulltrain  <- df_all_combineds[which(df_all_combineds$fault_severity > -1), ]
Fulltest   <- df_all_combineds[which(df_all_combineds$fault_severity <  0), ]

Fulltest$fault_severity  <- NULL 

# names(Fulltrain)
# names(Fulltest[c(2:456)])

featureNames <- names(Fulltrain[-c(1)])

TestfeatureNames <- names(Fulltest[-c(1)])

Fulltrain$fault_severity <- ifelse(Fulltrain$fault_severity==0,'Zero', ifelse(Fulltrain$fault_severity==1,'One', 'Two'))

Fulltrain$fault_severity  <- as.factor(Fulltrain$fault_severity)

# SVM Model training using direct svm using e1071

# model <- svm(fault_severity ~ ., data = Fulltrain[,featureNames], probability=TRUE)
# 
# 
# predictSVM <- predict(model, Fulltest[,TestfeatureNames], probability=TRUE)
# 
# ClassProb <- attr(predictSVM, "probabilities")
# 
# dim(ClassProb)
# 
# prediction <- cbind( id = Fulltest$id , severity_type = Fulltest$severity_type,  predict_0 = ClassProb[,2] , predict_1 = ClassProb[,1], predict_2 = ClassProb[,3] )
# 
# head(prediction)
# write.csv(prediction, "submission23.csv", quote=FALSE, row.names = FALSE)



ctr <- trainControl(method='cv',
                    number=5,
                    #repeats=3, 
                    classProbs=TRUE,
                    savePred=T

                    )
# Recall as C increases, the margin tends to get wider
grid <- data.frame(C=seq(0.01,5,0.5),sigma = seq(1,5,1))

svm.c <- train(fault_severity ~., 
               data = Fulltrain[,featureNames],
               method= 'svmRadial', #'svmPoly', #
               #preProc=c('center','scale'),
               trControl=ctr,
               tuneGrid=grid,
               probability=TRUE,
               verbose = TRUE)

summary(svm.c)
head(svm.c$pred)

# Training error rate
confusionMatrix(predict(svm.c, Fulltrain[,featureNames]), Fulltrain$fault_severity)
# Accuracy : 0.6931

# predictedClasses <- predict(svm.c, Fulltest[,TestfeatureNames] )
predictedProbs   <- predict(svm.c, Fulltest[,TestfeatureNames], type = "prob")

head(predictedProbs)

 prediction <- cbind( id = Fulltest$id ,  predict_0 = predictedProbs[,3] , predict_1 = predictedProbs[,1], predict_2 = predictedProbs[,2] )
# 
# head(prediction)
# write.csv(prediction, "submission24.csv", quote=FALSE, row.names = FALSE)
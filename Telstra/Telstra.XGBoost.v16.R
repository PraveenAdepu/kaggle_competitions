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
setwd("C:/Users/SriPrav/Documents/R/01Telstra/")

setwd("C:/Users/SriPrav/Documents/R/01Telstra")
root_directory = "C:/Users/SriPrav/Documents/R/01Telstra"


# 02. Set Seed
# you must know why I am using set.seed()
set.seed(546)

# 03. Import source files data
# Importing data into R
train       <- read.csv("./SourceFiles/train.csv"        , h=TRUE, sep=",")
test        <- read.csv("./SourceFiles/test.csv"         , h=TRUE, sep=",")
event       <- read.csv("./SourceFiles/event_type.csv"   , h=TRUE, sep=",")
log         <- read.csv("./SourceFiles/log_feature.csv"  , h=TRUE, sep=",")
resource    <- read.csv("./SourceFiles/resource_type.csv", h=TRUE, sep=",")
severity    <- read.csv("./SourceFiles/severity_type.csv", h=TRUE, sep=",")


# 04. Set target variable to test data
test$fault_severity <- -1

df_all  <-  rbind(train,test)


# merging data
Moves                     <- merge(event,log,by="id"      ,all = T)  
Moves                     <- merge(Moves,resource,by="id" ,all = T)
Moves                     <- merge(Moves,severity,by="id" ,all = T)
df_all_combinedForMovings <- merge(df_all,Moves,by="id"   ,all = T)

# filter(df_all_combinedForMovings, id == "7350" & resource_type =="resource_type 8" & event_type =="event_type 15")

df_all_combinedForMovings$location      <- as.numeric(gsub("location ","",df_all_combinedForMovings$location))
df_all_combinedForMovings$event_type    <- as.numeric(gsub("event_type ","",df_all_combinedForMovings$event_type))
df_all_combinedForMovings$log_feature   <- as.numeric(gsub("feature ","",df_all_combinedForMovings$log_feature))
df_all_combinedForMovings$resource_type <- as.numeric(gsub("resource_type ","",df_all_combinedForMovings$resource_type))
df_all_combinedForMovings$severity_type <- as.numeric(gsub("severity_type ","",df_all_combinedForMovings$severity_type))


Movings <- sqldf("SELECT id, 
-- MAX(fault_severity) fault_severity,
                 --MAX(location) location, 
                 --MAX(event_type) Maxevent_type, 
                 MAX(log_feature) Maxlog_feature, 
                 MAX(resource_type) Maxresource_type, 
                 MAX(severity_type) Maxseverity_type, 
                 MIN(event_type) Minevent_type, 
                 MIN(log_feature) Minlog_feature, 
                 MIN(resource_type) Minresource_type, 
                 --MIN(severity_type) Minseverity_type,
                 --MAX(volume) Maxvolume,
                 --MIN(volume) Minvolume,
                 --AVG(volume) Avgvolume,
                 --AVG(log_feature) Sumlog_feature,
                 
                 COUNT(*) + Max(resource_type) + MIN(resource_type) MixFeature, 
                 COUNT(*) as Rows
                 
                 FROM df_all_combinedForMovings group by id ")

require(GGally)
head(Movings)
# ggpairs(Movings[-c(1)], colour="fault_severity", alpha=0.4)


event$eventtype <- as.integer(gsub("event_type ","",event$event_type))
# head(event)
events       <- spread(event,   event_type ,  eventtype )
# head(events)
# sqldf("select * from events where id = 10024")
events[is.na(events)] <- 0

resource$resourcetype <- as.integer(gsub("resource_type ","",resource$resource_type))
# head(resource)
resources       <- spread(resource,   resource_type ,  resourcetype )
# head(resources)
# sqldf("select * from resources where id = 10024")
resources[is.na(resources)] <- 0

severity$severitytype <- as.integer(gsub("severity_type ","",severity$severity_type))
# head(severity)
severities       <- spread(severity,   severity_type ,  severitytype )
# head(severities)
# sqldf("select * from severities where id = 10024")
severities[is.na(severities)] <- 0

log$logfeature <- as.integer(gsub("feature ","",log$log_feature))
logDetails <- log
log$volume <- NULL
# head(log)
logs       <- spread(log,   log_feature ,  logfeature )
# names(events)
# sqldf("select * from logs where id = 10024")
logs[is.na(logs)] <- 0

logDetails <- sqldf("select id , sum(logfeature) Totallogfeatures , sum(volume) as Totalvolume from logDetails group by id")

# logs$logsVolume          <- rowSums(logs[,2:387])
# resources$resourcesTotal <-rowSums(resources[,2:11])
# events$eventsTotal       <- rowSums(events[,2:54])
# events$severitiesTotal       <- rowSums(severities[,2:6])



sessionsdata    <- merge(events,logs,by="id"      ,all = T) 
dim(sessionsdata)
sessionsdata    <- merge(sessionsdata,resources,by="id" ,all = T)
dim(sessionsdata)
sessionsdata    <- merge(sessionsdata,severities,by="id" ,all = T)
dim(sessionsdata)

dim(df_all)
df_all_combined <- merge(df_all,sessionsdata,by="id" ,all = T)

df_all_combined <- merge(df_all_combined, Movings, by="id"  ,all = T)

df_all_combined <- merge(df_all_combined, logDetails, by="id"  ,all = T)

# Locationfrequencies <- sqldf("select location , count(*) as LocationFreq from df_all_combined group by location ")
# df_all_combined <- merge(df_all_combined,Locationfrequencies,by="location" ,all.x = T)

dim(df_all_combined)

df_all_combined$location     <- as.numeric(gsub("location",'',df_all_combined$location))
df_all_combined$Totallogfeatures <- log(df_all_combined$Totallogfeatures)
df_all_combined$TotalvolumeBins <- ifelse(df_all_combined$Totalvolume > 750 , 0, 1)
#df_all_combined$FeatureTotal <-   df_all_combined$logsVolume + df_all_combined$resourcesTotal + df_all_combined$severitiesTotal
#df_all_combined$SeverityVolume <- log(df_all_combined$logsVolume * df_all_combined$severitiesTotal)

names(df_all_combined)

# Cl <- kmeans(df_all_combined[-c(1,3)], 10, nstart=25)
# 
# df_all_combined <- cbind(df_all_combined, Cluster = Cl$cluster)


# sqldf("select id, fault_severity from df_all_combined where id IN(10024,1,10059)")
Fulltrain  <- df_all_combined[which(df_all_combined$fault_severity > -1), ]
Fulltest   <- df_all_combined[which(df_all_combined$fault_severity <  0), ]

Fulltest$fault_severity  <- NULL 


# #######################################################################################################
# # Data Visualisations
## 3D Scatterplot
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

# M <- cor(Fulltrain[c(3,462:468,470:483)])
# corrplot.mixed(M)
# conputationally expensive
# require(GGally)
# ggpairs(Fulltrain[c(3,458:467)], colour="fault_severity", alpha=0.4)
summary(Fulltrain$'resource_type 1')
# head(Fulltrain[c(3,443:457)])
# 
# filter(Fulltrain, id == "7350")

filter(log, id == "7350")

# ggpairs(Fulltrain[c(3,443:457)], colour="fault_severity", alpha=0.4)


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


unique(target)

# xgboost parameters
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = num.class,    # number of classes 
              "eval_metric" = "mlogloss",    # evaluation metric 
              "nthread" = 24,   # number of threads to be used 
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
nround.cv = 20
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
min.merror.idx = 20
xgb.grid <- expand.grid(nrounds = c(20), # try with 195, get best nround from XGBoost model then apply here for Caret grid
                        eta = c(0.05),
                        max_depth = c(8),
                        colsample_bytree = c(0.7),
                        subsample = c(0.6),
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
                 #tuneGrid=xgb.grid,
                 verbose=1,
                 metric="logLoss",
                 nthread =24,
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

predict01 <- as.table(prediction)
head(predict01)
filter(predict01, id == "2")
predictions <- sqldf("select id , avg(predict_0) as predict_0 
                     , avg(predict_1) as predict_1
                     , avg(predict_2) as predict_2
                     from predict01")

write.csv(prediction, "submissionFullSet.csv", quote=FALSE, row.names = FALSE)

FullSet    <- read.csv("./Telstra/submissionFullset.csv", h=TRUE, sep=",")

submissionTest <- sqldf("select id , avg(predict_0) as predict_0 
                        , avg(predict_1) as predict_1
                        , avg(predict_2) as predict_2
                        from FullSet group by id")

write.csv(submissionTest, "submission48.csv", quote=FALSE, row.names = FALSE)

# 01. Libraries
#rm(list=ls())
setwd("C:/Users/prav/Documents/R")
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

# merging data

head(train,2)
head(test ,2)

# 04. Set target variable to test data
test$fault_severity <- -1

head(train,2)
head(test ,2)

dim(train)
dim(test)

# 05. Union train and test datasets
data  <- rbind(train,test)

severity$severity_type <- gsub("severity_type ","",severity$severity_type)
severity$severity_type <- as.integer(severity$severity_type)
head(severity)

dim(data)

# 06. MERGE all data sets
data  <- merge(data,severity,by="id",all = T)

head(data)


eventdata     <- merge(data,event,by="id"   ,all = T)  
logdata       <- merge(data,log,by="id"     ,all = T)
resourcedata  <- merge(data,resource,by="id",all = T)



filter(data, id==10024)
filter(eventdata, id==10024)
filter(logdata, id==10024)
filter(resourcedata, id==10024)


dataeventtypecount <- sqldf("select *, count(*) event_type_count from eventdata --where id==10024 
                            group by 1,2,3,4,5 ")

dim(dataeventtypecount)
head(dataeventtypecount)

filter(resourcedata, id==10024)

dataresourcetypecount <- sqldf("select *, count(*) resource_type_count from resourcedata --where id==10024
                               group by 1,2,3,4,5 ")

dim(dataresourcetypecount)

filter(datareseveritytype, id==10024)

# datareseveritytypecount <- sqldf("select *, count(*) severity_type_count from datareseveritytype --where id==10024
#                               group by 1,2,3,4,5 ")

head(dataeventtypecount)


normeventtypes <- cast(dataeventtypecount, id + location +  fault_severity + severity_type  ~ event_type)

head(normeventtypes)

normresourcetypes <- cast(dataresourcetypecount, id + location +  fault_severity + severity_type  ~ resource_type)

head(normresourcetypes)

#normrseveritytypes <- cast(datareseveritytypecount, id + location +  fault_severity  ~ severity_type)

#head(normrseveritytypes)
head(logdata)
logdata$volume <- as.integer(logdata$volume)

normrlogfeatures <- cast(logdata, id + location +  fault_severity + severity_type  ~ log_feature , fun.aggregate=sum)

filter(normrlogfeatures, id==10024)

# 06. MERGE all data sets
Normdata  <- merge(normeventtypes,normresourcetypes,by=c("id", "location","fault_severity", "severity_type")   ,all = T)  
#Normdata  <- merge(Normdata,normrseveritytypes,by=c("id", "location","fault_severity", "severity_type")     ,all = T)
Normdata  <- merge(Normdata,normrlogfeatures,by=c("id", "location","fault_severity", "severity_type"),all = T)

names(Normdata)
filter(Normdata , id == 10024)

sapply(Normdata, class)

# 07. Convert blank and empty values
# converting NA in to '0' and '" "' for mode Matrix Generation

for(i in 1:ncol(Normdata)){
  if(is.numeric(Normdata[,i])){
    Normdata[is.na(Normdata[,i]),i] = 0
  }else{
    Normdata[,i] = as.character(Normdata[,i])
    Normdata[is.na(Normdata[,i]),i] = " "
    Normdata[,i] = as.factor(Normdata[,i])
  }
}

Normdata$severity_type <- as.factor(Normdata$severity_type)


Fulltrain  <- Normdata[which(Normdata$fault_severity > -1), ]
Fulltest   <- Normdata[which(Normdata$fault_severity <  0), ]

Fulltest$fault_severity  <- NULL 

names(Fulltrain [c(4:453)])
names(Fulltest[c(3:452)])

#dropping some more variables

Fulltrain$severity_type <- gsub("severity_type ","",Fulltrain$severity_type)
Fulltest$severity_type <- gsub("severity_type ","",Fulltest$severity_type)

Fulltrain$location <- gsub("location ","",Fulltrain$location)
Fulltest$location <- gsub("location ","",Fulltest$location)

Fulltrain$severity_type <- as.factor(Fulltrain$severity_type)
Fulltest$severity_type <- as.factor(Fulltest$severity_type)

Fulltrain$location <- as.factor(Fulltrain$location)
Fulltest$location <- as.factor(Fulltest$location)

names(Fulltrain[c(4:453)]) # 453
names(Fulltest[c(3:452)])

train.matrix <- as.matrix(filter(Fulltrain[c(4:453)]))#, Fulltrain$severity_type == 3))
test.matrix  <- as.matrix(filter(Fulltest[c(3:452)])) #, Fulltest$severity_type == 3))

head(train.matrix,2)

sapply(train.matrix, class)

mode(train.matrix) = "numeric"
mode(test.matrix) = "numeric"

head(train.matrix,2)

testing <- filter(Fulltrain, Fulltrain$severity_type == 3)

target <- ifelse(Fulltrain$fault_severity==0,'Zero', ifelse(Fulltrain$fault_severity==1,'One', 'Two'))

classnames = unique(target)

#target = as.integer(colsplit(target,'_',names=c('x1','x2'))[,2])

target  <- as.factor(target)

outcome.org = Fulltrain$fault_severity
outcome = outcome.org 
levels(outcome)


y = target

y = as.matrix(as.integer(target)-1)

num.class = length(levels(target))

# check for zero variance
zero.var = nearZeroVar(Fulltrain[c(2,4:453)], saveMetrics=TRUE)
zero.var[zero.var[,"zeroVar"] == 0, ]
nzv <- zero.var[zero.var[,"zeroVar"] + zero.var[,"nzv"] > 0, ] 
zero.var
filter(zero.var, nzv$zeroVar == FALSE)
badCols <- nearZeroVar(Fulltrain[c(2,4:453)])
print(paste("Fraction of nearZeroVar columns:", round(length(badCols)/length(Fulltrain[c(2,4:453)]),4)))

# remove those "bad" columns from the training and cross-validation sets

train <- train[, -badCols]
test <- test[, -badCols]

# corrPlot
featurePlot(totaltrain[c(2,458,460:462)], outcome.org, "strip")

head(train.matrix)
# xgboost parameters
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = num.class,    # number of classes 
              "eval_metric" = "mlogloss",    # evaluation metric 
              "nthread" = 4,   # number of threads to be used 
              "max_depth" = 10,    # maximum depth of tree # 6
              "eta" = 0.1,    # step size shrinkage  # 0.5
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 0.5,    # part of data instances to grow tree # 0.5
              "colsample_bytree" = 0.5,  # subsample ratio of columns when constructing each tree 
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

# get CV's prediction decoding
pred.cv = matrix(bst.cv$pred, nrow=length(bst.cv$pred)/num.class, ncol=num.class)
pred.cv = max.col(pred.cv, "last")
# confusion matrix
confusionMatrix(factor(y+1), factor(pred.cv))


# real model fit training, with full data
system.time( bst <- xgboost(param=param, data=train.matrix, label=y, 
                            nrounds=min.merror.idx, verbose=1) )

# get the trained model
model = xgb.dump(bst, with.stats=TRUE)
# get the feature real names
names = dimnames(train.matrix)[[2]]
# compute feature importance matrix
importance_matrix = xgb.importance(names, model=bst)
print(importance_matrix)
# plot
gp = xgb.plot.importance(importance_matrix)
print(gp) 

tree = xgb.plot.tree(feature_names = names, model = bst, n_first_tree = 1)
print(tree)

# xgboost predict test data using the trained model
predict <- predict(bst, test.matrix)  
#head(predict, 10) 



# decode prediction
predict01 = matrix(predict, nrow=num.class, ncol=length(predict)/num.class , byrow=T)
predict01 = t(predict01)


colnames(predict01) = classnames

head(predict01)

names(predict01)

prediction <- cbind( id = Fulltest$id , severity_type = Fulltest$severity_type,  predict_0 = predict01[,2] , predict_1 = predict01[,1], predict_2 = predict01[,3] )

prediction00 <- data.frame(prediction)
filter(prediction00 , severity_type == 3)


write.table(prediction,file="TelstraPredictions21.csv", append=TRUE,sep=",",col.names=TRUE,row.names=FALSE)



# rm(list=ls())
require(data.table)
require(Matrix)
require(xgboost)
require(sqldf)
require(plyr)
require(dplyr)
require(ROCR)
require(Metrics)
require(pROC)
require(caret)
require(readr)
require(moments)
require(forecast)

train <- read.csv("./Toshipping/train_stacknet05.csv",  header = FALSE)
test  <- read.csv("./Toshipping/test_stacknet05.csv", header=FALSE)

pred  <- read.csv("./Toshipping/sigma_stack_pred05.csv", header=FALSE)

names(train)
Pred_file <- test[1]
names(train)
head(test)
head(pred)
head(Pred_file)

Pred_file <- cbind(Pred_file, pred)

head()
names(Pred_file) <- c("listing_id","high","medium","low")

write.csv(Pred_file, "./submissions/Prav_stacknet05.csv", row.names = FALSE)

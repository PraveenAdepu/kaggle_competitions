
setwd("C:/Users/SriPrav/Documents/R/19DSB2017")
root_directory = "C:/Users/SriPrav/Documents/R/19DSB2017"

# paste(root_directory, "/input/events.csv", sep='')

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

metric = "logloss"

normalit<-function(m){
  (m - min(m))/(max(m)-min(m))
} #train_test_combined[,cont.variables] <- apply(train_test_combined[,cont.variables], 2, normalit)

xgb.metric.mae <- function (preds, dtrain) {
  label = getinfo(dtrain, "label")
  #err= mae(exp(label),exp(preds) )
  err= mae(label,preds )
  return (list(metric = "mae", value = err))
}

xgb.metric.log.mae <- function (preds, dtrain) {
  label = getinfo(dtrain, "label")
  err= mae(exp(label)-constant,exp(preds)-constant )
  return (list(metric = "mae", value = err))
}

# library(devtools); install('win-library/3.2/xgboost/R-package')
# install.packages("drat", repos="https://cran.rstudio.com")
# drat:::addRepo("dmlc")
# install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")


xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

logcoshobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  grad <- tanh(preds-labels)
  hess <- 1-grad*grad
  return(list(grad = grad, hess = hess))
}

cauchyobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  c <- 3  #the lower the "slower/smoother" the loss is. Cross-Validate.
  x <-  preds-labels
  grad <- x / (x^2/c^2+1)
  hess <- -c^2*(x^2-c^2)/(x^2+c^2)^2
  return(list(grad = grad, hess = hess))
}


fairobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  c <- 2 #the lower the "slower/smoother" the loss is. Cross-Validate.
  x <-  preds-labels
  grad <- c*x / (abs(x)+c)
  hess <- c^2 / (abs(x)+c)^2
  return(list(grad = grad, hess = hess))
}

# 
# 
# setwd("C:/Users/SriPrav/Documents/R/19DSB2017")
# root_directory = "C:/Users/SriPrav/Documents/R/19DSB2017"
# 
# # rm(list=ls())
# 
# require(data.table)
# require(Matrix)
# require(xgboost)
# require(sqldf)
# require(plyr)
# require(dplyr)
# require(ROCR)
# require(Metrics)
# require(pROC)
# require(caret)
# require(readr)
# require(moments)
# require(forecast)
# 
# #####################################################################################################
# 
# 
# sample_sub <- read_csv("./submissions/stage1_solution.csv")
# 
# CNN02.fold1.test <- read_csv("./submissions/Prav.FE01.bl02_Meanfolds12345.csv")
# CNN02.fold2.test <- read_csv("./submissions/Prav.FE01.nn01_Meanfolds12345.csv")
# CNN02.fold3.test <- read_csv("./submissions/Prav.FE00.bl02_Meanfolds12345.csv")
# CNN02.fold4.test <- read_csv("./submissions/Prav.FE00.nn01_Meanfolds12345.csv")
# CNN02.fold5.test <- read_csv("./submissions/Prav.FE01.bl02_nn01_meanfolds12345_Mean.csv")
# CNN02.fold6.test <- read_csv("./submissions/Prav.FE01.bl02_FE00.bl02_meanfolds_Mean.csv")
# CNN02.fold7.test <- read_csv("./submissions/Prav.FE01.nn01_FE00.nn01_meanfolds_Mean.csv")
# 
# names(CNN02.fold1.test)[2]<- "fold1cancer"
# names(CNN02.fold2.test)[2]<- "fold2cancer"
# names(CNN02.fold3.test)[2]<- "fold3cancer"
# names(CNN02.fold4.test)[2]<- "fold4cancer"
# names(CNN02.fold5.test)[2]<- "fold5cancer"
# names(CNN02.fold6.test)[2]<- "fold6cancer"
# names(CNN02.fold7.test)[2]<- "fold7cancer"
# 
# 
# test_sub <- left_join(sample_sub, CNN02.fold1.test, by="id")
# test_sub <- left_join(test_sub,   CNN02.fold2.test, by="id")
# test_sub <- left_join(test_sub,   CNN02.fold3.test, by="id")
# test_sub <- left_join(test_sub,   CNN02.fold4.test, by="id")
# test_sub <- left_join(test_sub,   CNN02.fold5.test, by="id")
# test_sub <- left_join(test_sub,   CNN02.fold6.test, by="id")
# test_sub <- left_join(test_sub,   CNN02.fold7.test, by="id")
# 
# head(test_sub)
# 
# score(test_sub$cancer, test_sub$fold1cancer, metric)
# score(test_sub$cancer, test_sub$fold2cancer, metric)
# score(test_sub$cancer, test_sub$fold3cancer, metric)
# score(test_sub$cancer, test_sub$fold4cancer, metric)
# score(test_sub$cancer, test_sub$fold5cancer, metric)
# score(test_sub$cancer, test_sub$fold6cancer, metric)
# score(test_sub$cancer, test_sub$fold7cancer, metric)
# 
# # > score(test_sub$cancer, test_sub$fold1cancer, metric)
# # [1] 0.5758206
# # > score(test_sub$cancer, test_sub$fold2cancer, metric)
# # [1] 0.5962806
# # > score(test_sub$cancer, test_sub$fold3cancer, metric)
# # [1] 0.5709221
# # > score(test_sub$cancer, test_sub$fold4cancer, metric)
# # [1] 0.5896344
# # > score(test_sub$cancer, test_sub$fold5cancer, metric)
# # [1] 0.5809324
# # > score(test_sub$cancer, test_sub$fold6cancer, metric)
# # [1] 0.5649411
# # > score(test_sub$cancer, test_sub$fold7cancer, metric)
# # [1] 0.5874497
# # >

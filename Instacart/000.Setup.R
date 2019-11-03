
setwd("C:/Users/SriPrav/Documents/R/26Instacart")
root_directory = "C:/Users/SriPrav/Documents/R/26Instacart"

# paste(root_directory, "/input/events.csv", sep='')

# rm(list=ls())

require(data.table)
require(Matrix)
require(xgboost)
require(sqldf)
require(plyr)
require(dplyr)
require(ROCR)
require(pROC)
require(readr)
require(caret)





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

require(ModelMetrics)

print("Mean F1 Score for use with XGBoost")
xgb_eval_f1 <- function (yhat, dtrain) {

  y = getinfo(dtrain, "label")
  dt <- data.table(user_id=X_val$user_id, purch=y, pred=yhat)
  f1 <- mean(dt[,.(f1score=f1Score(purch, pred, cutoff=0.2)), by=user_id]$f1score)
  return (list(metric = "f1", value = f1))
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

auc_probability <- function(labels, scores, N=1e7){
  pos <- sample(scores[labels], N, replace=TRUE)
  neg <- sample(scores[!labels], N, replace=TRUE)
  # sum( (1 + sign(pos - neg))/2)/N # does the same thing
  (sum(pos > neg) + sum(pos == neg)/2) / N # give partial credit for ties
}

features.target.AUC.score <- function(trainingSet, feature.names, target){
  features.AUC.score <- data.frame(
    feature=character(), 
    AUCScore=double()
  ) 
  for ( feature in feature.names) {
    
    AUC.score = auc_probability(as.logical(trainingSet[[target]]), trainingSet[[feature]])
    
    present.score <- data.frame(feature=feature, AUCScore = AUC.score)
    
    features.AUC.score <- rbind(features.AUC.score,present.score)
    
  }
  features.AUC.score <- features.AUC.score %>%
                          arrange(AUCScore)
  return(features.AUC.score)
}



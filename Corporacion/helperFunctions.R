

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

xgb.metric.qwk <- function (preds, dtrain) {
  label = getinfo(dtrain, "label")
  preds = round(preds)
  #err= mae(exp(label),exp(preds) )
  err= ScoreQuadraticWeightedKappa(label,preds ,0,20)
  return (list(metric = "kappa", value = err))
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


Create5Folds_Classification <- function(train, CVSourceColumn, RandomSample, RandomSeed)
{
  set.seed(RandomSeed)
  if(RandomSample)
  {
    train <- as.data.frame(train[sample(1:nrow(train)), ])
    
  }
  for(i in 1:length(CVSourceColumn))
  {
    names(train)[i] <- CVSourceColumn[i]
  }
  
  folds <- createFolds(train[[CVSourceColumn[2]]], k = 5) # Assuming Classification flag is in 2 position of columns list
  
  trainingFold01 <- as.data.frame(train[folds$Fold1, ])
  trainingFold01$CVindices <- 1
  
  trainingFold02 <- as.data.frame(train[folds$Fold2, ])
  trainingFold02$CVindices <- 2
  
  trainingFold03 <- as.data.frame(train[folds$Fold3, ])
  trainingFold03$CVindices <- 3
  
  trainingFold04 <- as.data.frame(train[folds$Fold4, ])
  trainingFold04$CVindices <- 4
  
  trainingFold05 <- as.data.frame(train[folds$Fold5, ])
  trainingFold05$CVindices <- 5
  
  for(i in 1:length(CVSourceColumn))
  {
    names(trainingFold01)[i] <- CVSourceColumn[i]
    names(trainingFold02)[i] <- CVSourceColumn[i]
    names(trainingFold03)[i] <- CVSourceColumn[i]
    names(trainingFold04)[i] <- CVSourceColumn[i]
    names(trainingFold05)[i] <- CVSourceColumn[i]
  }
  
  
  
  trainingFolds <- rbind(trainingFold01, trainingFold02 , trainingFold03, trainingFold04, trainingFold05 )
  rm(trainingFold01,trainingFold02,trainingFold03,trainingFold04,trainingFold05); gc()
  
  
  return(trainingFolds)
}


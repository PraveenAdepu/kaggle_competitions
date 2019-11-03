

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


qwKappa <- function(rate_obs, rate_pred, min_rate = NULL, max_rate = NULL) {
  if (is.null(min_rate)) {
    min_rate <- 0
  }
  if (is.null(max_rate)) {
    max_rate <- max(rate_obs)
  }
  rate_pred <- round(pmax(pmin(rate_pred, max_rate), min_rate))
  
  # Transform rates to begin in 1
  rates <- data.table('A' = rate_pred - min_rate + 1, 'B' = rate_obs - min_rate + 1) 
  max_rate <- max_rate - min_rate + 1
  
  # Confusion matrix
  confusion <- data.table(expand.grid(A = seq(1:max_rate), B = seq(1:max_rate)))
  
  # Join with observed frequency
  rates.freq <- rates[, .N, by = .(A, B)]
  confusion <- merge(confusion, rates.freq, by = c('A', 'B'), all.x = TRUE)
  confusion[is.na(N), N := 0]  
  
  # Weights
  confusion[, W := (A - B)^2]
  
  # Marginals
  confusion[, MB := sum(N) / nrow(rates), by = B]
  confusion[, MA := sum(N) / nrow(rates), by = A]
  
  # Expected
  confusion[, E := MA * MB * nrow(rates)]
  
  # qwKappa  
  1 - sum(confusion$W * confusion$N) / sum(confusion$W * confusion$E)
}

xgb.metric.qwk <- function (preds, dtrain) {
  label = getinfo(dtrain, "label")
  #preds = round(preds)
  #err= mae(exp(label),exp(preds) )
  err = qwKappa(label,preds ,0,20)
  #err= ScoreQuadraticWeightedKappa(label,preds ,0,20)
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
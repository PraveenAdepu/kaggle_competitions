
setwd("C:/Users/SriPrav/Documents/R/14Allstate")
root_directory = "C:/Users/SriPrav/Documents/R/14Allstate"

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

metric = "mae"

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

imp.features <- c(    "cat80_cat81_cat23"   , "cat80_cat81_cat28"   , "cat80_cat73_cat82"   , "cat80_cat90"         , "cat80_cat1_cat13"    , "cat80_cat81"        
                    , "cat80_cat1_cat4"     , "cat80_cat81_cat50"   , "cat80_cat81_cat3"    , "cat80_cat81_cat6"    , "cat80_cat73_cat40"   , "cat80_cat1_cat90"   
                    , "cat80_cat50_cat82"   , "cat80_cat72"         , "cat80_cat87"         , "cat80_cat79_cat7"    , "cat80_cat9_cat82"    , "cat80_cat57_cat82"  
                    , "cat80_cat90_cat14"   , "cat80_cat5_cat4"     , "cat80_cat28_cat24"   , "cat80_cat7_cat3"     , "cat80_cat57_cat90"   , "cat80_cat7_cat25"   
                    , "cat80_cat23_cat36"   , "cat80_cat1_cat6"     , "cat80_cat1_cat73"    , "cat80_cat1_cat103"   , "cat80_cat10_cat82"   , "cat80_cat1_cat9"    
                    , "cat80_cat1_cat23"    , "cat80_cat81_cat16"   , "cat80_cat82"         , "cat80_cat81_cat103"  , "cat80_cat81_cat11"   , "cat80_cat73_cat76"  
                    , "cat80_cat82_cat25"   , "cat80_cat73_cat28"   , "cat80_cat16_cat103"  , "cat80_cat1_cat14"    , "cat80_cat81_cat5"    , "cat80_cat81_cat14"  
                    , "cat80_cat73_cat4"    , "cat80_cat73_cat50"   , "cat80_cat81_cat90"   , "cat80_cat73_cat5"    , "cat80_cat73_cat25"   , "cat80_cat1_cat36"   
                    , "cat80_cat1_cat5"     , "cat80_cat1_cat25"    , "cat80_cat103_cat14"  , "cat80_cat1_cat28"    , "cat80_cat1_cat16"    , "cat80_cat103_cat24" 
                    , "cat80_cat50_cat4"    , "cat80_cat103_cat5"   , "cat80_cat90_cat25"   , "cat80_cat103_cat40"  , "cat80_cat28_cat82"   , "cat80_cat40_cat14"  
                    , "cat80_cat87_cat9"    , "cat80_cat16_cat6"    , "cat80_cat87_cat6"    , "cat80_cat11_cat76"   , "cat80_cat50_cat5"    , "cat80_cat79_cat82"  
                    , "cat80_cat40_cat4"    , "cat80_cat87_cat57"   , "cat80_cat89_cat81"   , "cat80_cat11_cat103"  , "cat80_cat76_cat82"   , "cat80_cat87_cat40"  
                    , "cat80_cat12_cat79"   , "cat80_cat9_cat73"    , "cat80_cat87_cat5"    , "cat80_cat2_cat73"    , "cat80_cat72_cat28"   , "cat80_cat72_cat1"   
                    , "cat80_cat28_cat6"    , "cat80_cat9_cat90"    , "cat80_cat36_cat76"   , "cat80_cat90_cat38"   , "cat80_cat12_cat9"    , "cat80_cat40_cat82"  
                    , "cat80_cat3_cat50"    , "cat80_cat14_cat82"   , "cat80_cat4_cat25"    , "cat80_cat9_cat40"    , "cat80_cat16_cat82"   , "cat80_cat10_cat72"  
                    , "cat80_cat89_cat111"  , "cat80_cat90_cat24"   , "cat80_cat9_cat25"    , "cat80_cat16_cat73"   , "cat80_cat40"         , "cat80_cat11_cat50"  
                    , "cat80_cat16_cat23"   , "cat80_cat2_cat36"    , "cat80_cat3_cat36"    , "cat80_cat36_cat14"   , "cat80_cat5"          , "cat80_cat89_cat50"  
                    , "cat80_cat11_cat5"    , "cat80_cat10_cat14"   , "cat80_cat72_cat5"    , "cat80_cat57_cat81"   , "cat80_cat40_cat25"   , "cat80_cat40_cat50"  
                    , "cat80_cat7_cat76"    , "cat80_cat11_cat24"   , "cat80_cat7_cat24"    , "cat80_cat90_cat36"   , "cat80_cat16_cat36"   , "cat80_cat7_cat14"   
                    , "cat80_cat28"         , "cat80_cat7_cat9"     , "cat80_cat7_cat103"   , "cat80_cat10_cat40"   , "cat12_cat79_cat7"    , "cat79_cat89_cat40"  
                    , "cat79_cat7"          , "cont_allEnergy"      , "cont_energy"         , "cont2"               , "cat12_cat103_cat82"  , "cat12_cat111_cat24" 
                    , "cont7"               , "cat53"               , "cat114"              , "cat57_cat12_cat38"   , "cat12_cat79_cat40"   , "cont11"             
                    , "cat12_cat103_cat14"  , "cont12"              , "cat79_cat89_cat28"   , "cat12_cat72_cat111"  , "cat12_cat111_cat76"  , "cat57_cat12_cat79"  
                    , "cat12_cat79_cat111"  , "cat10_cat103_cat82"  , "cat12_cat79_cat3"    , "cont3"               , "cat12_cat89_cat1"    , "cat12_cat89_cat24"  
                    , "cat12_cat11_cat103"  , "cat10_cat103_cat24"  , "cat10_cat103_cat38"  , "cat12_cat103_cat38"  , "cat106"              , "cat12_cat79_cat89"  
                    , "cat57_cat79_cat5"    , "cat80_cat12_cat103"  , "cat94"               , "cat12_cat79_cat23"   , "cat12_cat3_cat103"   , "cat57_cat12_cat81"  
                    , "cat10_cat103_cat40"  , "cat12_cat111_cat82"  , "cat87_cat12_cat111"  , "cat12_cat7_cat3"     , "cat57_cat12_cat11"   , "cat72_cat103_cat4"  
                    , "cat57_cat79_cat73"   , "cat12_cat79_cat50"   , "cat12_cat7_cat4"     , "cont6"               , "cat79_cat2_cat103"   , "cat57_cat12_cat40"  
                    , "cat105"              , "cat57_cat12_cat89"   , "cont14"              , "cat12_cat79_cat4"    , "cat57_cat12_cat111"  , "cat12_cat79_cat16"  
                    , "cat57_cat12_cat36"   , "cat12_cat38_cat25"   , "cat23_cat103_cat14"  , "cat12_cat13_cat111"  , "cat12_cat76_cat24"   , "cat12_cat89_cat81"  
                    , "cat12_cat103_cat6"   , "cat57_cat12_cat23"   , "cat12_cat3_cat16"    , "cat79_cat10_cat72"   , "cat12_cat79_cat11"   , "cat23_cat103_cat38" 
                    , "cat10_cat103_cat76"  , "cat12_cat79_cat28"   , "cat108"              , "cat12_cat89_cat103"  , "cat79_cat10_cat38"   , "cat80_cat73_cat38"  
                    , "cat57_cat79_cat10"   , "cat10_cat103"        , "cat116"              , "cat12_cat79_cat103"  , "cat12_cat111_cat6"   , "cat11_cat103_cat82" 
                    , "cat12_cat103_cat4"   , "cat104"              , "cat80_cat13_cat111"  , "cat12_cat103_cat25"  , "cat79_cat16_cat40"   , "cat79_cat2_cat82"   
                    , "cat12_cat103_cat76"  , "cat12_cat89_cat111"  , "cat57_cat12_cat72"   , "cat100"              , "cat79_cat10_cat2"    , "cat57_cat79_cat1"   
                    , "cat12_cat89_cat9"    , "cat57_cat12_cat50"   , "cat112"              , "cat11_cat111_cat4"   , "cont8"               , "cat10_cat103_cat111"
                    , "cont4"               , "cat109"              , "cat57_cat79_cat14"   , "cat12_cat103_cat24"  , "cat79_cat2_cat28"    , "cat57_cat79_cat111" 
                    , "cont_energyCont"     , "cat12_cat79_cat81"   , "cat79_cat10_cat13"   , "cat79_cat10_cat50"   , "cat12_cat89_cat14"   , "cat12_cat103"       
                    , "cat79_cat72_cat40"   , "cat12_cat7_cat89"    , "cat10_cat111_cat4"   , "cat79_cat13_cat23"   , "cat13_cat111_cat6"   , "cat12_cat40_cat111" 
                    , "cat57_cat12_cat90"   , "cat13_cat103_cat6"   , "cat11_cat103"        , "cat10_cat13_cat103"  , "cat12_cat10_cat103"  , "cat13_cat111_cat24" 
                    , "cat12_cat72_cat103"  , "cat12_cat16_cat103"  , "cat57_cat12_cat5"    , "cont1"               , "cat11_cat103_cat50"  , "cat23_cat103_cat111"
                    , "cat113"              , "cat103_cat111_cat82" , "cat110"              , "cat12_cat7_cat40"    , "cat115"              , "cat12_cat10_cat111" 
                    , "cat12_cat89_cat76"   , "cat10_cat111_cat6"   , "cat57_cat12_cat103"  , "cat57_cat12_cat13"   , "cont10"              , "cat79_cat13_cat103" 
                    , "cat12_cat38_cat82"   , "cat57_cat12_cat25"   , "cat12_cat79_cat90"   , "cat107"              , "cat103_cat111_cat24" , "cat57_cat111_cat5"  
                    , "cat12_cat38_cat24"   , "cat12_cat79_cat73"   , "cat57_cat79_cat90"   , "cat16_cat103_cat111" , "cat11_cat111_cat14"  , "cat10_cat103_cat28" 
                    , "cat87_cat103_cat24"  , "cat79_cat10_cat81"   , "cat11_cat111_cat82"  , "cat10_cat103_cat25"  , "cat75"               , "cat12_cat7_cat111"  
                    , "cat10_cat111_cat5"   , "cat10_cat11_cat103"  , "cat12_cat23_cat14"   , "cat12_cat89_cat38"   , "cat12_cat76_cat25"   , "cat57_cat79_cat38"  
                    , "cat79_cat82_cat25"   , "cat79_cat10_cat103"  , "cat79_cat89_cat11"   , "cat12_cat9_cat103"   , "cat11_cat103_cat25"  , "cat12_cat81_cat103" 
                    , "cont5"               , "cat79_cat11_cat13"   , "cat11_cat111_cat76"  , "cat12_cat79_cat6"    , "cat57_cat79_cat89"   , "cat23_cat103_cat28" 
                    , "cat79_cat2_cat25"    , "cat79_cat23_cat103"  , "cat10_cat3_cat103"   , "cat79_cat11_cat28"   , "cat79_cat5"          , "cat10_cat23_cat103" 
                    , "cat13_cat111_cat82"  , "cat13_cat111_cat50"  , "cat12_cat76_cat82"   , "cat12_cat76_cat4"    , "cat10_cat16_cat103"  , "cat12_cat79_cat2"   
                    , "cat12_cat111_cat50"  , "cat12_cat7_cat9"     , "cat81_cat1_cat6"     , "cat12_cat76_cat14"   , "cat87_cat12_cat76"   , "cat12_cat89_cat50"  
                    , "cat12_cat89_cat6"    , "cat12_cat9_cat38"    , "cat12_cat81_cat9"    , "cat103_cat111_cat38" , "cat12_cat79_cat25"   , "cat103_cat111_cat50"
                    , "cat87_cat12_cat81"   , "cat10_cat111_cat24"  , "cat81_cat1_cat111"   , "cat12_cat81_cat4"    , "cat87_cat12_cat90"   , "cat12_cat73_cat111" 
                    , "cat81_cat1_cat9"     , "cat12_cat16_cat82"   , "cat12_cat90_cat28"   , "cat10_cat36_cat111"  , "cat101"              , "cat99"              
                    , "cat111_cat5_cat38"   , "cat91"               , "cat12_cat7_cat5"     , "cat12_cat90_cat14"   , "cat87_cat12_cat5"    , "cat12_cat23_cat36"  
                    , "cat12_cat11_cat111"  , "cat81_cat1_cat25"    , "cat111_cat76_cat25"  , "cat10_cat103_cat14"  , "cat12_cat90_cat76"   , "cat12_cat2_cat72"   
                    , "cat80_cat12_cat89"   , "cat1_cat82"          , "cat12_cat72_cat14"   , "cat103_cat6_cat4"    , "cat87_cat103_cat4"   , "cat12_cat79_cat9"   
                    , "cat23_cat103_cat24"  , "cat12_cat10_cat50"   , "cat79_cat103_cat6"   , "cat12_cat16_cat38"   , "cat12_cat111_cat14"  , "cat12_cat3_cat50"   
                    , "cat103_cat111_cat14" , "cat11_cat103_cat38"  , "cat79_cat2_cat5"     , "cat103_cat111_cat76" , "cat12_cat24_cat25"   , "cat13_cat103_cat76" 
                    , "cat81_cat1_cat40"    , "cat1_cat40_cat38"    , "cat52"               , "cat49"               , "cat23_cat103_cat50"  , "cat57_cat79_cat28"  
                    , "cat12_cat28"         , "cat79_cat11_cat111"  , "cat81_cat73_cat50"   , "cat57_cat12_cat7"    , "cat79_cat72_cat11"   , "cat11_cat111_cat24" 
                    , "cat12_cat79_cat10"   , "cat23_cat103_cat4"   , "cat87_cat103_cat5"   , "cat57_cat79_cat4"    , "cat79_cat11_cat103"  , "cat79_cat103_cat111"
                    , "cat57_cat79_cat13"   , "cat79_cat10_cat36"   , "cat79_cat72_cat81"   , "cat103_cat14_cat82"  , "cat1_cat73_cat82"    , "cat80_cat87_cat12"  
                    , "cat57_cat79_cat40"   , "cat10_cat111_cat82"  , "cat111_cat6_cat50"   , "cat11_cat103_cat14"  , "cont13"              , "cat72_cat111_cat24" 
                    , "cat81_cat73_cat82"   , "cat1_cat73_cat40"    , "cat2_cat111_cat6"    , "cat79_cat10_cat23"   , "cat95"               , "cat79_cat3_cat28"   
                    , "cat12_cat103_cat5"   , "cat13_cat103_cat50"  , "cat89_cat103_cat24"  , "cat57_cat79_cat82"   , "cat79_cat7_cat111"   , "cat97"              
                    , "cat81_cat1_cat14"    , "cat79_cat7_cat72"    , "cat79_cat11_cat25"   , "cat79_cat10_cat111"  , "cat6_cat4"           , "cat79_cat10_cat3"   
                    , "cat79_cat7_cat11"    , "cat87_cat11_cat111"  , "cat57_cat111_cat50"  , "cat79_cat10_cat7"    , "cat103_cat50_cat5"   , "cat13_cat103_cat40" 
                    , "cat79_cat90_cat25"   , "cat13_cat103_cat38"  , "cat1_cat90_cat23"    , "cat2_cat50"          , "cat3_cat103_cat40"   , "cat79_cat10_cat6"   
                    , "cat79_cat13_cat111"  , "cat79_cat10_cat82"   , "cat79_cat89_cat4"    , "cat79_cat10_cat24"   , "cat103_cat5_cat24"   , "cat2_cat6"          
                    , "cat79_cat40_cat24"   , "cat79_cat89_cat38"   , "cat57_cat79_cat103"  , "cat79_cat89_cat81"   , "cat79_cat10_cat76"   , "cat16_cat103_cat82" 
                    , "cat79_cat13_cat90"   , "cat79_cat11_cat50"   , "cat79_cat89_cat23"   , "cat79_cat2_cat3"     , "cat79_cat81_cat11"   , "cat72_cat103"       
                    , "cat79_cat90_cat82"   , "cat79_cat7_cat9"     , "cat79_cat72_cat28"   , "cat79_cat7_cat38"    , "cat1_cat3_cat90"     , "cat81_cat1_cat76"   
                    , "cat103_cat28_cat25"  , "cat79_cat2_cat4"     , "cat79_cat1_cat111"   , "cat79_cat10_cat73"   , "cat79_cat3_cat36"    , "cat79_cat3_cat25"   
                    , "cat79_cat7_cat50"    , "cat79_cat81_cat6"    , "cat13_cat103_cat14"  , "cat79_cat89_cat16"   , "cat79_cat16_cat24"   , "cat81_cat1_cat73"   
                    , "cat79_cat7_cat76"    , "cat79_cat1_cat9"     , "cat92"               , "cat79_cat7_cat23"    , "cat79_cat3_cat111"   , "cat13_cat111"       
                    , "cat79_cat1_cat28"    , "cat1_cat36_cat6"     , "cat79_cat7_cat6"     , "cat79_cat13_cat24"   , "cat79_cat72_cat4"    , "cat13_cat111_cat38" 
                    , "cat86"               , "cat10_cat111_cat76"  , "cat79_cat7_cat2"     , "cat79_cat2_cat13"    , "cat111_cat76_cat4"   , "cat79_cat16_cat28"  
                    , "cat9_cat111_cat25"   , "cat79_cat40_cat14"   , "cat10_cat111_cat14"  , "cat79_cat13_cat28"   , "cat10_cat11_cat23"   , "cat9_cat6"          
                    , "cat111_cat4_cat82"   , "cat72_cat11_cat103"  , "cat87_cat11_cat103"  , "cat79_cat89_cat73"   , "cat87_cat79_cat14"   , "cat57_cat111_cat4"  
                    , "cat111_cat24_cat25"  , "cat1_cat90"          , "cat79_cat2_cat72"    , "cat111_cat76_cat24"  , "cat2_cat111_cat5"    , "cat81_cat103_cat14" 
                    , "cat90_cat111_cat6"   , "cat81_cat40_cat28"   , "cat87_cat103_cat38"  , "cat111_cat76"        , "cat81_cat13_cat73"   , "cat57_cat12_cat73"  
                    , "cat81_cat1_cat3"     , "cat1_cat76_cat4"     , "cat81_cat76_cat4"    , "cat57_cat12_cat6"    , "cat81_cat38_cat24"   , "cat6_cat82_cat25"   
                    , "cat98"               , "cat57_cat12_cat76"   , "cat23_cat103_cat5"   , "cat7_cat111_cat14"   , "cat13_cat103_cat4"   , "cat7_cat111_cat5"   
                    , "cat57_cat1_cat90"    , "cat3_cat103_cat5"    , "cat87_cat111_cat4"   , "cat90_cat103_cat40"  , "cat87_cat111"        , "cat11_cat111_cat25" 
                    , "cat81_cat36_cat111"  , "cat87_cat10_cat40"   , "cat72_cat111_cat14"  , "cat90_cat103_cat38"  , "cat87_cat13_cat103"  , "cat3_cat111_cat14"  
                    , "cat79_cat103_cat4"   , "cat7_cat103_cat50"   , "cat11_cat111"        , "cat81_cat13_cat82"   , "cat1_cat73_cat28"    , "cat90_cat103_cat14" 
                    , "cat72_cat23_cat111"  , "cat81_cat36_cat4"    , "cat89_cat103_cat14"  , "cat72_cat9_cat103"   , "cat79_cat111_cat50"  , "cat10_cat111_cat25" 
                    , "cat16_cat103_cat50"  , "cat79_cat16_cat103"  , "cat103_cat5_cat14"   , "cat81_cat1_cat24"    , "cat3_cat111"         , "cat6_cat5"          
                    , "cat1_cat36_cat5"     , "cat111"              , "cat81_cat1_cat13"    , "cat81_cat1_cat90"    , "cat89_cat103_cat82"  , "cat3_cat111_cat24"  
                    , "cat1_cat82_cat25"    , "cat12_cat111_cat25"  , "cat1_cat76_cat50"    , "cat1_cat4_cat14"     , "cat1_cat103_cat38"   , "cat12_cat7_cat38"   
                    , "cat80_cat103_cat111" , "cat6_cat82"          , "cat12_cat89_cat28"   , "cat23_cat111_cat82"  , "cat12_cat103_cat50"  , "cat103_cat4_cat24"  
                    , "cat90_cat103_cat6"   , "cat12_cat79_cat14"   , "cat6_cat76"          , "cat72_cat103_cat5"   , "cat57_cat23_cat82"   , "cat81_cat82"        
                    , "cat2_cat111_cat82"   , "cat2_cat72_cat90"    , "cat81_cat6_cat82"    , "cat81_cat3_cat28"    , "cat2_cat72_cat73"    , "cat57_cat103"       
                    , "cat1_cat6_cat25"     , "cat36_cat111_cat14"  , "cat12_cat23_cat103"  , "cat2_cat111_cat4"    , "cat13_cat103_cat5"   , "cat6_cat76_cat24"   
                    , "cat81_cat73_cat103"  , "cat90_cat103_cat76"  , "cat81_cat4_cat38"    , "cat23_cat103_cat25"  , "cat103_cat40_cat14"  , "cat50_cat5"         
                    , "cat12_cat7_cat28"    , "cat83"               , "cat103_cat4_cat14"   , "cat12_cat111_cat5"   , "cat80_cat111_cat25"  , "cat1_cat36_cat50"   
                    , "cat10_cat72_cat38"   , "cat13_cat111_cat76"  , "cat12_cat9_cat111"   , "cat81_cat6_cat4"     , "cat57_cat11_cat111"  , "cat1_cat73_cat38"   
                    , "cat6_cat76_cat14"    , "cat16_cat111_cat50"  , "cat87_cat103_cat111" , "cat50_cat4"          , "cat1_cat4"           , "cat16_cat111_cat5"  
                    , "cat103_cat76_cat50"  , "cat89_cat81_cat50"   , "cat1_cat9_cat50"     , "cat11_cat103_cat6"   , "cat2_cat72_cat16"    , "cat87_cat10_cat103" 
                    , "cat57_cat111_cat38"  , "cat50_cat82"         , "cat12_cat1_cat36"    , "cat1_cat111_cat82"   , "cat1_cat111_cat4"    , "cat10_cat103_cat4"  
                    , "cat72_cat111"        , "cat1_cat76_cat24"    , "cat73_cat82_cat25"   , "cat89_cat103_cat111" , "cat87_cat111_cat50"  , "cat50_cat82_cat25"  
                    , "cat12_cat72_cat25"   , "cat57_cat13_cat111"  , "cat1_cat28_cat76"    , "cat9_cat111_cat50"   , "cat57_cat103_cat82"  , "cat6_cat50"         
                    , "cat12_cat14_cat82"   , "cat72_cat111_cat38"  , "cat6_cat76_cat50"    , "cat1_cat3_cat36"     , "cat57_cat111_cat76"  , "cat9_cat111_cat76"  
                    , "cat13_cat111_cat14"  , "cat96"               , "cat3_cat111_cat6"    , "cat6_cat25"          , "cat11_cat111_cat6"   , "cat12_cat7_cat81"   
                    , "cat81_cat11_cat36"   , "cat2_cat72_cat9"     , "cat72_cat36_cat111"  , "cat79_cat24_cat82"   , "cat10_cat103_cat50"  , "cat57_cat72_cat11"  
                    , "cat1_cat73_cat25"    , "cat72_cat111_cat4"   , "cat87_cat111_cat82"  , "cat10_cat103_cat5"   , "cat79_cat111_cat82"  , "cat57_cat12_cat14"  
                    , "cat12_cat2_cat11"    , "cat6_cat38"          , "cat12_cat82_cat25"   , "cat90_cat111_cat24"  , "cat57_cat40"         , "cat72_cat103_cat6"  
                    , "cat90_cat111_cat38"  , "cat1_cat9_cat25"     , "cat10_cat13_cat111"  , "cat111_cat50_cat25"  , "cat111_cat14_cat82"  , "cat6_cat76_cat5"    
                    , "cat80_cat81_cat1"    , "cat6_cat14_cat25"    , "cat1_cat76_cat38"    , "cat12_cat89_cat90"   , "cat1_cat3_cat40"     , "cat13_cat3_cat111"  
                    , "cat57_cat79_cat25"   , "cat103_cat6_cat14"   , "cat111_cat4_cat14"   , "cat111_cat4_cat38"   , "cat6_cat76_cat25"    , "cat89_cat103_cat25" 
                    , "cat1_cat9_cat14"     , "cat12_cat4_cat14"    , "cat90_cat103_cat24"  , "cat111_cat50_cat38"  , "cat2_cat72_cat40"    , "cat80_cat103_cat76" 
                    , "cat81_cat73"         , "cat6_cat5_cat4"      , "cat71"               , "cat81_cat73_cat38"   , "cat1_cat16_cat103"   , "cat13_cat103"       
                    , "cat4_cat82_cat25"    , "cat82"               , "cat10_cat89_cat111"  , "cat13_cat111_cat5"   , "cat6_cat50_cat25"    , "cat1_cat103_cat111" 
                    , "cat57_cat103_cat38"  , "cat23_cat103_cat82"  , "cat3_cat103_cat28"   , "cat12_cat72_cat16"   , "cat81_cat90_cat23"   , "cat73_cat82"        
                    , "cat2_cat111_cat14"   , "cat79_cat103_cat40"  , "cat4_cat82"          , "cat1_cat24_cat82"    , "cat2_cat72_cat5"     , "cat81_cat111_cat24" 
                    , "cat81_cat1_cat82"    , "cat80_cat111_cat38"  , "cat12_cat81_cat73"   , "cat6_cat4_cat24"     , "cat57_cat89_cat50"   , "cat6_cat76_cat82"   
                    , "cat3_cat111_cat4"    , "cat6_cat38_cat82"    , "cat103_cat4_cat38"   , "cat6_cat4_cat14"     , "cat72_cat9_cat14"    , "cat10_cat72_cat76"  
                    , "cat12_cat72_cat13"   , "cat36_cat111_cat5"   , "cat50_cat38_cat24"   , "cat1_cat76"          , "cat80_cat72_cat111"  , "cat57_cat103_cat76" 
                    , "cat6_cat50_cat24"    , "cat81_cat90_cat36"   , "cat57_cat82_cat25"   , "cat81_cat1_cat16"    , "cat50_cat38_cat25"   , "cat80_cat11_cat111" 
                    , "cat9_cat50"          , "cat6_cat5_cat25"     , "cat39"               , "cat93"               , "cat72_cat11_cat36"   , "cat1_cat16_cat6"    
                    , "cat7_cat111_cat6"    , "cat1_cat36"          , "cat1_cat73_cat103"   , "cat6_cat24_cat25"    , "cat57_cat111_cat82"  , "cat2_cat82_cat25"   
                    , "cat90_cat103_cat5"   , "cat12_cat72_cat23"   , "cat6_cat4_cat25"     , "cat9_cat111_cat38"   , "cat16_cat111_cat82"  , "cat10_cat72_cat25"  
                    , "cat50_cat5_cat82"    , "cat82_cat25"         , "cat12_cat1_cat103"   , "cat9_cat111_cat6"    , "cat13_cat111_cat25"  , "cat103_cat4_cat25"  
                    , "cat79_cat111_cat5"   , "cat12_cat79_cat38"   , "cat81_cat1_cat103"   , "cat1_cat73_cat6"     , "cat6_cat50_cat5"     , "cat111_cat5"        
                    , "cat7_cat103_cat28"   , "cat6_cat5_cat24"     , "cat12_cat79_cat36"   , "cat6_cat76_cat38"    , "cat87_cat111_cat76"  , "cat57_cat10_cat82"  
                    , "cat6_cat4_cat38"     , "cat90_cat111_cat82"  , "cat1_cat111_cat38"   , "cat57_cat111"        , "cat111_cat76_cat5"   , "cat90_cat103_cat50" 
                    , "cat79_cat111_cat6"   , "cat1_cat3_cat23"     , "cat1_cat4_cat25"     , "cat57_cat103_cat5"   , "cat72_cat13_cat111"  , "cat1_cat90_cat5"    
                    , "cat5_cat82"          , "cat9_cat103_cat4"    , "cat6_cat76_cat4"     , "cat72_cat16_cat111"  , "cat1_cat9_cat24"     , "cat57_cat79"        
                    , "cat57_cat9_cat23"    , "cat1_cat16_cat28"    , "cat2_cat111_cat76"   , "cat57_cat13_cat90"   , "cat81_cat103_cat38"  , "cat81_cat3_cat36"   
                    , "cat12_cat16_cat28"   , "cat57_cat111_cat24"  , "cat81_cat13_cat25"   , "cat81_cat90_cat82"   , "cat12_cat11_cat14"   , "cat9_cat103_cat40"  
                    , "cat111_cat24_cat82"  , "cat50_cat4_cat14"    , "cat103_cat82"        , "cat84"               , "cat81_cat13_cat40"   , "cat40_cat111_cat25" 
                    , "cat7_cat111"         , "cat1_cat40_cat50"    , "cat1_cat24"          , "cat2_cat103_cat4"    , "cat9_cat111_cat4"    , "cat72_cat9_cat111"  
                    , "cat2_cat111_cat38"   , "cat72_cat9_cat28"    , "cat72_cat81_cat73"   , "cat31"               , "cat12_cat111_cat4"   , "cat50_cat5_cat25"   
                    , "cat6_cat14_cat38"    , "cat12_cat81_cat25"   , "cat50_cat4_cat24"    , "cat36_cat111_cat76"  , "cat72_cat11_cat5"    , "cat36_cat111_cat4"  
                    , "cat81_cat111_cat4"   , "cat50_cat4_cat82"    , "cat57_cat7_cat81"    , "cat80_cat81_cat73"   , "cat12_cat79"         , "cat13_cat111_cat4"  
                    , "cat6_cat38_cat25"    , "cat72_cat111_cat25"  , "cat1_cat111_cat14"   , "cat6_cat24"          , "cat50_cat24_cat25"   , "cat12_cat1_cat25"   
                    , "cat103_cat76_cat14"  , "cat6_cat14"          , "cat12_cat72_cat4"    , "cat81_cat73_cat14"   , "cat103_cat4"         , "cat57_cat10_cat90"  
                    , "cat12_cat73_cat103"  , "cat1_cat23_cat111"   , "cat12_cat1_cat4"     , "cat111_cat6_cat38"   , "cat1_cat28"          , "cat36_cat111_cat50" 
                    , "cat1_cat50_cat14"    , "cat50_cat38"         , "cat2_cat103_cat111"  , "cat111_cat50_cat5"   , "cat9_cat111_cat82"   , "cat57_cat36_cat111" 
                    , "cat72_cat90_cat82"   , "cat87_cat111_cat25"  , "cat81_cat90_cat40"   , "cat88"               , "cat9_cat82"          , "cat73_cat6_cat76"   
                    , "cat1_cat5"           , "cat57_cat79_cat72"   , "cat1_cat9_cat36"     , "cat1_cat3_cat4"      , "cat72_cat103_cat28"  , "cat50_cat14_cat82"  
                    , "cat2_cat72_cat38"    , "cat57_cat111_cat14"  , "cat1_cat3_cat25"     , "cat11_cat28_cat111"  , "cat57_cat3_cat111"   , "cat36_cat111"       
                    , "cat81_cat103_cat111" , "cat36_cat82_cat25"   , "cont9"               , "cat111_cat5_cat14"   , "cat72_cat82_cat25"   , "cat1_cat28_cat50"   
                    , "cat72_cat9_cat73"    , "cat111_cat14_cat24"  , "cat57_cat81_cat111"  , "cat6"                , "cat79_cat72_cat6"    , "cat81_cat24_cat82"  
                    , "cat111_cat50_cat4"   , "cat50_cat4_cat38"    , "cat12_cat5_cat24"    , "cat6_cat14_cat24"    , "cat6_cat50_cat38"    , "cat50_cat5_cat38"   
                    , "cat5_cat82_cat25"    , "cat50_cat5_cat14"    , "cat72_cat5_cat25"    , "cat11_cat111_cat5"   , "cat72_cat23_cat103"  , "cat73_cat111_cat14" 
                    , "cat10_cat111"        , "cat6_cat50_cat82"    , "cat81_cat73_cat6"    , "cat12_cat103_cat40"  , "cat57_cat89_cat76"   , "cat6_cat5_cat82"    
                    , "cat2_cat111_cat24"   , "cat2_cat72_cat13"    , "cat12_cat72_cat1"    , "cat36_cat111_cat24"  , "cat3_cat111_cat82"   , "cat57_cat2_cat72"   
                    , "cat103_cat111_cat5"  , "cat90_cat111_cat50"  , "cat103_cat6_cat5"    , "cat2_cat103_cat76"   , "cat1_cat5_cat82"     , "cat102"             
                    , "cat6_cat4_cat82"     , "cat81_cat103_cat76"  , "cat12_cat1_cat24"    , "cat57_cat40_cat76"   , "cat80_cat6_cat25"    , "cat103_cat50_cat24" 
                    , "cat87_cat111_cat24"  , "cat6_cat24_cat82"    , "cat6_cat5_cat38"     , "cat2_cat72_cat14"    , "cat73_cat111_cat38"  , "cat6_cat38_cat24"   
                    , "cat111_cat6_cat76"   , "cat12_cat72_cat50"   , "cat80_cat111_cat50"  , "cat9_cat111_cat5"    , "cat1_cat111_cat76"   , "cat87_cat12_cat10"  
                    , "cat50_cat25"         , "cat81_cat36_cat38"   , "cat1_cat25"          , "cat16_cat111_cat25"  , "cat1_cat90_cat6"     , "cat1_cat111_cat25"  
                    , "cat9_cat111_cat14"   , "cat2_cat111_cat25"   , "cat111_cat38_cat25"  , "cat57_cat12_cat1"    , "cat12_cat72_cat73"   , "cat81_cat9_cat50"   
                    , "cat12_cat111_cat38"  , "cat6_cat50_cat14"    , "cat81_cat90_cat4"    , "cat81_cat36"         , "cat1_cat23_cat5"     , "cat12_cat72_cat40"  
                    , "cat72_cat103_cat40"  , "cat9_cat82_cat25"    , "cat81_cat5_cat14"    , "cat76_cat82_cat25"   , "cat12_cat72_cat90"   , "cat36_cat111_cat25" 
                    , "cat10_cat111_cat38"  , "cat11_cat111_cat38"  , "cat38_cat82_cat25"   , "cat81_cat16_cat6"    , "cat9_cat103_cat76"   , "cat1_cat36_cat25"   
                    , "cat72_cat103_cat76"  , "cat73_cat111_cat50"  , "cat81_cat73_cat25"   , "cat12_cat90_cat24"   , "cat111_cat5_cat25"   , "cat81_cat14"        
                    , "cat36_cat111_cat6"   , "cat12_cat10_cat73"   , "cat1_cat23_cat38"    , "cat73_cat103_cat4"   , "cat57_cat2_cat111"   , "cat73_cat111_cat6"  
                    , "cat57_cat9_cat111"   , "cat50_cat14_cat38"   , "cat72_cat103_cat38"  , "cat1_cat3_cat38"     , "cat103_cat38_cat82"  , "cat6_cat5_cat14"    
                    , "cat111_cat38_cat82"  , "cat72_cat28_cat111"  , "cat57_cat89_cat13"   , "cat11_cat103_cat40"  , "cat111_cat6"         , "cat36_cat111_cat82" 
                    , "cat6_cat50_cat4"     , "cat50_cat5_cat24"    , "cat12_cat11_cat1"    , "cat72_cat81"         , "cat12_cat7_cat82"    , "cat81_cat50"        
                    , "cat72_cat5_cat82"    , "cat57_cat89_cat5"    , "cat12_cat81_cat13"   , "cat1_cat111_cat24"   , "cat72_cat82"         , "cat12_cat13_cat16"  
                    , "cat1_cat13_cat76"    , "cat1_cat36_cat28"    , "cat38_cat82"         , "cat36_cat82"         , "cat24_cat82"         , "cat103_cat38_cat25" 
                    , "cat81_cat90_cat103"  , "cat103_cat76"        , "cat111_cat6_cat24"   , "cat81_cat36_cat24")


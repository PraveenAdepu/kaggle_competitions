# rm(list=ls())
################################################################################################

# 20160921 - Baseline01 -- 2 features -- R Image

# Sys.time()
# load("Allstate_Baseline01_20161014.RData"); gc()
# Sys.time()
################################################################################################


#names(testingSet)
feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("id","loss", "CVindices" ))])
testfeature.names <- names(testingSet[,-which(names(testingSet) %in% c("id" ))])

# ratio.features <- grep("Ratio", names(trainingSet), value = T)
# 
# 
# # ratio.features <-  c("cat100","cat101","cat2","cat53","cat114","cat10","cat57","cat72","cat87")
# # 
# feature.names         <- setdiff(feature.names, ratio.features)
# testfeature.names     <- setdiff(testfeature.names, ratio.features)

##################################################################################

##################################################################################

# order of columns are matching 
##################################################################################

# Check for all columns order, all columns orders are same after removing id columns


cv          = 5
bags        = 5
nround.cv   = 2100 
printeveryn = 200
seed        = 2016

## for all remaining models, use same parameters 

param <- list(  "objective"        = "reg:linear", 
                "booster"          = "gbtree",
                #"eval_metric"      = "auc",
                "nthread"          = 7,     
                "max_depth"        = 7,     
                "eta"              = 0.02, 
                "subsample"        = 0.95,  
                "colsample_bytree" = 0.3,  
                "min_child_weight" = 1     
                
)


cat(cv, "-fold Cross Validation\n", sep = "")

set.seed(seed)
for (i in 1:cv)
  
{
  cat(i ,"fold Processing\n")
  
  cat("X_build fold Processing\n")
  X_build <- subset(trainingSet, CVindices != i, select = -c( CVindices))
  cat("X_val fold Processing\n")
  X_val   <- subset(trainingSet, CVindices == i, select = -c( CVindices)) 
  
  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=log(X_build$loss))
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=log(X_val$loss))
  watchlist <- list( val = dval,train = dtrain)
  
  
  pred_cv_bags   <- rep(0, nrow(X_val[, feature.names]))
  pred_test_bags <- rep(0, nrow(testingSet[,testfeature.names]))
  
  for (b in 1:bags) 
    {
        cat(b ," - bag Processing\n")
        seed = b + seed
        set.seed(seed)
          cat("X_build training Processing\n")
          XGModel <- xgb.train(   params              = param,
                                  feval               = xgb.metric.log.mae, #xgb.metric.mae
                                  data                = dtrain,
                                  watchlist           = watchlist,
                                  nrounds             = nround.cv ,
                                  print.every.n       = printeveryn,
                                  verbose             = TRUE, 
                                  #maximize            = TRUE,
                                  set.seed            = seed
          )
        cat("X_val prediction Processing\n")
        pred_cv    <- predict(XGModel, data.matrix(X_val[,feature.names]))
        cat("CV TestingSet prediction Processing\n")
        pred_test  <- predict(XGModel, data.matrix(testingSet[,testfeature.names]))
        
        pred_cv_bags   <- pred_cv_bags + exp(pred_cv)
        pred_test_bags <- pred_test_bags + exp(pred_test)
  }
  pred_cv_bags   <- pred_cv_bags / bags
  pred_test_bags <- pred_test_bags / bags
  
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$loss, pred_cv_bags, metric), "\n", sep = "")
    
    val_predictions <- data.frame(id=X_val$id, loss = pred_cv_bags)
    test_predictions <- data.frame(id=testingSet$id, loss = pred_test_bags)
    
    if(i == 1)
    {
      write.csv(val_predictions,  'prav.xgb04.fold1.csv', row.names=FALSE, quote = FALSE)
      write.csv(test_predictions, 'prav.xgb04.fold1-test.csv', row.names=FALSE, quote = FALSE)
    }
    if(i == 2)
    {
      write.csv(val_predictions,  'prav.xgb04.fold2.csv', row.names=FALSE, quote = FALSE)
      write.csv(test_predictions, 'prav.xgb04.fold2-test.csv', row.names=FALSE, quote = FALSE)
    }
    if(i == 3)
    {
      write.csv(val_predictions,  'prav.xgb04.fold3.csv', row.names=FALSE, quote = FALSE)
      write.csv(test_predictions, 'prav.xgb04.fold3-test.csv', row.names=FALSE, quote = FALSE)
    }
    if(i == 4)
    {
      write.csv(val_predictions,  'prav.xgb04.fold4.csv', row.names=FALSE, quote = FALSE)
      write.csv(test_predictions, 'prav.xgb04.fold4-test.csv', row.names=FALSE, quote = FALSE)
    }
    if(i == 5)
    {
      write.csv(val_predictions,  'prav.xgb04.fold5.csv', row.names=FALSE, quote = FALSE)
      write.csv(test_predictions, 'prav.xgb04.fold5-test.csv', row.names=FALSE, quote = FALSE)
    }
  
}

# Full training

dtrain<-xgb.DMatrix(data=data.matrix(trainingSet[, feature.names]),label=log(trainingSet$loss))
watchlist <- list( train = dtrain)

fulltrainnrounds = 1.1 * nround.cv


# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

fulltest_ensemble <- rep(0, nrow(testingSet[,testfeature.names]))


for (b in 1:bags) {
  cat(b ," - bag Processing\n")
  seed = b + seed
  set.seed(seed)
  cat("Bagging Full TrainingSet training\n")
  XGModelFulltrain <- xgb.train(    params              = param,
                                    feval               = xgb.metric.log.mae,
                                    data                = dtrain,
                                    watchlist           = watchlist,
                                    nrounds             = fulltrainnrounds,
                                    print.every.n       = printeveryn,
                                    verbose             = TRUE, 
                                    #maximize            = TRUE,
                                    set.seed            = seed
  )
  cat("Bagging Full Model prediction Processing\n")
  
  predfull_test         <- predict(XGModelFulltrain, data.matrix(testingSet[,testfeature.names]))
  
  fulltest_ensemble <- fulltest_ensemble + exp(predfull_test)
  
}

fulltest_ensemble <- fulltest_ensemble / bags
testfull_predictions  <- data.frame(id=testingSet$id, loss = fulltest_ensemble)
write.csv(testfull_predictions, './submissions/prav.xgb04.full.csv', row.names=FALSE, quote = FALSE)




# #########################################################################################################
# Bagging - Full train
# #########################################################################################################



# head(testfull_predictions)

############################################################################################
model = xgb.dump(XGModel, with.stats=TRUE)

names = dimnames(trainingSet[,feature.names])[[2]]
importance_matrix = xgb.importance( names , model = XGModel)
#xgb.plot.importance(importance_matrix)
impMatrix <- as.data.frame(importance_matrix)
# impMatrix[1:12,1]
# impMatrix[,1]
ImpFeature <- impMatrix[,1]


# #########################################################################################################



trainingSet <- train
testingSet  <- test 

trainingFolds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")

trainingSet <- left_join(trainingSet, trainingFolds, by="user_id")


feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("user_id","order_id","product_id","eval_set", "reordered", "CVindices" ))])

# Model -------------------------------------------------------------------


cv          = 5
bags        = 1
nround.cv   = 1500 
printeveryn = 100
seed        = 2016

param <- list(
  "objective"           = "reg:logistic",
  "booster"             = "gbtree",
  "eval_metric"         = "logloss",
  "tree_method"         = "exact",
  "nthread"             = 28,  
  "max_depth"           = 8,
  "eta"                 = 0.01,
  "min_child_weight"    = 10,
  "gamma"               = 0.7,
  "subsample"           = 0.7,
  "colsample_bytree"    = 0.95,
  "alpha"               = 2e-05,
  "lambda"              = 10
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
  
  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=X_build$reordered)
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=X_val$reordered)
  watchlist <- list( val = dval,train = dtrain)
  
  cat("X_build training Processing\n")
  XGModel <- xgb.train(   params              = param,
                          data                = dtrain,
                          watchlist           = watchlist,
                          nrounds             = nround.cv ,
                          print_every_n       = printeveryn,
                          verbose             = TRUE, 
                          #maximize           = TRUE,
                          set.seed            = seed
  )
  
  cat("X_val prediction Processing\n")
  pred_cv  <- predict(XGModel, data.matrix(X_val[,feature.names]))
  val_predictions <- data.frame(user_id=X_val$user_id,order_id=X_val$order_id,product_id=X_val$product_id, pred = pred_cv)
  
  dt <- data.frame(user_id=X_val$user_id, purch=X_val$reordered, pred=pred_cv)
  f1score <- dt %>%
    group_by(user_id) %>%
    summarise(f1score=f1Score(purch, pred, cutoff=0.22))
  
  cat("fold " , i  , " F1 score - including NA : " , mean(f1score$f1score, na.rm = TRUE), "\n", sep = "")
  f1score[is.na(f1score)] <- 0
  cat("fold " , i  , " F1 score - NA replace with 0 : " , mean(f1score$f1score), "\n", sep = "")
  
  
  cat("CV TestingSet prediction Processing\n")
  pred_test  <- predict(XGModel, data.matrix(testingSet[,feature.names]))
  test_predictions <- data.frame(user_id=testingSet$user_id,order_id=testingSet$order_id,product_id=testingSet$product_id, pred = pred_test)
  
  
  if(i == 1)
  {
    write.csv(val_predictions,  './submissions/prav.xgb03.fold1.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb03.fold1-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 2)
  {
    write.csv(val_predictions,  './submissions/prav.xgb03.fold2.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb03.fold2-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 3)
  {
    write.csv(val_predictions,  './submissions/prav.xgb03.fold3.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb03.fold3-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 4)
  {
    write.csv(val_predictions,  './submissions/prav.xgb03.fold4.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb03.fold4-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 5)
  {
    write.csv(val_predictions,  './submissions/prav.xgb03.fold5.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb03.fold5-test.csv', row.names=FALSE, quote = FALSE)
  }
  
}



dtrain<-xgb.DMatrix(data=data.matrix(trainingSet[, feature.names]),label=trainingSet$reordered)
watchlist <- list( train = dtrain)

fulltrainnrounds = as.integer(1.2 * nround.cv)


# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

fulltest_ensemble <- rep(0, nrow(testingSet[,feature.names]))

for (b in 1:bags) {
  # seed = seed + b
  # set.seed(seed)
  cat(b ," - bag Processing\n")
  cat("Bagging Full TrainingSet training\n")
  XGModelFulltrain <- xgb.train(    params              = param,
                                    data                = dtrain,
                                    watchlist           = watchlist,
                                    nrounds             = fulltrainnrounds,
                                    print_every_n       = printeveryn,
                                    verbose             = TRUE, 
                                    #maximize            = TRUE,
                                    set.seed            = seed
  )
  cat("Bagging Full Model prediction Processing\n")
  
  predfull_test         <- predict(XGModelFulltrain, data.matrix(testingSet[,feature.names]))
  fulltest_ensemble     <- fulltest_ensemble + predfull_test
}

fulltest_ensemble <- fulltest_ensemble / bags
testfull_predictions <- data.frame(user_id=testingSet$user_id,order_id=testingSet$order_id,product_id=testingSet$product_id, pred = fulltest_ensemble)

write.csv(testfull_predictions, './submissions/prav.xgb03.full.csv', row.names=FALSE, quote = FALSE)

# #########################################################################################################
# Bagging - Full train
# #########################################################################################################


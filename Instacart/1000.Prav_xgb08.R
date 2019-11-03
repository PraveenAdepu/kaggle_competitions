
train <- readRDS( file="./input/trainingSet.rds")
test  <- readRDS( file="./input/testingSet.rds" )

train$v1 <- NULL
train$v2 <- NULL
train$order_streak <- NULL
train$CVindices <- NULL

test$v1 <- NULL
test$v2 <- NULL
test$order_streak <- NULL
test$CVindices <- NULL

product_embeds <- read.csv("./input/product_vector_features2.csv")

user_product_streak <- read.csv("./input/order_streaks.csv")

trainingSet <- left_join(train, user_product_streak, by=c("user_id","product_id"))
testingSet  <- left_join(test, user_product_streak, by=c("user_id","product_id"))

trainingSet <- left_join(trainingSet, product_embeds, by="product_id")
testingSet  <- left_join(testingSet, product_embeds, by="product_id")


trainingFolds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")

trainingSet <- left_join(trainingSet, trainingFolds, by="user_id")



feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("user_id","order_id","product_id","eval_set", "reordered", "CVindices",
                                                                         "user_product_max_prior_days_max", 
                                                                         "user_department_min_dow",  
                                                                         "user_product_min_add_to_cart_order_min",  
                                                                         "user_min_hour_since_prior",  
                                                                         "user_median_hour_since_prior",  
                                                                         "user_product_maxRank_hour",  
                                                                         "user_product_mean_hour",  
                                                                         "user_department_mean_hour",  
                                                                         "user_aisle_mean_hour",  
                                                                         "user_aisle_maxRank_hour",  
                                                                         "user_mean_hour_since_prior",  
                                                                         "user_department_maxRank_hour",  
                                                                         "user_max_hour_since_prior"
                                                                         
                                                                         # "user_product_mean_hour_mean", 
                                                                         # "user_aisle_minRank_hour",
                                                                         # "user_department_max_dow", 
                                                                         # "user_department_maxRank_dow", 
                                                                         # "prod_max_dow", 
                                                                         # "v2", 
                                                                         # "user_department_max_hour", 
                                                                         # "user_product_mean_dow", 
                                                                         # "user_department_minRank_hour", 
                                                                         # "user_aisle_minRank_dow", 
                                                                         # "user_max_days_since_prior", 
                                                                         # "user_department_max_add_to_cart_order", 
                                                                         # "user_department_minRank_dow", 
                                                                         # "user_product_min_prior_days", 
                                                                         # "aisle_id", 
                                                                         # "user_product_minRank_hour", 
                                                                         # "user_department_min_hour" 
                                                                         ))])

 


head(trainingSet)
head(testingSet)

trainingSet[is.na(trainingSet)] <- 0
testingSet[is.na(testingSet)]   <- 0

# write.csv(trainingSet,  './input/trainingSet_01.csv', row.names=FALSE, quote = FALSE) # 8474661
# write.csv(testingSet,  './input/testingSet_01.csv', row.names=FALSE, quote = FALSE)   # 4833292

# feature.scores <- features.target.AUC.score(trainingSet, feature.names, target = "reordered")
# 
# feature.scores

# user_product_min_add_to_cart_order_min 0.5000000
# Model -------------------------------------------------------------------

cv          = 5
bags        = 1
nround.cv   = 1500 
printeveryn = 100
seed        = 2017

param <- list(
  "objective"           = "reg:logistic",
  "booster"             = "gbtree",
  "eval_metric"         = "logloss",
  "tree_method"         = "exact",
  "nthread"             = 28,  
  "max_depth"           = 6,
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
    write.csv(val_predictions,  './submissions/prav.xgb08.fold1.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb08.fold1-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 2)
  {
    write.csv(val_predictions,  './submissions/prav.xgb08.fold2.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb08.fold2-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 3)
  {
    write.csv(val_predictions,  './submissions/prav.xgb08.fold3.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb08.fold3-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 4)
  {
    write.csv(val_predictions,  './submissions/prav.xgb08.fold4.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb08.fold4-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 5)
  {
    write.csv(val_predictions,  './submissions/prav.xgb08.fold5.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb08.fold5-test.csv', row.names=FALSE, quote = FALSE)
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

write.csv(testfull_predictions, './submissions/prav.xgb08.full.csv', row.names=FALSE, quote = FALSE)

# #########################################################################################################
# Bagging - Full train
# #########################################################################################################
############################################################################################
model = xgb.dump(XGModelFulltrain, with.stats=TRUE)

names = dimnames(trainingSet[,feature.names])[[2]]
importance_matrix = xgb.importance( names , model = XGModelFulltrain)
#xgb.plot.importance(importance_matrix)
impMatrix <- as.data.frame(importance_matrix)
# impMatrix[1:12,1]
# impMatrix[,1]
ImpFeature <- impMatrix[,1]


# #########################################################################################################


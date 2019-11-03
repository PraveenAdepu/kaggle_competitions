# rm(list=ls())
################################################################################################

# 20160921 - Baseline01 -- 2 features -- R Image

# Sys.time()
# load("Allstate_Baseline01_20161014.RData"); gc()
# Sys.time()
################################################################################################
# IMO : Used from Source files

train <- read_csv('./input/train_preprocess.csv')
test  <- read_csv('./input/test_preprocess.csv')

CVindices5folds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")
names(train[,1])
names(test)
names(CVindices5folds)

head(train[,1:2])
head(test[,1:2])

train[,1] <- NULL
test[,1]  <- NULL

names(train)[names(train)=="id_1"] <- "id"
names(test)[names(test)=="id_1"] <- "id"

head(CVindices5folds)


test$loss <- NULL

trainingSet <- left_join(train, CVindices5folds, by = "id")
testingSet  <- test

rm(train,test ,CVindices5folds); gc()

# hist(trainingSet$loss, col = "tomato")
# 
# skewness(trainingSet$loss)
# 
# skew.score <- function(c, x) (skewness(log(x + c)))^2
# 
# cval <- seq(0, 500, l = 101)
# skew <- cval * 0
# for (i in 1:length(cval)) 
#   skew[i] <- skewness(log(cval[i] + trainingSet$loss))
# plot(cval, skew, type = "l", ylab = expression(b[3](c)), xlab = expression(c))
# abline(h = 0, lty = 3)
# 
# best.c <- optimise(skew.score, c(0, 500), x = trainingSet$loss)$minimum
# best.c
# 
# ozone.transformed <- log(ozone + best.c)
# hist(ozone.transformed, col = "azure")
# 
# skewness(ozone.transformed)
# 
# qqnorm(log(trainingSet$loss+200))
# qqline(log(trainingSet$loss+200))
# 



#names(testingSet)
feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("id","loss", "CVindices" ))])
testfeature.names <- names(testingSet[,-which(names(testingSet) %in% c("id" ))])


##################################################################################

##################################################################################

# order of columns are matching 
##################################################################################

# Check for all columns order, all columns orders are same after removing id columns
constant = 200

cv          = 5
bags        = 1
nround.cv   = 660 
printeveryn = 10
seed        = 2021

## for all remaining models, use same parameters 

param <- list(  "objective"        = fairobj,
                #objective = logcoshobj, #fairobj #logcoshobj, #cauchyobj #,#'reg:linear',
                "booster"          = "gbtree",
                #"eval_metric"      = "auc",
                "nthread"          = 18,     
                "max_depth"        = 12,     
                "eta"              = 0.03, 
                "subsample"        = 0.7,  
                "colsample_bytree" = 0.7,
#                 "alpha"            =  1,
#                 "gamma"            =  1,
                "min_child_weight" = 100     
                
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
  
  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=log(X_build$loss+constant))
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=log(X_val$loss+constant))
  watchlist <- list( val = dval,train = dtrain)
  
  
  pred_cv_bags   <- rep(0, nrow(X_val[, feature.names]))
  pred_test_bags <- rep(0, nrow(testingSet[,testfeature.names]))
  
  for (b in 1:bags) 
  {
    seed = seed + b
    cat(seed , " - Random Seed\n ")
    cat(b ," - bag Processing\n")
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
    
    pred_cv_bags   <- pred_cv_bags + (exp(pred_cv) - constant)
    pred_test_bags <- pred_test_bags + (exp(pred_test) - constant)
  }
  pred_cv_bags   <- pred_cv_bags / bags
  pred_test_bags <- pred_test_bags / bags
  
  cat("CV Fold-", i, " ", metric, ": ", score(X_val$loss, pred_cv_bags, metric), "\n", sep = "")
  
  val_predictions <- data.frame(id=X_val$id, loss = pred_cv_bags)
  test_predictions <- data.frame(id=testingSet$id, loss = pred_test_bags)
  
  # if(i == 1)
  # {
  #   write.csv(val_predictions,  'prav.xgb10.fold1.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.xgb10.fold1-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 2)
  # {
  #   write.csv(val_predictions,  'prav.xgb10.fold2.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.xgb10.fold2-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 3)
  # {
  #   write.csv(val_predictions,  'prav.xgb10.fold3.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.xgb10.fold3-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 4)
  # {
  #   write.csv(val_predictions,  'prav.xgb10.fold4.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.xgb10.fold4-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 5)
  # {
  #   write.csv(val_predictions,  'prav.xgb10.fold5.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.xgb10.fold5-test.csv', row.names=FALSE, quote = FALSE)
  # }
  
}

# Full training

dtrain<-xgb.DMatrix(data=data.matrix(trainingSet[, feature.names]),label=log(trainingSet$loss+constant))
watchlist <- list( train = dtrain)

fulltrainnrounds = 1.2 * nround.cv


# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

fulltest_ensemble <- rep(0, nrow(testingSet[,testfeature.names]))


for (b in 1:bags) {
  cat(b ," - bag Processing\n")
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
  
  fulltest_ensemble <- fulltest_ensemble + (exp(predfull_test) - constant)
  
}

fulltest_ensemble <- fulltest_ensemble / bags
testfull_predictions  <- data.frame(id=testingSet$id, loss = fulltest_ensemble)
write.csv(testfull_predictions, './submissions/prav.xgb10.full.csv', row.names=FALSE, quote = FALSE)




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



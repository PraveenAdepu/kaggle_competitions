train_01 <- read_csv("./input/build_set_FE_01.csv")
train_02 <- read_csv("./input/build_set_FE_30.csv")

build_set <- left_join(train_01, train_02, by = "Patient_ID")

trainingSet <- subset(build_set, Patient_ID < 279201 )
testingSet  <- subset(build_set, Patient_ID >= 279201 )

##################################################################################################################
CVSchema  <- read_csv("./CVSchema/Prav_CVindices_5folds_10.csv") 

names(CVSchema)

trainingSet <- left_join(trainingSet, CVSchema, by = "Patient_ID")


########################################################################################################################
# 246 features
feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c('Patient_ID', 'CVindices','DiabetesDispense'
                                                                         ))])


trainingSet[, feature.names][is.na(trainingSet[, feature.names])] <- 0 
testingSet[, feature.names][is.na(testingSet[, feature.names])]   <- 0 


cv          = 5
bags        = 5
nround.cv   = 610
printeveryn = 50
seed        = 2017

param <- list(  "objective"        = "reg:linear", 
                "booster"          = "gbtree",
                "eval_metric"      = "auc",
                "nthread"          = 25,     
                "max_depth"        = 8,     
                "eta"              = 0.01, 
                "subsample"        = 0.7,  
                "colsample_bytree" = 0.7,  
                "min_child_weight" = 1     
                
)

set.seed(seed)
for (i in 1:cv)
  
{
  cat(i ,"fold Processing\n")
  
  cat("X_build fold Processing\n")
  X_build <- subset(trainingSet, CVindices != i, select = -c( CVindices))
  cat("X_val fold Processing\n")
  X_val   <- subset(trainingSet, CVindices == i, select = -c( CVindices)) 
  
  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=X_build$DiabetesDispense)
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=X_val$DiabetesDispense)
  watchlist <- list( val = dval,train = dtrain)
  pred_cv_bags   <- rep(0, nrow(X_val[, feature.names]))
  pred_test_bags <- rep(0, nrow(testingSet[,feature.names]))
  for (b in 1:bags) 
  {
    cat(b ," - bag Processing\n")
    cat("X_build training Processing\n")
    XGModel <- xgb.train(   params              = param, 
                            data                = dtrain,
                            watchlist           = watchlist,
                            nrounds             = nround.cv ,
                            print_every_n       = printeveryn,
                            verbose             = TRUE, 
                            maximize            = TRUE,
                            set.seed            = seed
    )
    cat("X_val prediction Processing\n")
    pred_cv  <- predict(XGModel, data.matrix(X_val[,feature.names]))
    
    
    cat("CV TestingSet prediction Processing\n")
    pred_test  <- predict(XGModel, data.matrix(testingSet[,feature.names]))
    
    pred_cv_bags   <- pred_cv_bags + pred_cv
    pred_test_bags <- pred_test_bags + pred_test
  }
  pred_cv_bags   <- pred_cv_bags / bags
  pred_test_bags <- pred_test_bags / bags
  cat("CV Fold-", i, " ", metric, ": ", score(X_val$DiabetesDispense, pred_cv_bags, metric), "\n", sep = "")
  val_predictions <- data.frame(Patient_ID=X_val$Patient_ID, Diabetes = pred_cv_bags)
  test_predictions <- data.frame(Patient_ID=testingSet$Patient_ID, Diabetes = pred_test_bags)
  
  if(i == 1)
  {
    write.csv(val_predictions,  './submissions/prav.xgb21.fold1.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb21.fold1-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 2)
  {
    write.csv(val_predictions,  './submissions/prav.xgb21.fold2.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb21.fold2-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 3)
  {
    write.csv(val_predictions,  './submissions/prav.xgb21.fold3.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb21.fold3-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 4)
  {
    write.csv(val_predictions,  './submissions/prav.xgb21.fold4.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb21.fold4-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 5)
  {
    write.csv(val_predictions,  './submissions/prav.xgb21.fold5.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb21.fold5-test.csv', row.names=FALSE, quote = FALSE)
  }

}

##########################################################################################################
# Full training

dtrain<-xgb.DMatrix(data=data.matrix(trainingSet[, feature.names]),label=trainingSet$DiabetesDispense)
watchlist <- list( train = dtrain)

fulltrainnrounds = 1.2 * nround.cv


# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

fulltest_ensemble <- rep(0, nrow(testingSet[,feature.names]))


for (b in 1:bags) {
  seed = seed + b
  cat(seed, "- seed")
  cat(b ," - bag Processing\n")
  cat("Bagging Full TrainingSet training\n")
  XGModelFulltrain <- xgb.train(    params              = param, 
                                    data                = dtrain,
                                    watchlist           = watchlist,
                                    nrounds             = fulltrainnrounds,
                                    print_every_n       = printeveryn,
                                    verbose             = TRUE, 
                                    maximize            = TRUE,
                                    set.seed            = seed
  )
  cat("Bagging Full Model prediction Processing\n")
  
  predfull_test         <- predict(XGModelFulltrain, data.matrix(testingSet[,feature.names]))
  fulltest_ensemble     <- fulltest_ensemble + predfull_test
  
}

fulltest_ensemble     <- fulltest_ensemble / bags
testfull_predictions  <- data.frame(Patient_ID=testingSet$Patient_ID, Diabetes = fulltest_ensemble)
write.csv(testfull_predictions, './submissions/prav.xgb21.full.csv', row.names=FALSE, quote = FALSE)


# #########################################################################################################
# Bagging - Full train
# #########################################################################################################


##########################################################################################################

BenchMark <- read.csv("./submissions/Prav_benchmark_01.csv")
val_sub <- read.csv("./submissions/prav.xgb21.full.csv")

MergeSub <- left_join(BenchMark, val_sub, by ="Patient_ID")

head(MergeSub,10)
tail(MergeSub,10)

summary(MergeSub$Diabetes.y)

MergeSub <- arrange(MergeSub,Diabetes.y)
head(MergeSub,10)
tail(MergeSub,10)
MergeSub$Diabetes_rank <-  rank(MergeSub$Diabetes.y)
MergeSub$Diabetes <- MergeSub$Diabetes_rank / max(MergeSub$Diabetes_rank)

Final_sub <- MergeSub[,c("Patient_ID","Diabetes")]
head(Final_sub,10)
write.csv(Final_sub,  './submissions/Prav_xgb21_rank.csv', row.names=FALSE, quote = FALSE)
#########################################################################################################

############################################################################################
model = xgb.dump(XGModelFulltrain, with_stats=TRUE)

names = dimnames(trainingSet[,feature.names])[[2]]
importance_matrix = xgb.importance( names , model = XGModel)
#xgb.plot.importance(importance_matrix)
impMatrix <- as.data.frame(importance_matrix)
# impMatrix[1:12,1]
# impMatrix[,1]
ImpFeature <- impMatrix[,1]


# #########################################################################################################


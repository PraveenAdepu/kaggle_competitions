####################################################################################
# L2 Stacking
####################################################################################

CVindices5folds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")
train <- read_csv('./input/train.csv')
train.cols <- c('id',"loss")
train <- train[, train.cols]
train <- left_join(train, CVindices5folds, by = "id", all.x = TRUE)
rm(CVindices5folds); gc()

test <- read_csv('./input/test.csv')
test.cols <- c('id')
test <- test[, test.cols]


xgb04.fold1 <- read_csv("./submissions/prav.xgb04.fold1.csv")
xgb04.fold2 <- read_csv("./submissions/prav.xgb04.fold2.csv")
xgb04.fold3 <- read_csv("./submissions/prav.xgb04.fold3.csv")
xgb04.fold4 <- read_csv("./submissions/prav.xgb04.fold4.csv")
xgb04.fold5 <- read_csv("./submissions/prav.xgb04.fold5.csv")
xgb04.full  <- read_csv("./submissions/prav.xgb04.full.csv")

xgb04 <- rbind(xgb04.fold1,xgb04.fold2,xgb04.fold3,xgb04.fold4,xgb04.fold5)
names(xgb04)
colnames(xgb04)<-c("id","xgb04loss")
names(xgb04)

names(xgb04.full)
colnames(xgb04.full)<-c("id","xgb04loss")
names(xgb04.full)
rm(xgb04.fold1,xgb04.fold2,xgb04.fold3,xgb04.fold4,xgb04.fold5); gc()

xgb05.fold1 <- read_csv("./submissions/prav.xgb05.fold1.csv")
xgb05.fold2 <- read_csv("./submissions/prav.xgb05.fold2.csv")
xgb05.fold3 <- read_csv("./submissions/prav.xgb05.fold3.csv")
xgb05.fold4 <- read_csv("./submissions/prav.xgb05.fold4.csv")
xgb05.fold5 <- read_csv("./submissions/prav.xgb05.fold5.csv")
xgb05.full  <- read_csv("./submissions/prav.xgb05.full.csv")

xgb05 <- rbind(xgb05.fold1,xgb05.fold2,xgb05.fold3,xgb05.fold4,xgb05.fold5)
names(xgb05)
colnames(xgb05)<-c("id","xgb05loss")
names(xgb05)

names(xgb05.full)
colnames(xgb05.full)<-c("id","xgb05loss")
names(xgb05.full)
rm(xgb05.fold1,xgb05.fold2,xgb05.fold3,xgb05.fold4,xgb05.fold5); gc()

xgb06.fold1 <- read_csv("./submissions/prav.xgb06.fold1.csv")
xgb06.fold2 <- read_csv("./submissions/prav.xgb06.fold2.csv")
xgb06.fold3 <- read_csv("./submissions/prav.xgb06.fold3.csv")
xgb06.fold4 <- read_csv("./submissions/prav.xgb06.fold4.csv")
xgb06.fold5 <- read_csv("./submissions/prav.xgb06.fold5.csv")
xgb06.full  <- read_csv("./submissions/prav.xgb06.full.csv")

xgb06 <- rbind(xgb06.fold1,xgb06.fold2,xgb06.fold3,xgb06.fold4,xgb06.fold5)
names(xgb06)
colnames(xgb06)<-c("id","xgb06loss")
names(xgb06)

names(xgb06.full)
colnames(xgb06.full)<-c("id","xgb06loss")
names(xgb06.full)
rm(xgb06.fold1,xgb06.fold2,xgb06.fold3,xgb06.fold4,xgb06.fold5); gc()

kerasnnet.fold1 <- read_csv("./submissions/prav.kerasnnet.fold1.csv")
kerasnnet.fold2 <- read_csv("./submissions/prav.kerasnnet.fold2.csv")
kerasnnet.fold3 <- read_csv("./submissions/prav.kerasnnet.fold3.csv")
kerasnnet.fold4 <- read_csv("./submissions/prav.kerasnnet.fold4.csv")
kerasnnet.fold5 <- read_csv("./submissions/prav.kerasnnet.fold5.csv")
kerasnnet.full  <- read_csv("./submissions/prav.kerasnnet.full.csv")

kerasnnet <- rbind(kerasnnet.fold1,kerasnnet.fold2,kerasnnet.fold3,kerasnnet.fold4,kerasnnet.fold5)
names(kerasnnet)
colnames(kerasnnet)<-c("id","kerasnnetloss")
names(kerasnnet)

names(kerasnnet.full)
colnames(kerasnnet.full)<-c("id","kerasnnetloss")
names(kerasnnet.full)
rm(kerasnnet.fold1,kerasnnet.fold2,kerasnnet.fold3,kerasnnet.fold4,kerasnnet.fold5); gc()


kerasnnet.all       <- read_csv("./submissions/preds_oob_keras_01.csv")
kerasnnet.all.full  <- read_csv("./submissions/submission_keras_01.csv")

names(kerasnnet.all)
colnames(kerasnnet.all)<-c("id","kerasnnetallloss")
names(kerasnnet.all)

names(kerasnnet.all.full)
colnames(kerasnnet.all.full)<-c("id","kerasnnetallloss")
names(kerasnnet.all.full)

train <- left_join(train, xgb04,      by="id", all.X = TRUE)
test  <- left_join(test , xgb04.full, by="id", all.X = TRUE)
rm(xgb04,xgb04.full); gc()

train <- left_join(train, xgb05,      by="id", all.X = TRUE)
test  <- left_join(test , xgb05.full, by="id", all.X = TRUE)
rm(xgb05,xgb05.full); gc()

train <- left_join(train, xgb06,      by="id", all.X = TRUE)
test  <- left_join(test , xgb06.full, by="id", all.X = TRUE)
rm(xgb06,xgb06.full); gc()

train <- left_join(train, kerasnnet,      by="id", all.X = TRUE)
test  <- left_join(test , kerasnnet.full, by="id", all.X = TRUE)
rm(kerasnnet,kerasnnet.full); gc()

train <- left_join(train, kerasnnet.all,      by="id", all.X = TRUE)
test  <- left_join(test , kerasnnet.all.full, by="id", all.X = TRUE)
rm(kerasnnet.all,kerasnnet.all.full); gc()


names(train)
names(test)

trainingSet <- train
testingSet  <- test

rm(train, test); gc()

feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("id","loss", "CVindices" ))])
testfeature.names <- names(testingSet[,-which(names(testingSet) %in% c("id" ))])

##################################################################################

# order of columns are matching 
##################################################################################

# Check for all columns order, all columns orders are same after removing id columns


cv          = 5
bags        = 2
nround.cv   = 200 
printeveryn = 20
seed        = 2016

## for all remaining models, use same parameters 

param <- list(  "objective"        = "reg:linear", 
                "booster"          = "gbtree",
                #"eval_metric"      = "auc",
                "nthread"          = 7,     
                "max_depth"        = 4,     
                "eta"              = 0.05, 
                "subsample"        = 0.5,  
                "colsample_bytree" = 0.5,  
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
  
#   val_predictions <- data.frame(id=X_val$id, loss = pred_cv_bags)
#   test_predictions <- data.frame(id=testingSet$id, loss = pred_test_bags)
#   
#   if(i == 1)
#   {
#     write.csv(val_predictions,  'prav.xgb05.fold1.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.xgb05.fold1-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 2)
#   {
#     write.csv(val_predictions,  'prav.xgb05.fold2.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.xgb05.fold2-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 3)
#   {
#     write.csv(val_predictions,  'prav.xgb05.fold3.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.xgb05.fold3-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 4)
#   {
#     write.csv(val_predictions,  'prav.xgb05.fold4.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.xgb05.fold4-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 5)
#   {
#     write.csv(val_predictions,  'prav.xgb05.fold5.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.xgb05.fold5-test.csv', row.names=FALSE, quote = FALSE)
#   }
  
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
write.csv(testfull_predictions, './submissions/StackingL2/prav.L2.xgb01.csv', row.names=FALSE, quote = FALSE)









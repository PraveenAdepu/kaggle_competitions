####################################################################################
# L2 Stacking
####################################################################################

CVindices5folds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")


CNN02.fold1 <- read_csv("./submissions/1prav.CNN02.fold1.csv")
CNN02.fold2 <- read_csv("./submissions/1prav.CNN02.fold2.csv")
CNN02.fold3 <- read_csv("./submissions/1prav.CNN02.fold3.csv")
CNN02.fold4 <- read_csv("./submissions/1prav.CNN02.fold4.csv")
CNN02.fold5 <- read_csv("./submissions/1prav.CNN02.fold5.csv")
CNN02.full  <- read_csv("./submissions/1prav.CNN02.full.csv")

xgb20 <- rbind(xgb20.fold1,xgb20.fold2,xgb20.fold3,xgb20.fold4,xgb20.fold5)
names(xgb20)
colnames(xgb20)<-c("id","xgb20loss")
names(xgb20)

names(xgb20.full)
colnames(xgb20.full)<-c("id","xgb20loss")
names(xgb20.full)
rm(xgb20.fold1,xgb20.fold2,xgb20.fold3,xgb20.fold4,xgb20.fold5); gc()

keras20.fold1 <- read_csv("./submissions/prav.keras20.fold1.csv")
keras20.fold2 <- read_csv("./submissions/prav.keras20.fold2.csv")
keras20.fold3 <- read_csv("./submissions/prav.keras20.fold3.csv")
keras20.fold4 <- read_csv("./submissions/prav.keras20.fold4.csv")
keras20.fold5 <- read_csv("./submissions/prav.keras20.fold5.csv")
keras20.full  <- read_csv("./submissions/prav.keras20.full.csv")

keras20 <- rbind(keras20.fold1,keras20.fold2,keras20.fold3,keras20.fold4,keras20.fold5)
names(keras20)
colnames(keras20)<-c("id","keras20loss")
names(keras20)

names(keras20.full)
colnames(keras20.full)<-c("id","keras20loss")
names(keras20.full)
rm(keras20.fold1,keras20.fold2,keras20.fold3,keras20.fold4,keras20.fold5); gc()

dl20.fold1 <- read_csv("./submissions/prav.dl20.fold1.csv")
dl20.fold2 <- read_csv("./submissions/prav.dl20.fold2.csv")
dl20.fold3 <- read_csv("./submissions/prav.dl20.fold3.csv")
dl20.fold4 <- read_csv("./submissions/prav.dl20.fold4.csv")
dl20.fold5 <- read_csv("./submissions/prav.dl20.fold5.csv")
dl20.full  <- read_csv("./submissions/prav.dl20.full.csv")

dl20 <- rbind(dl20.fold1,dl20.fold2,dl20.fold3,dl20.fold4,dl20.fold5)
names(dl20)
colnames(dl20)<-c("id","dl20loss")
names(dl20)

names(dl20.full)
colnames(dl20.full)<-c("id","dl20loss")
names(dl20.full)
rm(dl20.fold1,dl20.fold2,dl20.fold3,dl20.fold4,dl20.fold5); gc()

dl21.fold1 <- read_csv("./submissions/prav.dl21.fold1.csv")
dl21.fold2 <- read_csv("./submissions/prav.dl21.fold2.csv")
dl21.fold3 <- read_csv("./submissions/prav.dl21.fold3.csv")
dl21.fold4 <- read_csv("./submissions/prav.dl21.fold4.csv")
dl21.fold5 <- read_csv("./submissions/prav.dl21.fold5.csv")
dl21.full  <- read_csv("./submissions/prav.dl21.full.csv")

dl21 <- rbind(dl21.fold1,dl21.fold2,dl21.fold3,dl21.fold4,dl21.fold5)
names(dl21)
colnames(dl21)<-c("id","dl21loss")
names(dl21)

names(dl21.full)
colnames(dl21.full)<-c("id","dl21loss")
names(dl21.full)
rm(dl21.fold1,dl21.fold2,dl21.fold3,dl21.fold4,dl21.fold5); gc()

rf20.fold1 <- read_csv("./submissions/prav.rf20.fold1.csv")
rf20.fold2 <- read_csv("./submissions/prav.rf20.fold2.csv")
rf20.fold3 <- read_csv("./submissions/prav.rf20.fold3.csv")
rf20.fold4 <- read_csv("./submissions/prav.rf20.fold4.csv")
rf20.fold5 <- read_csv("./submissions/prav.rf20.fold5.csv")
rf20.full  <- read_csv("./submissions/prav.rf20.full.csv")

rf20 <- rbind(rf20.fold1,rf20.fold2,rf20.fold3,rf20.fold4,rf20.fold5)
names(rf20)
colnames(rf20)<-c("id","rf20loss")
names(rf20)

names(rf20.full)
colnames(rf20.full)<-c("id","rf20loss")
names(rf20.full)
rm(rf20.fold1,rf20.fold2,rf20.fold3,rf20.fold4,rf20.fold5); gc()

et20.fold1 <- read_csv("./submissions/prav.et20.fold1.csv")
et20.fold2 <- read_csv("./submissions/prav.et20.fold2.csv")
et20.fold3 <- read_csv("./submissions/prav.et20.fold3.csv")
et20.fold4 <- read_csv("./submissions/prav.et20.fold4.csv")
et20.fold5 <- read_csv("./submissions/prav.et20.fold5.csv")
et20.full  <- read_csv("./submissions/prav.et20.full.csv")

et20 <- rbind(et20.fold1,et20.fold2,et20.fold3,et20.fold4,et20.fold5)
names(et20)
colnames(et20)<-c("id","et20loss")
names(et20)

names(et20.full)
colnames(et20.full)<-c("id","et20loss")
names(et20.full)
rm(et20.fold1,et20.fold2,et20.fold3,et20.fold4,et20.fold5); gc()

xgb21.fold1 <- read_csv("./submissions/prav.xgb21.fold1.csv")
xgb21.fold2 <- read_csv("./submissions/prav.xgb21.fold2.csv")
xgb21.fold3 <- read_csv("./submissions/prav.xgb21.fold3.csv")
xgb21.fold4 <- read_csv("./submissions/prav.xgb21.fold4.csv")
xgb21.fold5 <- read_csv("./submissions/prav.xgb21.fold5.csv")
xgb21.full  <- read_csv("./submissions/prav.xgb21.full.csv")

xgb21 <- rbind(xgb21.fold1,xgb21.fold2,xgb21.fold3,xgb21.fold4,xgb21.fold5)
names(xgb21)
colnames(xgb21)<-c("id","xgb21loss")
names(xgb21)

names(xgb21.full)
colnames(xgb21.full)<-c("id","xgb21loss")
names(xgb21.full)
rm(xgb21.fold1,xgb21.fold2,xgb21.fold3,xgb21.fold4,xgb21.fold5); gc()

keras22.fold1 <- read_csv("./submissions/prav.keras22.fold1.csv")
keras22.fold2 <- read_csv("./submissions/prav.keras22.fold2.csv")
keras22.fold3 <- read_csv("./submissions/prav.keras22.fold3.csv")
keras22.fold4 <- read_csv("./submissions/prav.keras22.fold4.csv")
keras22.fold5 <- read_csv("./submissions/prav.keras22.fold5.csv")
keras22.full  <- read_csv("./submissions/prav.keras22.full.csv")

keras22 <- rbind(keras22.fold1,keras22.fold2,keras22.fold3,keras22.fold4,keras22.fold5)
names(keras22)
colnames(keras22)<-c("id","keras22loss")
names(keras22)

names(keras22.full)
colnames(keras22.full)<-c("id","keras22loss")
names(keras22.full)
rm(keras22.fold1,keras22.fold2,keras22.fold3,keras22.fold4,keras22.fold5); gc()

# dl22.fold1 <- read_csv("./submissions/prav.dl22.fold1.csv")
# dl22.fold2 <- read_csv("./submissions/prav.dl22.fold2.csv")
# dl22.fold3 <- read_csv("./submissions/prav.dl22.fold3.csv")
# dl22.fold4 <- read_csv("./submissions/prav.dl22.fold4.csv")
# dl22.fold5 <- read_csv("./submissions/prav.dl22.fold5.csv")
# dl22.full  <- read_csv("./submissions/prav.dl22.full.csv")
# 
# dl22 <- rbind(dl22.fold1,dl22.fold2,dl22.fold3,dl22.fold4,dl22.fold5)
# names(dl22)
# colnames(dl22)<-c("id","dl22loss")
# names(dl22)
# 
# names(dl22.full)
# colnames(dl22.full)<-c("id","dl22loss")
# names(dl22.full)
# rm(dl22.fold1,dl22.fold2,dl22.fold3,dl22.fold4,dl22.fold5); gc()

rf21.fold1 <- read_csv("./submissions/prav.rf21.fold1.csv")
rf21.fold2 <- read_csv("./submissions/prav.rf21.fold2.csv")
rf21.fold3 <- read_csv("./submissions/prav.rf21.fold3.csv")
rf21.fold4 <- read_csv("./submissions/prav.rf21.fold4.csv")
rf21.fold5 <- read_csv("./submissions/prav.rf21.fold5.csv")
rf21.full  <- read_csv("./submissions/prav.rf21.full.csv")

rf21 <- rbind(rf21.fold1,rf21.fold2,rf21.fold3,rf21.fold4,rf21.fold5)
names(rf21)
colnames(rf21)<-c("id","rf21loss")
names(rf21)

names(rf21.full)
colnames(rf21.full)<-c("id","rf21loss")
names(rf21.full)
rm(rf21.fold1,rf21.fold2,rf21.fold3,rf21.fold4,rf21.fold5); gc()

# 
dl21.all       <- read_csv("./submissions/preds_oob_keras_02.csv")
dl21.all.full  <- read_csv("./submissions/submission_keras_02.csv")

names(dl21.all)
colnames(dl21.all)<-c("id","dl21allloss")
names(dl21.all)

names(dl21.all.full)
colnames(dl21.all.full)<-c("id","dl21allloss")
names(dl21.all.full)

train <- left_join(train, xgb20,      by="id", all.X = TRUE)
test  <- left_join(test , xgb20.full, by="id", all.X = TRUE)
rm(xgb20,xgb20.full); gc()

train <- left_join(train, keras20,      by="id", all.X = TRUE)
test  <- left_join(test , keras20.full, by="id", all.X = TRUE)
rm(keras20,keras20.full); gc()

train <- left_join(train, dl20,      by="id", all.X = TRUE)
test  <- left_join(test , dl20.full, by="id", all.X = TRUE)
rm(dl20,dl20.full); gc()

train <- left_join(train, dl21,      by="id", all.X = TRUE)
test  <- left_join(test , dl21.full, by="id", all.X = TRUE)
rm(dl21,dl21.full); gc()

train <- left_join(train, rf20,      by="id", all.X = TRUE)
test  <- left_join(test , rf20.full, by="id", all.X = TRUE)
rm(rf20,rf20.full); gc()

train <- left_join(train, et20,      by="id", all.X = TRUE)
test  <- left_join(test , et20.full, by="id", all.X = TRUE)
rm(et20,et20.full); gc()

train <- left_join(train, xgb21,      by="id", all.X = TRUE)
test  <- left_join(test , xgb21.full, by="id", all.X = TRUE)
rm(xgb21,xgb21.full); gc()

train <- left_join(train, keras22,      by="id", all.X = TRUE)
test  <- left_join(test , keras22.full, by="id", all.X = TRUE)
rm(keras22,keras22.full); gc()

# train <- left_join(train, dl22,      by="id", all.X = TRUE)
# test  <- left_join(test , dl22.full, by="id", all.X = TRUE)
# rm(dl22,dl22.full); gc()

train <- left_join(train, rf21,      by="id", all.X = TRUE)
test  <- left_join(test , rf21.full, by="id", all.X = TRUE)
rm(rf21,rf21.full); gc()


train <- left_join(train, dl21.all,      by="id", all.X = TRUE)
test  <- left_join(test , dl21.all.full, by="id", all.X = TRUE)
rm(dl21.all,dl21.all.full); gc()

names(train)
names(test)

trainingSet <- train
testingSet  <- test

rm(train, test); gc()

feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("id","loss", "CVindices","et20loss" ))])
testfeature.names <- names(testingSet[,-which(names(testingSet) %in% c("id","et20loss" ))])

# trainingSet[, feature.names] <- apply(trainingSet[, feature.names], 2, normalit)
# testingSet[, feature.names]  <- apply(testingSet[, feature.names], 2, normalit)
# 
# write.csv(trainingSet,  './input/prav.L2train.csv', row.names=FALSE, quote = FALSE)
# write.csv(testingSet,  './input/prav.L2test.csv', row.names=FALSE, quote = FALSE)
##################################################################################

cor(trainingSet[,feature.names])
# order of columns are matching 
##################################################################################

# Check for all columns order, all columns orders are same after removing id columns


cv          = 5
bags        = 10
nround.cv   = 200 
printeveryn = 20
seed        = 2016
constant = 200
## for all remaining models, use same parameters 

param <- list(  "objective"        = "reg:linear", 
                "booster"          = "gbtree",
                #"eval_metric"      = "auc",
                "nthread"          = 27,     
                "max_depth"        = 4,     
                "eta"              = 0.05, 
                "subsample"        = 0.7,  
                "colsample_bytree" = 0.7,  
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
  
  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=log(X_build$loss+constant))
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=log(X_val$loss+constant))
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
    
    pred_cv_bags   <- pred_cv_bags + (exp(pred_cv)-constant)
    pred_test_bags <- pred_test_bags + (exp(pred_test)-constant)
  }
  pred_cv_bags   <- pred_cv_bags / bags
  pred_test_bags <- pred_test_bags / bags
  
  cat("CV Fold-", i, " ", metric, ": ", score(X_val$loss, pred_cv_bags, metric), "\n", sep = "")
  
  #   val_predictions <- data.frame(id=X_val$id, loss = pred_cv_bags)
  #   test_predictions <- data.frame(id=testingSet$id, loss = pred_test_bags)
  #   
  #   if(i == 1)
  #   {
  #     write.csv(val_predictions,  'prav.keras20.fold1.csv', row.names=FALSE, quote = FALSE)
  #     write.csv(test_predictions, 'prav.keras20.fold1-test.csv', row.names=FALSE, quote = FALSE)
  #   }
  #   if(i == 2)
  #   {
  #     write.csv(val_predictions,  'prav.keras20.fold2.csv', row.names=FALSE, quote = FALSE)
  #     write.csv(test_predictions, 'prav.keras20.fold2-test.csv', row.names=FALSE, quote = FALSE)
  #   }
  #   if(i == 3)
  #   {
  #     write.csv(val_predictions,  'prav.keras20.fold3.csv', row.names=FALSE, quote = FALSE)
  #     write.csv(test_predictions, 'prav.keras20.fold3-test.csv', row.names=FALSE, quote = FALSE)
  #   }
  #   if(i == 4)
  #   {
  #     write.csv(val_predictions,  'prav.keras20.fold4.csv', row.names=FALSE, quote = FALSE)
  #     write.csv(test_predictions, 'prav.keras20.fold4-test.csv', row.names=FALSE, quote = FALSE)
  #   }
  #   if(i == 5)
  #   {
  #     write.csv(val_predictions,  'prav.keras20.fold5.csv', row.names=FALSE, quote = FALSE)
  #     write.csv(test_predictions, 'prav.keras20.fold5-test.csv', row.names=FALSE, quote = FALSE)
  #   }
  
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
  
  fulltest_ensemble <- fulltest_ensemble + (exp(predfull_test)-constant)
  
}

fulltest_ensemble <- fulltest_ensemble / bags
testfull_predictions  <- data.frame(id=testingSet$id, loss = fulltest_ensemble)
write.csv(testfull_predictions, './submissions/StackingL2/prav.L2.xgb30_9.csv', row.names=FALSE, quote = FALSE)

xgbl2_predictions <- testfull_predictions
########################################################################################################################
########################################################################################################################
## et model L2 stacking
########################################################################################################################
########################################################################################################################

options( java.parameters = "-Xmx50g" )
require("caret")
require("rJava")
require("extraTrees")

trainingSet$loss <- log(trainingSet$loss+200)

#names(trainingSet)
testingSet$loss <- 0
feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("id", "CVindices" ))])
testfeature.names <- names(testingSet[,-which(names(testingSet) %in% c("id" ))])

##################################################################################
names(trainingSet)
col_idx <- grep("loss", names(trainingSet))
trainingSet <- trainingSet[, c(col_idx, (1:ncol(trainingSet))[-col_idx])]
names(trainingSet)

names(testingSet)
col_idx <- grep("loss", names(testingSet))
testingSet <- testingSet[, c(col_idx, (1:ncol(testingSet))[-col_idx])]
names(testingSet)
##################################################################################

# order of columns are matching 
##################################################################################

# Check for all columns order, all columns orders are same after removing id columns


cv                 = 5
bags               = 2
seed               = 2016
#metric             = "auc"
Parammtry          = 8  # Regression 1/3 of variables
ParamnumThreads    = 27
Paramntree         = 500
ParamnumRandomCuts = 2

## for all remaining models, use same parameters 



cat(cv, "-fold Cross Validation\n", sep = "")

set.seed(seed)
for (i in 1:1)
  
{
  cat(i ,"fold Processing\n")
  
  cat("X_build fold Processing\n")
  X_build <- subset(trainingSet, CVindices != i, select = -c( CVindices))
  cat("X_val fold Processing\n")
  X_val   <- subset(trainingSet, CVindices == i, select = -c( CVindices)) 
  
  
  pred_cv_bags   <- rep(0, nrow(X_val[, feature.names]))
  pred_test_bags <- rep(0, nrow(testingSet[,testfeature.names]))
  
  for (b in 1:bags) 
  {
    cat(b ," - bag Processing\n")
    cat("X_build training Processing\n")
    set.seed(seed)
    etModel <- extraTrees(x = model.matrix(loss ~ ., data = X_build[,feature.names])
                          , y             = X_build$loss
                          , mtry          = Parammtry
                          , numThreads    = ParamnumThreads
                          , ntree         = Paramntree
                          , numRandomCuts = ParamnumRandomCuts
    )
    cat("X_val prediction Processing\n")
    pred_cv  <- predict(etModel, newdata = model.matrix(loss ~ .,X_val[,feature.names]))
    cat("CV TestingSet prediction Processing\n")
    pred_test  <- predict(etModel, newdata = model.matrix(loss ~ .,testingSet[,testfeature.names]))
    
    
    pred_cv_bags   <- pred_cv_bags + (exp(pred_cv)-200)
    pred_test_bags <- pred_test_bags + (exp(pred_test)-200)
  }
  pred_cv_bags   <- pred_cv_bags / bags
  pred_test_bags <- pred_test_bags / bags
  
  cat("CV Fold-", i, " ", metric, ": ", score((exp(X_val$loss)-200), pred_cv_bags, metric), "\n", sep = "")
  val_predictions <- data.frame(id=X_val$id, loss = pred_cv_bags)
  test_predictions <- data.frame(id=testingSet$id, loss = pred_test_bags)
  
  # if(i == 1)
  # {
  #   write.csv(val_predictions,  'prav.et20.fold1.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.et20.fold1-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 2)
  # {
  #   write.csv(val_predictions,  'prav.et20.fold2.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.et20.fold2-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 3)
  # {
  #   write.csv(val_predictions,  'prav.et20.fold3.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.et20.fold3-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 4)
  # {
  #   write.csv(val_predictions,  'prav.et20.fold4.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.et20.fold4-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 5)
  # {
  #   write.csv(val_predictions,  'prav.et20.fold5.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.et20.fold5-test.csv', row.names=FALSE, quote = FALSE)
  # }
  
}

# Full training


# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

fulltest_ensemble <- rep(0, nrow(testingSet[,testfeature.names]))


for (b in 1:bags) {
  cat(b ," - bag Processing\n")
  cat("Bagging Full TrainingSet training\n")
  set.seed(seed)
  etFullModel <- extraTrees(x = model.matrix(loss ~ ., data = trainingSet[,feature.names])
                            , y = trainingSet$loss
                            , mtry          = Parammtry
                            , numThreads    = ParamnumThreads
                            , ntree         = Paramntree
                            , numRandomCuts = ParamnumRandomCuts
  )
  
  cat("Bagging Full Model prediction Processing\n")
  predfull_test  <- predict(etFullModel, newdata = model.matrix(loss ~ .,testingSet[,testfeature.names]))
  
  fulltest_ensemble <- fulltest_ensemble + (exp(predfull_test)-200)
  
}

fulltest_ensemble <- fulltest_ensemble / bags
testfull_predictions  <- data.frame(id=testingSet$id, loss = fulltest_ensemble)
write.csv(testfull_predictions, './submissions/StackingL2/prav.L2.et30_1.csv', row.names=FALSE, quote = FALSE)

etl2_predictions <- testfull_predictions
#####################################################################################################################

L2Models_predictions <-  left_join(xgbl2_predictions, etl2_predictions,   by="id", all.X = TRUE)

cor(L2Models_predictions[2:3])
head(L2Models_predictions)
L2Models_predictions$loss <- L2Models_predictions$loss.x * 0.5 + L2Models_predictions$loss.y  * 0.5
head(L2Models_predictions)
L2Models_predictions$loss.x  <- NULL
L2Models_predictions$loss.y  <- NULL

write.csv(L2Models_predictions, './submissions/StackingL2/prav.L3.XGBet40_1.csv', row.names=FALSE, quote = FALSE)

# #########################################################################################################
# Bagging - Full train
# #########################################################################################################








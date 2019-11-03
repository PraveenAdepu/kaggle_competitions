# rm(list=ls())
################################################################################################

# 20160921 - Baseline01 -- 2 features -- R Image

# Sys.time()
# load("Allstate_Baseline01_20161014.RData"); gc()
# Sys.time()
################################################################################################
# IMO : Used from Source files

train <- read_csv('./input/train.csv')
test  <- read_csv('./input/test.csv')

CVindices5folds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")
names(train)
summary(train$loss)
names(test)
test$loss <- -100
summary(test$loss)

train_test = rbind(train, test)

LETTERS_AY <- LETTERS[-length(LETTERS)]
LETTERS702 <- c(LETTERS_AY, sapply(LETTERS_AY, function(x) paste0(x, LETTERS_AY)), "ZZ")


feature.names     <- names(train[,-which(names(train) %in% c("id","loss"))])

for (f in feature.names) {
  if (class(train_test[[f]])=="character") {
    levels <- intersect(LETTERS702, unique(train_test[[f]])) # get'em ordered!
    labels <- match(levels, LETTERS702)
    #train_test[[f]] <- factor(train_test[[f]], levels=levels) # uncomment this for one-hot
    train_test[[f]] <- as.integer(as.character(factor(train_test[[f]], levels=levels, labels = labels))) # comment this one away for one-hot
  }
}

training <- train_test[train_test$loss != -100,]
testing  <- train_test[train_test$loss == -100,]

summary(training$loss)
summary(testing$loss)

testing$loss <- NULL

trainingSet <- left_join(training, CVindices5folds, by = "id")
testingSet  <- testing

rm(train,test, training, testing, train_test ,CVindices5folds); gc()


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
nround.cv   = 420 
printeveryn = 100
seed        = 2016

## for all remaining models, use same parameters 

param <- list(  "objective"        = "reg:linear", 
                "booster"          = "gbtree",
                #"eval_metric"      = "auc",
                "nthread"          = 7,     
                "max_depth"        = 7,     
                "eta"              = 0.1, 
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
  
  X_build.sparse    <- sparse.model.matrix( ~ .-1, data = X_build[, feature.names])
  X_val.sparse      <- sparse.model.matrix( ~ .-1, data = X_val[, feature.names])
  testingSet.sparse <- sparse.model.matrix( ~ .-1, data = testingSet[, testfeature.names])
  
  #dim(X_build.sparse); dim(X_val.sparse) ; dim(testingSet.sparse)
  dtrain <-xgb.DMatrix(data=X_build.sparse,label=log(X_build$loss))
  dval   <-xgb.DMatrix(data=X_val.sparse  ,label=log(X_val$loss))
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
    pred_cv    <- predict(XGModel, X_val.sparse)
    cat("CV TestingSet prediction Processing\n")
    pred_test  <- predict(XGModel, testingSet.sparse)
    
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
    write.csv(val_predictions,  'prav.xgb06.fold1.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, 'prav.xgb06.fold1-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 2)
  {
    write.csv(val_predictions,  'prav.xgb06.fold2.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, 'prav.xgb06.fold2-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 3)
  {
    write.csv(val_predictions,  'prav.xgb06.fold3.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, 'prav.xgb06.fold3-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 4)
  {
    write.csv(val_predictions,  'prav.xgb06.fold4.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, 'prav.xgb06.fold4-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 5)
  {
    write.csv(val_predictions,  'prav.xgb06.fold5.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, 'prav.xgb06.fold5-test.csv', row.names=FALSE, quote = FALSE)
  }
  
}

# Full training

trainingSet.sparse    <- sparse.model.matrix( ~ .-1, data = trainingSet[, feature.names])

dtrain<-xgb.DMatrix(data=trainingSet.sparse,label=log(trainingSet$loss))
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
  
  predfull_test         <- predict(XGModelFulltrain, testingSet.sparse)
  
  fulltest_ensemble <- fulltest_ensemble + exp(predfull_test)
  
}

fulltest_ensemble <- fulltest_ensemble / bags
testfull_predictions  <- data.frame(id=testingSet$id, loss = fulltest_ensemble)
write.csv(testfull_predictions, './submissions/prav.xgb06.full.csv', row.names=FALSE, quote = FALSE)




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


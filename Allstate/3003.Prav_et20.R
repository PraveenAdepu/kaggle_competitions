options( java.parameters = "-Xmx50g" )
require("caret")
require("rJava")
require("extraTrees")


train <- read_csv('./input/train.csv')
test  <- read_csv('./input/test.csv')

CVindices5folds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")
test$loss <- -100

train_test = rbind(train, test)


LETTERS_AY <- LETTERS[-length(LETTERS)]
LETTERS702 <- c(LETTERS_AY, sapply(LETTERS_AY, function(x) paste0(x, LETTERS_AY)), "ZZ")

feature.names     <- names(train_test[,-which(names(train_test) %in% c("id","loss"))])

for (f in feature.names) {
  if (class(train_test[[f]])=="character") {
    levels <- intersect(LETTERS702, unique(train_test[[f]])) # get'em ordered!
    labels <- match(levels, LETTERS702)
    #train_test[[f]] <- factor(train_test[[f]], levels=levels) # uncomment this for one-hot
    train_test[[f]] <- as.integer(as.character(factor(train_test[[f]], levels=levels, labels = labels))) # comment this one away for one-hot
  }
}

cont.features <- grep("con", names(train_test), value = TRUE)

for (f in cont.features) {
  if (class(train_test[[f]])=="numeric" & (skewness(train_test[[f]]) > 0.25 | skewness(train_test[[f]]) < -0.25)) {
    lambda = BoxCox.lambda( train_test[[f]] )
    skewness = skewness( train_test[[f]] )
    kurtosis = kurtosis( train_test[[f]] )
    cat("VARIABLE : ",f, "lambda : ",lambda, "skewness : ",skewness, "kurtosis : ",kurtosis, "\n")
    train_test[[f]] = BoxCox( train_test[[f]], lambda)
    
  }
}



training <- train_test[train_test$loss != -100,]
testing  <- train_test[train_test$loss == -100,]

rm(train,test); gc()
summary(training$loss)
summary(testing$loss)

testing$loss <- NULL

trainFeatures <- read_csv('./input/trainFeatures.csv')
testFeatures  <- read_csv('./input/testFeatures.csv')

training <- left_join(training, trainFeatures, by = "id")
testing  <- left_join(testing, testFeatures, by = "id")

trainingSet <- left_join(training, CVindices5folds, by = "id")
rm(training); gc()
rm(train_test); gc()
testingSet  <- testing

rm(testing,  CVindices5folds,trainFeatures,testFeatures); gc()



# 


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
Parammtry          = 20  # Regression 1/3 of variables
ParamnumThreads    = 27
Paramntree         = 500
ParamnumRandomCuts = 2

## for all remaining models, use same parameters 



cat(cv, "-fold Cross Validation\n", sep = "")

set.seed(seed)
for (i in 1:cv)
  
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
  
  if(i == 1)
  {
    write.csv(val_predictions,  'prav.et20.fold1.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, 'prav.et20.fold1-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 2)
  {
    write.csv(val_predictions,  'prav.et20.fold2.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, 'prav.et20.fold2-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 3)
  {
    write.csv(val_predictions,  'prav.et20.fold3.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, 'prav.et20.fold3-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 4)
  {
    write.csv(val_predictions,  'prav.et20.fold4.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, 'prav.et20.fold4-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 5)
  {
    write.csv(val_predictions,  'prav.et20.fold5.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, 'prav.et20.fold5-test.csv', row.names=FALSE, quote = FALSE)
  }
  
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
write.csv(testfull_predictions, './submissions/prav.et20.full.csv', row.names=FALSE, quote = FALSE)




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


require(h2o)

# rm(list=ls())
################################################################################################

# 20160921 - Baseline01 -- 2 features -- R Image
# Sys.time()
# load("Allstate_Baseline01_20161014.RData"); gc()
# Sys.time()
################################################################################################



 
## Create an H2O cloud 
h2o.init(
  nthreads=27,            
  max_mem_size = "75G")    
h2o.removeAll() 


trainingSet <- h2o.importFile("./input/train.csv", destination_frame = "trainingSet.hex")
testingSet  <- h2o.importFile("./input/test.csv", destination_frame = "testingSet.hex")

names(trainingSet)
CVindices5folds <- h2o.importFile("./CVSchema/Prav_CVindices_5folds.csv", destination_frame = "CVindices5folds.hex")
# 
trainingSet <- h2o.merge(trainingSet, CVindices5folds, all.x = TRUE)
# 
# trainingSet <- trainingSet2
# names(trainingSet)
# trainingSet <- as.h2o(trainingSet, destination_frame = "trainingSet.hex")
# testingSet  <- as.h2o(testingSet,  destination_frame = "testingSet.hex")

# cate.variables <- grep("cat", names(trainingSet), value = T)
# cate.Ratio.variables <- grep("Ratio", names(trainingSet), value = T)
# 
# onehot.variables <- setdiff(cate.variables, cate.Ratio.variables)
# 
# formula <-  as.formula(paste("~ ", paste(onehot.variables, collapse= "+"))) 
#  ohe_feats = onehot.variables
#   
# dummies <- dummyVars(formula, data = trainingSet)
# trainingSet_ohe <- as.data.frame(predict(dummies, newdata = trainingSet))
# trainingSet_combined <- cbind(trainingSet[,-c(which(colnames(trainingSet) %in% ohe_feats))],trainingSet_ohe)


feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("id","loss", "CVindices" ))]) 
testfeature.names <- names(testingSet[,-which(names(testingSet) %in% c("id"))])

# feature.names     <- setdiff(feature.names, cate.Ratio.variables)
# testfeature.names <- setdiff(testfeature.names, cate.Ratio.variables)
# trainingSet$loss <- log(trainingSet$loss)

# proc.time() - ptm

# order of columns are matching 
##################################################################################

# Check for all columns order, all columns orders are same after removing id columns
# names(trainingSet)
# names(testingSet)
 trainingSet$loss <- log(trainingSet$loss+200)

cv       = 5
bags     = 2
ntrees   = 1000
maxdepth = 15
mtry     = 20
seed     = 2016

## for all remaining models, use same parameters 

cat(cv, "-fold Cross Validation\n", sep = "")

set.seed(seed)
for (i in 1:cv)
{
  
  cat(i ,"fold Processing\n")
  cat("X_build fold Processing\n")
  X_build <- trainingSet[trainingSet$CVindices != i, colnames(trainingSet)]
  cat("X_val fold Processing\n")
  X_val   <- trainingSet[trainingSet$CVindices == i, colnames(trainingSet)]
  
  pred_cv_bags   <- as.data.frame(rep(0, nrow(X_val[, feature.names])))
  pred_test_bags <- as.data.frame(rep(0, nrow(testingSet[,testfeature.names])))
  
  for (b in 1:bags) 
  {
    cat(b ," - bag Processing\n")
    # seed = seed + b
    # set.seed(seed)
    cat(seed ," - Random Seed\n")
    cat("X_build training Processing\n")
    Model <- h2o.randomForest(         
                      training_frame   = X_build ,  
                      validation_frame = X_val,     
                      x                = feature.names,              
                      y                = "loss",                 
                      mtries           = mtry,
                      ntrees           = ntrees,              
                      max_depth        = maxdepth,         
                      seed=seed) 
    cat("X_val prediction Processing\n")
    pred_cv              <- h2o.predict(Model, X_val[,feature.names])
    pred_test            <- h2o.predict(Model, testingSet[,testfeature.names])
    pred_cv_bags         <- pred_cv_bags   + (exp(as.data.frame(pred_cv))-200)
    pred_test_bags       <- pred_test_bags + (exp(as.data.frame(pred_test))-200)
  }
  pred_cv_bags   <- pred_cv_bags / bags
  pred_test_bags <- pred_test_bags / bags
  
  ValSetId          <- as.data.frame(X_val$id)
  val_loss          <- exp(as.data.frame(X_val$loss))-200
  val_predictions    <- cbind(ValSetId, val_loss,pred_cv_bags)
  colnames(val_predictions)<-c("id","loss","pred_loss")
  
  cat("CV TestingSet prediction Processing\n")
  
  testingSetId      <- as.data.frame(testingSet$id)
  test_predictions  <- cbind(testingSetId,pred_test_bags)
  colnames(test_predictions)<-c("id","loss")
  #  head(val_predictions); head(val_predictions$pred_loss)
  
  cat("CV Fold-", i, " ", metric, ": ", score(val_predictions$loss, val_predictions$pred_loss, metric), "\n", sep = "")
  cat("CV score calculation Processing\n")
  val_predictions$loss <- NULL
  colnames(val_predictions)<-c("id","loss")
  
  if(i == 1)
  {
    write.csv(val_predictions, paste(root_directory, "/submissions/prav.rf20.fold1.csv", sep=''), row.names=FALSE, quote = FALSE)
    write.csv(test_predictions,paste(root_directory, "./submissions/prav.rf20.fold1-test.csv", sep=''), row.names=FALSE, quote = FALSE)
    
    
  }
  if(i == 2)
  {
    write.csv(val_predictions, paste(root_directory, "/submissions/prav.rf20.fold2.csv", sep=''), row.names=FALSE, quote = FALSE)
    write.csv(test_predictions,paste(root_directory, "./submissions/prav.rf20.fold2-test.csv", sep=''), row.names=FALSE, quote = FALSE)
    
    
  }
  if(i == 3)
  {
    write.csv(val_predictions, paste(root_directory, "/submissions/prav.rf20.fold3.csv", sep=''), row.names=FALSE, quote = FALSE)
    write.csv(test_predictions,paste(root_directory, "./submissions/prav.rf20.fold3-test.csv", sep=''), row.names=FALSE, quote = FALSE)
    
    
  }
  if(i == 4)
  {
    write.csv(val_predictions, paste(root_directory, "/submissions/prav.rf20.fold4.csv", sep=''), row.names=FALSE, quote = FALSE)
    write.csv(test_predictions,paste(root_directory, "./submissions/prav.rf20.fold4-test.csv", sep=''), row.names=FALSE, quote = FALSE)
    
    
  }
  if(i == 5)
  {
    write.csv(val_predictions, paste(root_directory, "/submissions/prav.rf20.fold5.csv", sep=''), row.names=FALSE, quote = FALSE)
    write.csv(test_predictions,paste(root_directory, "./submissions/prav.rf20.fold5-test.csv", sep=''), row.names=FALSE, quote = FALSE)
    
    
  }
  
}

##############################################################################################################


pred_fulltest_bags <- as.data.frame(rep(0, nrow(testingSet[,testfeature.names])))

set.seed(seed)
for (b in 1:bags) 
{
  cat(b ," - bag Processing\n")
  # seed = seed + b
  # set.seed(seed)
  cat(seed ," - Random Seed\n")
  cat("Full training Processing\n")
  FullModel <- h2o.randomForest(                              
                      training_frame   = trainingSet ,    
                      validation_frame = trainingSet,     
                      x=feature.names,              
                      y="loss",                 
                      mtries = mtry,
                      ntrees = ntrees,              
                      max_depth = maxdepth,               
                      seed=seed)   
  cat("testingSet prediction Processing\n")      
  pred_fulltest        <- h2o.predict(FullModel, testingSet[,testfeature.names])
  pred_fulltest_bags   <- pred_fulltest_bags + (exp(as.data.frame(pred_fulltest))-200)
}

pred_fulltest_bags     <- pred_fulltest_bags / bags      

testingSetId           <- as.data.frame(testingSet$id)
fulltest_predictions   <- cbind(testingSetId,pred_fulltest_bags)
colnames(fulltest_predictions)<-c("id","loss")

write.csv(fulltest_predictions,paste(root_directory, "./submissions/prav.rf20.full.csv", sep=''), row.names=FALSE, quote = FALSE)


#head(testfull_predictions)
####################################################################################################
h2o.scoreHistory(rfFulltrain)
impMatrix <- as.data.frame(h2o.varimp(rfFulltrain))
tail(impMatrix,350)

imp.features <- head(impMatrix[,1],747)

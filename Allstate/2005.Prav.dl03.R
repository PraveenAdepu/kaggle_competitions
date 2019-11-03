# rm(list=ls())
################################################################################################

# 20160921 - Baseline01 -- 2 features -- R Image

# Sys.time()
# load("Allstate_Baseline01_20161014.RData"); gc()
# Sys.time()
################################################################################################


require(h2o); 
## Create an H2O cloud 
h2o.init(
  nthreads=20,            
  max_mem_size = "50G")    
h2o.removeAll() 


trainingSet <- h2o.importFile("./input/train.csv", destination_frame = "trainingSet.hex")
testingSet  <- h2o.importFile("./input/test.csv", destination_frame = "testingSet.hex")

names(trainingSet)
CVindices5folds <- h2o.importFile("./CVSchema/Prav_CVindices_5folds.csv", destination_frame = "CVindices5folds.hex")
# 
trainingSet <- h2o.merge(trainingSet, CVindices5folds, all.x = TRUE)

feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("id","loss", "CVindices" ))])
testfeature.names <- names(testingSet[,-which(names(testingSet) %in% c("id" ))])



trainingSet$loss <- log(trainingSet$loss)

#head(trainingSet)

cv       = 5
bags     = 25
seed     = 2016

## for all remaining models, use same parameters 

cat(cv, "-fold Cross Validation\n", sep = "")

set.seed(seed)
for (i in 5:cv)
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
    seed = seed + b
    set.seed(seed)
    cat(seed ," - Random Seed\n")
    cat("X_build training Processing\n")
    Model <- h2o.deeplearning(
      
      training_frame   = X_build,	      # assign training frame (80%)
      validation_frame = X_val,	        # assign test frame (20%)
      x                = feature.names,	# assign x
      y                = "loss", 			  # assign y
      #nfolds=0,				                # Cross validation
      #stopping_rounds=5,
      overwrite_with_best_model=T,
      activation= "Rectifier",          #"Tanh",	# RectifierWithDropout
      distribution="huber",
      hidden=c(80,80,80),			          # select hidden topology (smaller is faster)
      epochs=20,				                # add more to increase accuracy
      #adaptive_rate=T,
      # standardize = TRUE,
      #score_training_samples=10000, 	  # faster training (can be set to 0)
      #stopping_rounds=40,			        # increase to run longer more accurate
      #stopping_metric="AUTO", 		      # "MSE","logloss","r2","misclassification","auto"
      #stopping_tolerance=0.0001,		    # stop if not getting better
      #max_runtime_secs=0,			        # dont run too long
      #                         overwrite_with_best_model=TRUE,   # use best model along the way
      #                         distribution="tweedie", 		    # gaussian, poisson and others
      #                         tweedie_power=1.5,			    # set tweedie power
      #                         input_dropout_ratio=0.4,		    # Input layer dropout ratio (can improve generalization, try 0.1 or 0.2)
      #                         hidden_dropout_ratios = c(0.2,0.1),
      #                         l1=0.01, 				            # L1 regularization (can add stability and improve generalization
      #                         l2=0,
      #                         use_all_factor_levels = TRUE,
      seed = seed)				        # L2 regularization (can add stability and improve generalization
    cat("X_val prediction Processing\n")
    pred_cv              <- h2o.predict(Model, X_val[,feature.names])
    pred_test            <- h2o.predict(Model, testingSet[,testfeature.names])
    pred_cv_bags         <- pred_cv_bags   + exp(as.data.frame(pred_cv))
    pred_test_bags       <- pred_test_bags + exp(as.data.frame(pred_test))
  }
  pred_cv_bags   <- pred_cv_bags / bags
  pred_test_bags <- pred_test_bags / bags
  
  ValSetId          <- as.data.frame(X_val$id)
  val_loss          <- exp(as.data.frame(X_val$loss))
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
  
  # if(i == 1)
  # {
  #   write.csv(val_predictions, paste(root_directory, "/submissions/prav.dl02.fold1.csv", sep=''), row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions,paste(root_directory, "./submissions/prav.dl02.fold1-test.csv", sep=''), row.names=FALSE, quote = FALSE)
  #   
  #   
  # }
  # if(i == 2)
  # {
  #   write.csv(val_predictions, paste(root_directory, "/submissions/prav.dl02.fold2.csv", sep=''), row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions,paste(root_directory, "./submissions/prav.dl02.fold2-test.csv", sep=''), row.names=FALSE, quote = FALSE)
  #   
  #   
  # }
  # if(i == 3)
  # {
  #   write.csv(val_predictions, paste(root_directory, "/submissions/prav.dl02.fold3.csv", sep=''), row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions,paste(root_directory, "./submissions/prav.dl02.fold3-test.csv", sep=''), row.names=FALSE, quote = FALSE)
  #   
  #   
  # }
  # if(i == 4)
  # {
  #   write.csv(val_predictions, paste(root_directory, "/submissions/prav.dl02.fold4.csv", sep=''), row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions,paste(root_directory, "./submissions/prav.dl02.fold4-test.csv", sep=''), row.names=FALSE, quote = FALSE)
  #   
  #   
  # }
  # if(i == 5)
  # {
  #   write.csv(val_predictions, paste(root_directory, "/submissions/prav.dl02.fold5.csv", sep=''), row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions,paste(root_directory, "./submissions/prav.dl02.fold5-test.csv", sep=''), row.names=FALSE, quote = FALSE)
  #   
  #   
  # }
  # 
}

# 10 bag : CV Fold-5 mae: 1148.392 
# 25 bag : CV Fold-5 mae: 1146.873 (100,100)

# 25 bag : CV Fold-5 mae: 1146.416 (80,80, 80)
#################################################################################################################################################
# Full Model training - Start
#################################################################################################################################################

pred_fulltest_bags <- as.data.frame(rep(0, nrow(testingSet[,testfeature.names])))

for (b in 1:bags) 
{
  cat(b ," - bag Processing\n")
  seed = seed + b
  set.seed(seed)
  cat(seed ," - Random Seed\n")
  cat("Full training Processing\n")
  FullModel <- h2o.deeplearning(
    
    training_frame   = trainingSet,	        # assign training frame (80%)
    validation_frame = trainingSet,	        # assign test frame (20%)
    x                = feature.names,	      # assign x
    y                = "loss", 				      # assign y
    #nfolds=0,				                      # Cross validation
    overwrite_with_best_model=T,
    activation= "Rectifier",                #"Tanh",	# RectifierWithDropout
    distribution="huber",
    hidden=c(80,80,80),			                # select hidden topology (smaller is faster)
    epochs=20,				                      # add more to increase accuracy
    #adaptive_rate=T,
    # standardize = TRUE,
    #score_training_samples=10000, 	        # faster training (can be set to 0)
    #stopping_rounds=40,			              # increase to run longer more accurate
    #stopping_metric="AUTO", 		            # "MSE","logloss","r2","misclassification","auto"
    #stopping_tolerance=0.0001,		          # stop if not getting better
    #max_runtime_secs=0,			        # dont run too long
    #                         overwrite_with_best_model=TRUE,       # use best model along the way
    #                         distribution="tweedie", 		        # gaussian, poisson and others
    #                         tweedie_power=1.5,			        # set tweedie power
    #                         input_dropout_ratio=0.4,		        # Input layer dropout ratio (can improve generalization, try 0.1 or 0.2)
    #                         hidden_dropout_ratios = c(0.2,0.1),
    #                         l1=0.01, 				                # L1 regularization (can add stability and improve generalization
    #                         l2=0,
    #                         use_all_factor_levels = TRUE,
    seed = seed)				            # L2 regularization (can add stability and improve generalization
  cat("testingSet prediction Processing\n")      
  pred_fulltest        <- h2o.predict(FullModel, testingSet[,testfeature.names])
  pred_fulltest_bags   <- pred_fulltest_bags + exp(as.data.frame(pred_fulltest))
}

pred_fulltest_bags     <- pred_fulltest_bags / bags      

testingSetId           <- as.data.frame(testingSet$id)
fulltest_predictions   <- cbind(testingSetId,pred_fulltest_bags)
colnames(fulltest_predictions)<-c("id","loss")

write.csv(fulltest_predictions,paste(root_directory, "./submissions/prav.dl02_03.full.csv", sep=''), row.names=FALSE, quote = FALSE)


#################################################################################################################################################
# Full Model training - End
#################################################################################################################################################





feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("Patient", "rowID" ,"Drug","dataset","Addicted","CVindices"))])

print(feature.names)

trainingSet[is.na(trainingSet)]   <- -1

trainingSet$Age <- as.numeric(trainingSet$Age)

cv = 5
nround.cv =  100
printeveryn = 20
seed = 2017


## for all remaining models, use same parameters 

param <- list(  "objective"        = "reg:linear",
                "booster"          = "gbtree",
                "tree_method"      = "exact",
                "eval_metric"      = "logloss",
                "nthread"          = 32,  #system CPU   
                "max_depth"        = 8,     
                "eta"              = 0.1, 
                "subsample"        = 0.7,  
                "colsample_bytree" = 0.7,  
                "min_child_weight" = 3     
                
)

validation.pred.columns <- c("Patient","rowID","CVindices","Addicted")
cat(cv, "-fold Cross Validation\n", sep = "")

set.seed(seed)
for (i in 1:cv)
  
{
  cat(i ,"fold Processing\n")
  
  cat("X_build fold Processing\n")
  X_build <- subset(trainingSet, CVindices != i)
  cat("X_val fold Processing\n")
  X_val   <- subset(trainingSet, CVindices == i)

  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=X_build$Addicted)
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=X_val$Addicted)
  watchlist <- list( val = dval,train = dtrain)
  
  cat("X_build training Processing\n")
  XGModel <- xgb.train(   params              = param,
                          data                = dtrain,
                          watchlist           = watchlist,
                          nrounds             = nround.cv ,
                          print_every_n       = printeveryn,
                          verbose             = TRUE, 
                          set.seed            = seed
  )
  
  X_val_pred    <- predict(XGModel, data.matrix(X_val[,feature.names]))
  X_val_pred        <- data.frame(X_val_pred)
  names(X_val_pred) <- "AddictionProbability"
  if(i == 1)
    
  {
    OOF_preds   <- cbind(X_val[, validation.pred.columns],X_val_pred)
  }
  else{
    
    cv_preds    <- cbind(X_val[, validation.pred.columns],X_val_pred)
    OOF_preds   <- rbind(OOF_preds,cv_preds)
  }
  
}

head(OOF_preds)

# Full training

# #########################################################################################################
# Full train
# #########################################################################################################

# dtrain<-xgb.DMatrix(data=data.matrix(trainingSet[, feature.names]),label=trainingSet$target)
# watchlist <- list( train = dtrain)
# 
# fulltrainnrounds = 1.2 * nround.cv
# 
# cat("Full TrainingSet training\n")
# XGModelFulltrain <- xgb.train(    params              = param,
#                                   #feval               = xgb.metric.log.mae,
#                                   data                = dtrain,
#                                   watchlist           = watchlist,
#                                   nrounds             = fulltrainnrounds,
#                                   print_every_n       = printeveryn,
#                                   verbose             = TRUE,
#                                   
#                                   #maximize            = TRUE,
#                                   set.seed            = seed
# )
# cat("Full Model prediction Processing\n")
# 
# predfull_test         <- predict(XGModelFulltrain, data.matrix(testingSet[,feature.names]))
# testfull_predictions  <- data.frame(id=testingSet$id, target = round(predfull_test))
# write.csv(testfull_predictions, './submissions/prav.xgb01.full.csv', row.names=FALSE, quote = FALSE)



# head(testfull_predictions)

############################################################################################
model = xgb.dump(XGModel, with_stats=TRUE)
names = dimnames(trainingSet[,feature.names])[[2]]
importance_matrix = xgb.importance( names , model = XGModel)
xgb.plot.importance(importance_matrix)
impMatrix <- as.data.frame(importance_matrix)

impMatrix


# #########################################################################################################



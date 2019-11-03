# rm(list=ls())
################################################################################################

# 20160921 - Baseline01 -- 2 features -- R Image

# Sys.time()
# load("Allstate_Baseline01_20161013.RData"); gc()
# Sys.time()
################################################################################################

names(trainingSet)

trainingSet <- as.data.table(trainingSet)

trainingSet[,cat80countRatio:= .N/length(trainingSet$cat80),by= c("cat80")]
trainingSet[,cat79countRatio:= .N/length(trainingSet$cat79),by= c("cat79")]
trainingSet[,cat12countRatio:= .N/length(trainingSet$cat12),by= c("cat12")]
trainingSet[,cat81countRatio:= .N/length(trainingSet$cat81),by= c("cat81")]
trainingSet[,cat103countRatio:= .N/length(trainingSet$cat103),by= c("cat103")]
trainingSet[,cat1countRatio:= .N/length(trainingSet$cat1),by= c("cat1")]

trainingSet[,cat8079countRatio:= .N/length(trainingSet$cat80),by= c("cat80","cat79")]
trainingSet[,cat8012countRatio:= .N/length(trainingSet$cat80),by= c("cat80","cat12")]
trainingSet[,cat8081countRatio:= .N/length(trainingSet$cat80),by= c("cat80","cat81")]
trainingSet[,cat80103countRatio:= .N/length(trainingSet$cat80),by= c("cat80","cat103")]
trainingSet[,cat801countRatio:= .N/length(trainingSet$cat80),by= c("cat80","cat1")]

trainingSet[,cat7912countRatio:= .N/length(trainingSet$cat80),by= c("cat79","cat12")]
trainingSet[,cat7981countRatio:= .N/length(trainingSet$cat80),by= c("cat79","cat81")]
trainingSet[,cat79103countRatio:= .N/length(trainingSet$cat80),by= c("cat79","cat103")]
trainingSet[,cat791countRatio:= .N/length(trainingSet$cat80),by= c("cat79","cat1")]

trainingSet[,cat1281countRatio:= .N/length(trainingSet$cat80),by= c("cat12","cat81")]
trainingSet[,cat12103countRatio:= .N/length(trainingSet$cat80),by= c("cat12","cat103")]
trainingSet[,cat121countRatio:= .N/length(trainingSet$cat80),by= c("cat12","cat1")]

trainingSet[,cat81103countRatio:= .N/length(trainingSet$cat80),by= c("cat81","cat103")]
trainingSet[,cat811countRatio:= .N/length(trainingSet$cat80),by= c("cat81","cat1")]

trainingSet[,cat1031countRatio:= .N/length(trainingSet$cat80),by= c("cat103","cat1")]

trainingSet[,cat807912countRatio:= .N/length(trainingSet$cat80),by= c("cat80","cat79","cat12")]
trainingSet[,cat807981countRatio:= .N/length(trainingSet$cat80),by= c("cat80","cat79","cat81")]
trainingSet[,cat8079103countRatio:= .N/length(trainingSet$cat80),by= c("cat80","cat79","cat103")]
trainingSet[,cat80791countRatio:= .N/length(trainingSet$cat80),by= c("cat80","cat79","cat1")]

trainingSet[,cat791281countRatio:= .N/length(trainingSet$cat80),by= c("cat79","cat12","cat81")]
trainingSet[,cat7912103countRatio:= .N/length(trainingSet$cat80),by= c("cat79","cat12","cat103")]
trainingSet[,cat79121countRatio:= .N/length(trainingSet$cat80),by= c("cat79","cat12","cat1")]
trainingSet[,cat1281103countRatio:= .N/length(trainingSet$cat80),by= c("cat12","cat81","cat103")]
trainingSet[,cat12811countRatio:= .N/length(trainingSet$cat80),by= c("cat12","cat81","cat1")]
trainingSet[,cat811031countRatio:= .N/length(trainingSet$cat80),by= c("cat81","cat103","cat1")]



trainingSet <- as.data.frame(trainingSet)


testingSet <- as.data.table(testingSet)

testingSet[,cat80countRatio:= .N/length(testingSet$cat80),by= c("cat80")]
testingSet[,cat79countRatio:= .N/length(testingSet$cat79),by= c("cat79")]
testingSet[,cat12countRatio:= .N/length(testingSet$cat12),by= c("cat12")]
testingSet[,cat81countRatio:= .N/length(testingSet$cat81),by= c("cat81")]
testingSet[,cat103countRatio:= .N/length(testingSet$cat103),by= c("cat103")]
testingSet[,cat1countRatio:= .N/length(testingSet$cat1),by= c("cat1")]

testingSet[,cat8079countRatio:= .N/length(testingSet$cat80),by= c("cat80","cat79")]
testingSet[,cat8012countRatio:= .N/length(testingSet$cat80),by= c("cat80","cat12")]
testingSet[,cat8081countRatio:= .N/length(testingSet$cat80),by= c("cat80","cat81")]
testingSet[,cat80103countRatio:= .N/length(testingSet$cat80),by= c("cat80","cat103")]
testingSet[,cat801countRatio:= .N/length(testingSet$cat80),by= c("cat80","cat1")]

testingSet[,cat7912countRatio:= .N/length(testingSet$cat80),by= c("cat79","cat12")]
testingSet[,cat7981countRatio:= .N/length(testingSet$cat80),by= c("cat79","cat81")]
testingSet[,cat79103countRatio:= .N/length(testingSet$cat80),by= c("cat79","cat103")]
testingSet[,cat791countRatio:= .N/length(testingSet$cat80),by= c("cat79","cat1")]

testingSet[,cat1281countRatio:= .N/length(testingSet$cat80),by= c("cat12","cat81")]
testingSet[,cat12103countRatio:= .N/length(testingSet$cat80),by= c("cat12","cat103")]
testingSet[,cat121countRatio:= .N/length(testingSet$cat80),by= c("cat12","cat1")]

testingSet[,cat81103countRatio:= .N/length(testingSet$cat80),by= c("cat81","cat103")]
testingSet[,cat811countRatio:= .N/length(testingSet$cat80),by= c("cat81","cat1")]

testingSet[,cat1031countRatio:= .N/length(testingSet$cat80),by= c("cat103","cat1")]

testingSet[,cat807912countRatio:= .N/length(testingSet$cat80),by= c("cat80","cat79","cat12")]
testingSet[,cat807981countRatio:= .N/length(testingSet$cat80),by= c("cat80","cat79","cat81")]
testingSet[,cat8079103countRatio:= .N/length(testingSet$cat80),by= c("cat80","cat79","cat103")]
testingSet[,cat80791countRatio:= .N/length(testingSet$cat80),by= c("cat80","cat79","cat1")]

testingSet[,cat791281countRatio:= .N/length(testingSet$cat80),by= c("cat79","cat12","cat81")]
testingSet[,cat7912103countRatio:= .N/length(testingSet$cat80),by= c("cat79","cat12","cat103")]
testingSet[,cat79121countRatio:= .N/length(testingSet$cat80),by= c("cat79","cat12","cat1")]
testingSet[,cat1281103countRatio:= .N/length(testingSet$cat80),by= c("cat12","cat81","cat103")]
testingSet[,cat12811countRatio:= .N/length(testingSet$cat80),by= c("cat12","cat81","cat1")]
testingSet[,cat811031countRatio:= .N/length(testingSet$cat80),by= c("cat81","cat103","cat1")]



testingSet <- as.data.frame(testingSet)

#names(testingSet)
feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("id","loss", "CVindices" ))])
testfeature.names <- names(testingSet[,-which(names(testingSet) %in% c("id" ))])


remove.features       <- union(features.bad.transformation, features.remove.totransform)
feature.names         <- setdiff(feature.names, remove.features)
testfeature.names     <- setdiff(testfeature.names, remove.features)

# ratio.features <- c("cat80","cat79","cat12","cat81","cat103","cat1")
# 
# feature.names         <- setdiff(feature.names, ratio.features)
# testfeature.names     <- setdiff(testfeature.names, ratio.features)

##################################################################################

##################################################################################

# order of columns are matching 
##################################################################################

# Check for all columns order, all columns orders are same after removing id columns


cv = 5
nround.cv =  1000 
printeveryn = 10
seed = 2016

## for all remaining models, use same parameters 

param <- list(  "objective"        = "reg:linear", 
                "booster"          = "gbtree",
                #"eval_metric"      = "auc",
                "nthread"          = 7,     
                "max_depth"        = 8,     
                "eta"              = 0.02, 
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
  
  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=log(X_build$loss))
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=log(X_val$loss))
  watchlist <- list( val = dval,train = dtrain)
      
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
   
#   cat("X_val prediction Processing\n")
#   pred_cv  <- predict(XGModel, data.matrix(X_val[,feature.names]))
#   val_predictions <- data.frame(id=X_val$id, loss = pred_cv)
#   
#   cat("CV TestingSet prediction Processing\n")
#   pred_test  <- predict(XGModel, data.matrix(testingSet[,testfeature.names]))
#   test_predictions <- data.frame(id=testingSet$id, loss = pred_test)
#   
#   if(i == 1)
#   {
#     write.csv(val_predictions,  'prav.xgb01.fold1.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.xgb01.fold1-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 2)
#   {
#     write.csv(val_predictions,  'prav.xgb01.fold2.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.xgb01.fold2-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 3)
#   {
#     write.csv(val_predictions,  'prav.xgb01.fold3.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.xgb01.fold3-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 4)
#   {
#     write.csv(val_predictions,  'prav.xgb01.fold4.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.xgb01.fold4-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 5)
#   {
#     write.csv(val_predictions,  'prav.xgb01.fold5.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.xgb01.fold5-test.csv', row.names=FALSE, quote = FALSE)
#   }
  
}

# Full training

dtrain<-xgb.DMatrix(data=data.matrix(trainingSet[, feature.names]),label=log(trainingSet$loss))
watchlist <- list( train = dtrain)

fulltrainnrounds = 1.1 * nround.cv

# #########################################################################################################
# Full train
# #########################################################################################################

cat("Full TrainingSet training\n")
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
cat("Full Model prediction Processing\n")

predfull_test         <- predict(XGModelFulltrain, data.matrix(testingSet[,testfeature.names]))
testfull_predictions  <- data.frame(id=testingSet$id, loss = exp(predfull_test))
write.csv(testfull_predictions, 'prav.xgb01.full.csv', row.names=FALSE, quote = FALSE)

# #########################################################################################################
# Bagging - Full train
# #########################################################################################################
bags = 5
ensemble <- rep(0, nrow(testingSet[,testfeature.names]))


for (b in 1:bags) {
  cat(b ," - bag Processing\n")
  seed = b + seed
  set.seed(seed)
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

ensemble <- ensemble + predfull_test

}

ensemble <- ensemble / bags
testfull_predictions  <- data.frame(id=testingSet$id, loss = exp(ensemble))
write.csv(testfull_predictions, './submissions/prav.xgb01.bags5.full.csv', row.names=FALSE, quote = FALSE)




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


# rm(list=ls())


train <- read_csv('./input/training30_train.csv')
test  <- read_csv('./input/training30_valid.csv')



#names(train)
feature.names     <- names(train[,-which(names(train) %in% c("display_id","clicked" ))])
testfeature.names <- names(test[,-which(names(test) %in% c("display_id","clicked"))])

sapply(train, class)

##################################################################################

##################################################################################

# order of columns are matching 
##################################################################################

cv          = 5
bags        = 1
nround.cv   = 100
printeveryn = 2
seed        = 2017

## for all remaining models, use same parameters 


# list("objective" = "binary:logistic",
#      "eta" = 0.10,
#      "max_depth" = 7,
#      "subsample" = 0.50,
#      "colsample_bytree" = 0.60,
#      #"colsample_bylevel" = 0.90,
#      "min_child_weight" = 0.1,
#      #"eval_metric" = "auc",
#      "eval_metric" = "logloss",
#      "nthread" = 25,
#      "num_parallel_tree" = 1,
#      "tree_method" = "exact", #exact approx
#      "silent" = 1)
sapply( dat, as.numeric )

train[, feature.names] <- sapply( train[, feature.names], as.numeric )

train[,feature.names] <- lapply(train[,feature.names,drop=FALSE],as.numeric)
test[,feature.names]  <- lapply(test[,feature.names,drop=FALSE],as.numeric)

dtrain <-xgb.DMatrix(data=data.matrix(train[, feature.names]),label=train$clicked)
dval   <-xgb.DMatrix(data=data.matrix(test[, feature.names]) ,label=test$clicked)
watchlist <- list( val = dval,train = dtrain)
gc()

XGModel <- xgb.train(   params              = param,
                        data                = dtrain,
                        watchlist           = watchlist,
                        nrounds             = nround.cv ,
                        print.every.n       = printeveryn,
                        verbose             = TRUE, 
                        #maximize            = TRUE,
                        set.seed            = seed
)

pred_cv    <- predict(XGModel, data.matrix(test[,feature.names])) #, ntreelimit = 2400
#test$Prob_clicked    <- pred_cv

MAP12 <- function( display_id, clicked, prob  ){
  map12 <- data.table( display_id=display_id, clicked=clicked, prob=prob  )
  map12[ is.na(prob) , prob := mean(prob, na.rm=T)    ]
  setorderv( map12, c("display_id","prob"), c(1,-1)  )
  map12[, count := 1:.N , by="display_id" ]
  return(mean( map12[, sum(clicked/count) , by="display_id" ]$V1 ))
}

# previous sub score LB = 0.684
cat("MAP12 score using Ash CV and Prav 25 raw features without hashing - ", MAP12(test$display_id, test$clicked, pred_cv))


###################################################################################################################################################

gc()

cv          = 5
bags        = 1
nround.cv   = 100
printeveryn = 2
seed        = 2017

param <- list(  "objective"        = "binary:logistic",
                #objective = logcoshobj, #fairobj #logcoshobj, #cauchyobj #,#'reg:linear',
                "booster"          = "gbtree",
                "eval_metric"      = "logloss",
                "nthread"          = 30,     
                "max_depth"        = 8,     
                "eta"              = 0.1, 
                "subsample"        = 0.7,  
                "colsample_bytree" = 0.8,
                #                 "alpha"            =  1,
                #                 "gamma"            =  1,
                "min_child_weight" = 3     
                
)
# 
# 
# param <- list(  "objective"        = "rank:pairwise",
#                 #objective = logcoshobj, #fairobj #logcoshobj, #cauchyobj #,#'reg:linear',
#                 "booster"          = "gbtree",
#                 "eval_metric"      = "map@12",
#                 "nthread"          = 30,     
#                 "max_depth"        = 8,     
#                 "eta"              = 0.1, 
#                 "subsample"        = 0.7,  
#                 "colsample_bytree" = 0.8,
#                 #                 "alpha"            =  1,
#                 #                 "gamma"            =  1,
#                 "min_child_weight" = 1    
#                 
# )


feature.names     <- names(train[,-which(names(train) %in% c("display_id","clicked", "event_entity_id" , "day","event_Entconf","entity_id","Entconf","LastCat_id","Lasttopic_id","weekday","minutes"))])
testfeature.names <- names(test[,-which(names(test) %in% c("display_id","clicked","event_entity_id" , "day","event_Entconf","entity_id","Entconf","LastCat_id","Lasttopic_id","weekday","minutes"))])

groupCols <- c("display_id","adCount" )
groupCount <- unique(subset(train,,groupCols))
groupCounttest <- unique(subset(test,,groupCols))
# [45]	val-map@12:0.612484	train-map@12:0.619260 
train[,feature.names] <- lapply(train[,feature.names,drop=FALSE],as.numeric)
test[,feature.names]  <- lapply(test[,feature.names,drop=FALSE],as.numeric)

dtrain <-xgb.DMatrix(data=data.matrix(train[, feature.names]),label=train$clicked, group = groupCount$adCount)
dval   <-xgb.DMatrix(data=data.matrix(test[, feature.names]) ,label=test$clicked, group = groupCounttest$adCount)
watchlist <- list( val = dval,train = dtrain)


XGModel <- xgb.train(   params              = param,
                        data                = dtrain,
                        watchlist           = watchlist,
                        nrounds             = nround.cv ,
                        print.every.n       = printeveryn,
                        print_every_n       = printeveryn,
                        verbose             = TRUE, 
                        #maximize            = TRUE,
                        set.seed            = seed
)





































cat(cv, "-fold Cross Validation\n", sep = "")

set.seed(seed)
for (i in 5:cv)
  
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
    pred_cv    <- predict(XGModel, data.matrix(X_val[,feature.names]), ntreelimit = 2400)
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
  #   write.csv(val_predictions,  'prav.xgb11.fold1.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.xgb11.fold1-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 2)
  # {
  #   write.csv(val_predictions,  'prav.xgb11.fold2.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.xgb11.fold2-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 3)
  # {
  #   write.csv(val_predictions,  'prav.xgb11.fold3.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.xgb11.fold3-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 4)
  # {
  #   write.csv(val_predictions,  'prav.xgb11.fold4.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.xgb11.fold4-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 5)
  # {
  #   write.csv(val_predictions,  'prav.xgb11.fold5.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.xgb11.fold5-test.csv', row.names=FALSE, quote = FALSE)
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
  seed = seed + b
  cat(seed , " - Random Seed\n ")
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
write.csv(testfull_predictions, './submissions/prav.xgb13.full.csv', row.names=FALSE, quote = FALSE)




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



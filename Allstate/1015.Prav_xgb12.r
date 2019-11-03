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
test$loss <- -100

train_test = rbind(train, test)

train_test <- as.data.table(train_test)
new.cat.raw <- c("cat80","cat87","cat57","cat12","cat79","cat10","cat7","cat89","cat2","cat72",
                 "cat81","cat11","cat1","cat13","cat9","cat3","cat16","cat90","cat23","cat36",
                 "cat73","cat103","cat40","cat28","cat111","cat6","cat76","cat50","cat5",
                 "cat4","cat14","cat38","cat24","cat82","cat25")


features_pair <- combn(new.cat.raw, 2, simplify = F)

for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  train_test[, eval(as.name(paste(f1, f2, sep = "_"))) :=
        paste0(train_test[, eval(as.name(f1))], train_test[, eval(as.name(f2))])]
}

features_pair <- combn(new.cat.raw, 3, simplify = F)

for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  f3 <- pair[3]
  
  train_test[, eval(as.name(paste(f1, f2, f3, sep = "_"))) :=
               paste0(train_test[, eval(as.name(f1))], train_test[, eval(as.name(f2))], train_test[, eval(as.name(f3))])]
}



LETTERS_AY <- LETTERS#[-length(LETTERS)]
# LETTERS702 <- c(LETTERS_AY, sapply(LETTERS_AY, function(x) paste0(x, LETTERS_AY)), "ZZ")
# LETTERS703 <- c(LETTERS_AY, sapply(LETTERS_AY, function(x) paste0(x,paste0(x, LETTERS_AY))), "ZZZ")
# LETTERS7022 <-  union(LETTERS702,LETTERS703)
LETTERS7022 <- union(
                      union(sort(LETTERS_AY),
                            sort(apply(expand.grid(LETTERS_AY, LETTERS_AY), 1, paste, collapse = "", sep = ","))
                            ) 
                    ,
                     sort(apply(expand.grid(LETTERS_AY, LETTERS_AY,LETTERS_AY), 1, paste, collapse = "", sep = ",")) 
                    )



train_test <- as.data.frame(train_test)

feature.names     <- names(train_test[,-which(names(train_test) %in% c("id","loss"))])


for (f in feature.names) {
  if (class(train_test[[f]])=="character") {
    levels <- intersect(LETTERS7022, unique(train_test[[f]])) # get'em ordered!
    labels <- match(levels, LETTERS7022)
    #train_test[[f]] <- factor(train_test[[f]], levels=levels) # uncomment this for one-hot
    train_test[[f]] <- as.integer(as.character(factor(train_test[[f]], levels=levels, labels = labels))) # comment this one away for one-hot
  }
}

gc()

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

# write.csv(trainingSet,  './input/train_RFsource02.csv', row.names=FALSE, quote = FALSE)
# write.csv(testingSet,  './input/test_RFsource02.csv', row.names=FALSE, quote = FALSE)

# hist(trainingSet$loss, col = "tomato")
# 
# skewness(trainingSet$loss)
# 
# skew.score <- function(c, x) (skewness(log(x + c)))^2
# 
# cval <- seq(0, 500, l = 101)
# skew <- cval * 0
# for (i in 1:length(cval)) 
#   skew[i] <- skewness(log(cval[i] + trainingSet$loss))
# plot(cval, skew, type = "l", ylab = expression(b[3](c)), xlab = expression(c))
# abline(h = 0, lty = 3)
# 
# best.c <- optimise(skew.score, c(0, 500), x = trainingSet$loss)$minimum
# best.c
# 
# ozone.transformed <- log(ozone + best.c)
# hist(ozone.transformed, col = "azure")
# 
# skewness(ozone.transformed)
# 
# qqnorm(log(trainingSet$loss+200))
# qqline(log(trainingSet$loss+200))
# 



#names(testingSet)
feature.names     <- imp.features impMatrix[1:2500,1] names(trainingSet[,-which(names(trainingSet) %in% c("id","loss", "CVindices" ))])
testfeature.names <- imp.features impMatrix[1:2500,1] names(testingSet[,-which(names(testingSet) %in% c("id" ))])

# nan.test <-  function (x) {
#   w <- sapply(x, function(x)all(is.na(x)))
#   if (any(w)) {
#     stop(paste("All NAN in columns", paste(which(w), collapse=", ")))
#   }
# }
# 
# nan.test(trainingSet)
# 
# head(trainingSet[,c(1:10)])
# 
# names(trainingSet[,c(728)])
# 
# names(trainingSet[,c(1:1000)])


##################################################################################

##################################################################################

# order of columns are matching 
##################################################################################
constant = 200

cv          = 5
bags        = 1
nround.cv   = 1600
printeveryn = 100
seed        = 2021

## for all remaining models, use same parameters 

param <- list(  "objective"        = fairobj,
                #objective = logcoshobj, #fairobj #logcoshobj, #cauchyobj #,#'reg:linear',
                "booster"          = "gbtree",
                #"eval_metric"      = "auc",
                "nthread"          = 27,     
                "max_depth"        = 12,     
                "eta"              = 0.03, 
                "subsample"        = 0.7,  
                "colsample_bytree" = 0.7,
                #                 "alpha"            =  1,
                #                 "gamma"            =  1,
                "min_child_weight" = 100     
                
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
    pred_cv    <- predict(XGModel, data.matrix(X_val[,feature.names]))
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
  
#   if(i == 1)
#   {
#     write.csv(val_predictions,  'prav.xgb11.fold1.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.xgb11.fold1-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 2)
#   {
#     write.csv(val_predictions,  'prav.xgb11.fold2.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.xgb11.fold2-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 3)
#   {
#     write.csv(val_predictions,  'prav.xgb11.fold3.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.xgb11.fold3-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 4)
#   {
#     write.csv(val_predictions,  'prav.xgb11.fold4.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.xgb11.fold4-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 5)
#   {
#     write.csv(val_predictions,  'prav.xgb11.fold5.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.xgb11.fold5-test.csv', row.names=FALSE, quote = FALSE)
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
  
  fulltest_ensemble <- fulltest_ensemble + (exp(predfull_test) - constant)
  
}

fulltest_ensemble <- fulltest_ensemble / bags
testfull_predictions  <- data.frame(id=testingSet$id, loss = fulltest_ensemble)
write.csv(testfull_predictions, './submissions/prav.xgb11.full.csv', row.names=FALSE, quote = FALSE)




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
# impMatrix[1:2500,1]
# impMatrix[,1]
ImpFeature <- impMatrix[,1]
write.csv(impMatrix, './submissions/prav.xgb.ImpMatrix.csv', row.names=FALSE, quote = FALSE)

# #########################################################################################################



###############################################################################################################################
# Sys.time()
# save.image(file = "DSM2017_03.RData" , safe = TRUE)
Sys.time()
load("DSM2017_12.RData")
Sys.time()
###############################################################################################################################


# all_data_full <- subset(all_data_full, IsDeferredScript == 0) # 59450785

all_data_full <- as.data.table(all_data_full) # 64025564
all_data_full <- all_data_full[ order(all_data_full$Patient_ID, all_data_full$DispenseDate , decreasing = FALSE ),]

cols = c("Drug_ID","ChronicIllness","RepeatsLeft_Qty") #,"DispenseDate"

#names(all_data_full)

anscols = paste("lag1", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 1, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag2", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 2, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag3", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 3, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag4", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 4, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag5", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 5, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag6", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 6, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag7", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 7, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag8", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 8, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag9", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 9, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag10", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 10, NA, "lag"), .SDcols=cols, by=Patient_ID]

####################################################################################################################

anscols = paste("lag11", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 11, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag12", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 12, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag13", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 13, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag14", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 14, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag15", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 15, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag16", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 16, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag17", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 17, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag18", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 18, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag19", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 19, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag20", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 20, NA, "lag"), .SDcols=cols, by=Patient_ID]

##################################################################################################################

####################################################################################################################

anscols = paste("lag21", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 21, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag22", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 22, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag23", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 23, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag24", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 24, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag25", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 25, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag26", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 26, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag27", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 27, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag28", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 28, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag29", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 29, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag30", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 30, NA, "lag"), .SDcols=cols, by=Patient_ID]

##################################################################################################################

####################################################################################################################

anscols = paste("lag31", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 31, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag32", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 32, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag33", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 33, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag34", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 34, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag35", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 35, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag36", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 36, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag37", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 37, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag38", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 38, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag39", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 39, NA, "lag"), .SDcols=cols, by=Patient_ID]

anscols = paste("lag40", cols, sep="_")
all_data_full[, (anscols) := shift(.SD, 40, NA, "lag"), .SDcols=cols, by=Patient_ID]

##################################################################################################################


all_data_full_build   <- subset(all_data_full, year(DispenseDate) <= 2015)

features <- grep("lag", names(all_data_full_build), value=TRUE)


all_data_full_build <- as.data.table(all_data_full_build)

all_data_full_build <- all_data_full_build[ order(all_data_full_build$Patient_ID, all_data_full_build$DispenseDate , decreasing = TRUE ),]

all_data_full_build[, OrderRank := 1:.N, by = c("Patient_ID")]
all_data_full_build <- as.data.frame(all_data_full_build)
build_set <- subset(all_data_full_build, OrderRank == 1)


##################################################################################################################
OriginalFeatures <- c("Patient_ID","Drug_ID","ChronicIllness", "DispenseDate","year_of_birth","gender")

model.feature <- union(OriginalFeatures ,features)

build_set <- as.data.frame(build_set)
build_set$gender <- as.integer(as.factor(build_set$gender))
build_set[is.na(build_set)]  <- 0
##################################################################################################################

features_00  <- read_csv("./input/Prav_FE_10.csv") 
# build_set <- left_join(build_set, features_00, by = "Patient_ID")


features_20  <- read_csv("./input/Prav_FE_20.csv")
build_set    <- left_join(build_set, features_20, by = "Patient_ID")

# features_21  <- read_csv("./input/Prav_FE_21.csv")
# build_set    <- left_join(build_set, features_21, by = "Patient_ID")
# 
# features_22  <- read_csv("./input/Prav_FE_22.csv")
# build_set    <- left_join(build_set, features_22, by = "Patient_ID")
# 
# features_24  <- read_csv("./input/Prav_FE_24.csv")
# build_set    <- left_join(build_set, features_24, by = "Patient_ID")

features_25  <- read_csv("./input/Prav_FE_25.csv") 
build_set <- left_join(build_set, features_25, by = "Patient_ID")

features_26  <- read_csv("./input/Prav_FE_26.csv") 
build_set <- left_join(build_set, features_26, by = "Patient_ID")

trainingSet <- subset(build_set, Patient_ID < 279201 )
testingSet  <- subset(build_set, Patient_ID >= 279201 )


##################################################################################################################
CVSchema  <- read_csv("./CVSchema/Prav_CVindices_5folds_10.csv") 

names(CVSchema)

trainingSet <- left_join(trainingSet, CVSchema, by = "Patient_ID")

########################################################################################################################

feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("Patient_ID", "DispenseDate" ,"DiabetesDispense", "CVindices","X13"))])


cv          = 5
bags        = 5
nround.cv   = 610
printeveryn = 100
seed        = 2017

param <- list(  "objective"        = "reg:linear", 
                "booster"          = "gbtree",
                "eval_metric"      = "auc",
                "nthread"          = 25,     
                "max_depth"        = 8,     
                "eta"              = 0.01, 
                "subsample"        = 0.7,  
                "colsample_bytree" = 0.7,  
                "min_child_weight" = 1     
                
)

set.seed(seed)
for (i in 1:cv)
  
{
  cat(i ,"fold Processing\n")
  seed        = 2017
  cat("X_build fold Processing\n")
  X_build <- subset(trainingSet, CVindices != i, select = -c( CVindices))
  cat("X_val fold Processing\n")
  X_val   <- subset(trainingSet, CVindices == i, select = -c( CVindices)) 
  
  val_ensemble <- rep(0, nrow(X_val[,feature.names]))
  test_ensemble <- rep(0, nrow(testingSet[,feature.names]))
  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=X_build$DiabetesDispense)
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=X_val$DiabetesDispense)
  watchlist <- list( val = dval,train = dtrain)
  for (b in 1:bags) {
    seed = seed + b
    cat(seed, "- seed")
    cat(b ," - bag Processing\n")
    cat("X_build training Processing\n")
    XGModel <- xgb.train(   params              = param, 
                            data                = dtrain,
                            watchlist           = watchlist,
                            nrounds             = nround.cv ,
                            print_every_n       = printeveryn,
                            verbose             = TRUE, 
                            maximize            = TRUE,
                            set.seed            = seed
    )
    cat("X_val prediction Processing\n")
    pred_cv          <- predict(XGModel, data.matrix(X_val[,feature.names]))
    val_ensemble     <- val_ensemble + pred_cv
    
    
    cat("CV TestingSet prediction Processing\n")
    pred_test         <- predict(XGModel, data.matrix(testingSet[,feature.names]))
    test_ensemble     <- test_ensemble + pred_test
  }
  val_ensemble     <- val_ensemble / bags
  test_ensemble     <- test_ensemble / bags
  
  val_predictions <- data.frame(Patient_ID=X_val$Patient_ID, Diabetes = val_ensemble)
  test_predictions <- data.frame(Patient_ID=testingSet$Patient_ID, Diabetes = test_ensemble)
  
  if(i == 1)
  {
    write.csv(val_predictions,  './submissions/prav.xgb32.fold1.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb32.fold1-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 2)
  {
    write.csv(val_predictions,  './submissions/prav.xgb32.fold2.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb32.fold2-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 3)
  {
    write.csv(val_predictions,  './submissions/prav.xgb32.fold3.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb32.fold3-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 4)
  {
    write.csv(val_predictions,  './submissions/prav.xgb32.fold4.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb32.fold4-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 5)
  {
    write.csv(val_predictions,  './submissions/prav.xgb32.fold5.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb32.fold5-test.csv', row.names=FALSE, quote = FALSE)
  }
  
}

##########################################################################################################
# Full training

dtrain<-xgb.DMatrix(data=data.matrix(trainingSet[, feature.names]),label=trainingSet$DiabetesDispense)
watchlist <- list( train = dtrain)

fulltrainnrounds = 1.2 * nround.cv


# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

fulltest_ensemble <- rep(0, nrow(testingSet[,feature.names]))

seed        = 2017
for (b in 1:bags) {
  seed = seed + b
  cat(seed, "- seed")
  cat(b ," - bag Processing\n")
  cat("Bagging Full TrainingSet training\n")
  XGModelFulltrain <- xgb.train(    params              = param, 
                                    data                = dtrain,
                                    watchlist           = watchlist,
                                    nrounds             = fulltrainnrounds,
                                    print_every_n       = printeveryn,
                                    verbose             = TRUE, 
                                    maximize            = TRUE,
                                    set.seed            = seed
  )
  cat("Bagging Full Model prediction Processing\n")
  
  predfull_test         <- predict(XGModelFulltrain, data.matrix(testingSet[,feature.names]))
  fulltest_ensemble     <- fulltest_ensemble + predfull_test
  
}

fulltest_ensemble     <- fulltest_ensemble / bags
testfull_predictions  <- data.frame(Patient_ID=testingSet$Patient_ID, Diabetes = fulltest_ensemble)
write.csv(testfull_predictions, './submissions/prav.xgb32.full.csv', row.names=FALSE, quote = FALSE)


# #########################################################################################################
# Bagging - Full train
# #########################################################################################################
# #########################################################################################################
model = xgb.dump(XGModelFulltrain, with.stats=TRUE)

names = dimnames(trainingSet[,feature.names])[[2]]
importance_matrix = xgb.importance( names , model = XGModelFulltrain)
#xgb.plot.importance(importance_matrix)
impMatrix <- as.data.frame(importance_matrix)
# impMatrix[1:12,1]
# impMatrix[,1]

tail(impMatrix,25)
ImpFeature <- impMatrix[,1]

featureList <- feature.names
featureVector <- c() 
for (i in 1:length(featureList)) { 
  featureVector[i] <- paste(i-1, featureList[i], "q", sep="\t") 
}
write.table(featureVector, "fmap.txt", row.names=FALSE, quote = FALSE, col.names = FALSE)
xgb.dump(model = XGModelFulltrain, fname = 'xgb.dump', fmap = "fmap.txt", with.stats = TRUE)


# #########################################################################################################

##########################################################################################################

BenchMark <- read.csv("./submissions/Prav_benchmark_01.csv")
val_sub <- read.csv("./submissions/prav.xgb32.full.csv")

MergeSub <- left_join(BenchMark, val_sub, by ="Patient_ID")

head(MergeSub,10)
tail(MergeSub,10)

summary(MergeSub$Diabetes.y)

MergeSub <- arrange(MergeSub,Diabetes.y)
MergeSub$Diabetes_rank <-  rank(MergeSub$Diabetes.y)
MergeSub$Diabetes <- MergeSub$Diabetes_rank / max(MergeSub$Diabetes_rank)

Final_sub <- MergeSub[,c("Patient_ID","Diabetes")]

write.csv(Final_sub,  './submissions/Prav_xgb32_rank.csv', row.names=FALSE, quote = FALSE)
#########################################################################################################



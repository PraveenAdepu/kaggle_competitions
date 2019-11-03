####################################################################################
# L2 Stacking
# load("build_set.Rda")

#############################################################################################################################


trainingSet <- subset(build_set, Patient_ID < 279201 )
testingSet  <- subset(build_set, Patient_ID >= 279201 )

#############################################################################################################################
CVSchema  <- read_csv("./CVSchema/Prav_CVindices_5folds_10.csv") 

names(CVSchema)

trainingSet <- left_join(trainingSet, CVSchema, by = "Patient_ID")

#############################################################################################################################


train.cols <- c('Patient_ID',"DiabetesDispense","CVindices")
train <- trainingSet[, train.cols]

test.cols <- c('Patient_ID')
test <- as.data.frame(testingSet[, test.cols])
names(test) <- "Patient_ID"
head(test)

rm(build_set,CVSchema,trainingSet,testingSet); gc()
#############################################################################################################################

xgb12.fold1 <- read_csv("./submissions/prav.xgb12.fold1.csv")
xgb12.fold2 <- read_csv("./submissions/prav.xgb12.fold2.csv")
xgb12.fold3 <- read_csv("./submissions/prav.xgb12.fold3.csv")
xgb12.fold4 <- read_csv("./submissions/prav.xgb12.fold4.csv")
xgb12.fold5 <- read_csv("./submissions/prav.xgb12.fold5.csv")
xgb12.full  <- read_csv("./submissions/prav.xgb12.full.csv")

xgb12 <- rbind(xgb12.fold1,xgb12.fold2,xgb12.fold3,xgb12.fold4,xgb12.fold5)
names(xgb12)
colnames(xgb12)<-c("Patient_ID","xgb12Diabetes")
names(xgb12)

names(xgb12.full)
colnames(xgb12.full)<-c("Patient_ID","xgb12Diabetes")
names(xgb12.full)
rm(xgb12.fold1,xgb12.fold2,xgb12.fold3,xgb12.fold4,xgb12.fold5); gc()

et01.fold1 <- read_csv("./submissions/prav.et01.fold1.csv")
et01.fold2 <- read_csv("./submissions/prav.et01.fold2.csv")
et01.fold3 <- read_csv("./submissions/prav.et01.fold3.csv")
et01.fold4 <- read_csv("./submissions/prav.et01.fold4.csv")
et01.fold5 <- read_csv("./submissions/prav.et01.fold5.csv")
et01.full  <- read_csv("./submissions/prav.et01.full.csv")

et01 <- rbind(et01.fold1,et01.fold2,et01.fold3,et01.fold4,et01.fold5)
names(et01)
colnames(et01)<-c("Patient_ID","et01Diabetes")
names(et01)

names(et01.full)
colnames(et01.full)<-c("Patient_ID","et01Diabetes")
names(et01.full)
rm(et01.fold1,et01.fold2,et01.fold3,et01.fold4,et01.fold5); gc()

rf01.fold1 <- read_csv("./submissions/prav.rf01.fold1.csv")
rf01.fold2 <- read_csv("./submissions/prav.rf01.fold2.csv")
rf01.fold3 <- read_csv("./submissions/prav.rf01.fold3.csv")
rf01.fold4 <- read_csv("./submissions/prav.rf01.fold4.csv")
rf01.fold5 <- read_csv("./submissions/prav.rf01.fold5.csv")
rf01.full  <- read_csv("./submissions/prav.rf01.full.csv")

rf01 <- rbind(rf01.fold1,rf01.fold2,rf01.fold3,rf01.fold4,rf01.fold5)
names(rf01)
colnames(rf01)<-c("Patient_ID","rf01Diabetes")
names(rf01)

names(rf01.full)
colnames(rf01.full)<-c("Patient_ID","rf01Diabetes")
names(rf01.full)
rm(rf01.fold1,rf01.fold2,rf01.fold3,rf01.fold4,rf01.fold5); gc()

lgbm01.fold1 <- read_csv("./submissions/prav.lgbm01.fold1.csv")
lgbm01.fold2 <- read_csv("./submissions/prav.lgbm01.fold2.csv")
lgbm01.fold3 <- read_csv("./submissions/prav.lgbm01.fold3.csv")
lgbm01.fold4 <- read_csv("./submissions/prav.lgbm01.fold4.csv")
lgbm01.fold5 <- read_csv("./submissions/prav.lgbm01.fold5.csv")
lgbm01.full  <- read_csv("./submissions/prav.lgbm01.full.csv")

lgbm01 <- rbind(lgbm01.fold1,lgbm01.fold2,lgbm01.fold3,lgbm01.fold4,lgbm01.fold5)
names(lgbm01)
colnames(lgbm01)<-c("Patient_ID","lgbm01Diabetes")
names(lgbm01)

names(lgbm01.full)
colnames(lgbm01.full)<-c("Patient_ID","lgbm01Diabetes")
names(lgbm01.full)
rm(lgbm01.fold1,lgbm01.fold2,lgbm01.fold3,lgbm01.fold4,lgbm01.fold5); gc()

rf20.fold1 <- read_csv("./submissions/prav.rf20.fold1.csv")
rf20.fold2 <- read_csv("./submissions/prav.rf20.fold2.csv")
rf20.fold3 <- read_csv("./submissions/prav.rf20.fold3.csv")
rf20.fold4 <- read_csv("./submissions/prav.rf20.fold4.csv")
rf20.fold5 <- read_csv("./submissions/prav.rf20.fold5.csv")
rf20.full  <- read_csv("./submissions/prav.rf20.full.csv")

rf20 <- rbind(rf20.fold1,rf20.fold2,rf20.fold3,rf20.fold4,rf20.fold5)
names(rf20)
colnames(rf20)<-c("Patient_ID","rf20Diabetes")
names(rf20)

names(rf20.full)
colnames(rf20.full)<-c("Patient_ID","rf20Diabetes")
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
colnames(et20)<-c("Patient_ID","et20Diabetes")
names(et20)

names(et20.full)
colnames(et20.full)<-c("Patient_ID","et20Diabetes")
names(et20.full)
rm(et20.fold1,et20.fold2,et20.fold3,et20.fold4,et20.fold5); gc()

lr20.fold1 <- read_csv("./submissions/prav.lr20.fold1.csv")
lr20.fold2 <- read_csv("./submissions/prav.lr20.fold2.csv")
lr20.fold3 <- read_csv("./submissions/prav.lr20.fold3.csv")
lr20.fold4 <- read_csv("./submissions/prav.lr20.fold4.csv")
lr20.fold5 <- read_csv("./submissions/prav.lr20.fold5.csv")
lr20.full  <- read_csv("./submissions/prav.lr20.full.csv")

lr20 <- rbind(lr20.fold1,lr20.fold2,lr20.fold3,lr20.fold4,lr20.fold5)
names(lr20)
colnames(lr20)<-c("Patient_ID","lr20Diabetes")
names(lr20)

names(lr20.full)
colnames(lr20.full)<-c("Patient_ID","lr20Diabetes")
names(lr20.full)
rm(lr20.fold1,lr20.fold2,lr20.fold3,lr20.fold4,lr20.fold5); gc()

nn20.fold1 <- read_csv("./submissions/prav.nn20.fold1.csv")
nn20.fold2 <- read_csv("./submissions/prav.nn20.fold2.csv")
nn20.fold3 <- read_csv("./submissions/prav.nn20.fold3.csv")
nn20.fold4 <- read_csv("./submissions/prav.nn20.fold4.csv")
nn20.fold5 <- read_csv("./submissions/prav.nn20.fold5.csv")
nn20.full  <- read_csv("./submissions/prav.nn20.full.csv")

nn20 <- rbind(nn20.fold1,nn20.fold2,nn20.fold3,nn20.fold4,nn20.fold5)
names(nn20)
colnames(nn20)<-c("Patient_ID","nn20Diabetes")
names(nn20)

names(nn20.full)
colnames(nn20.full)<-c("Patient_ID","nn20Diabetes")
names(nn20.full)
rm(nn20.fold1,nn20.fold2,nn20.fold3,nn20.fold4,nn20.fold5); gc()

xgb10.fold1 <- read_csv("./submissions/prav.xgb10.fold1.csv")
xgb10.fold2 <- read_csv("./submissions/prav.xgb10.fold2.csv")
xgb10.fold3 <- read_csv("./submissions/prav.xgb10.fold3.csv")
xgb10.fold4 <- read_csv("./submissions/prav.xgb10.fold4.csv")
xgb10.fold5 <- read_csv("./submissions/prav.xgb10.fold5.csv")
xgb10.full  <- read_csv("./submissions/prav.xgb10.full.csv")

xgb10 <- rbind(xgb10.fold1,xgb10.fold2,xgb10.fold3,xgb10.fold4,xgb10.fold5)
names(xgb10)
colnames(xgb10)<-c("Patient_ID","xgb10Diabetes")
names(xgb10)

names(xgb10.full)
colnames(xgb10.full)<-c("Patient_ID","xgb10Diabetes")
names(xgb10.full)
rm(xgb10.fold1,xgb10.fold2,xgb10.fold3,xgb10.fold4,xgb10.fold5); gc()

xgb11.fold1 <- read_csv("./submissions/prav.xgb11.fold1.csv")
xgb11.fold2 <- read_csv("./submissions/prav.xgb11.fold2.csv")
xgb11.fold3 <- read_csv("./submissions/prav.xgb11.fold3.csv")
xgb11.fold4 <- read_csv("./submissions/prav.xgb11.fold4.csv")
xgb11.fold5 <- read_csv("./submissions/prav.xgb11.fold5.csv")
xgb11.full  <- read_csv("./submissions/prav.xgb11.full.csv")

xgb11 <- rbind(xgb11.fold1,xgb11.fold2,xgb11.fold3,xgb11.fold4,xgb11.fold5)
names(xgb11)
colnames(xgb11)<-c("Patient_ID","xgb11Diabetes")
names(xgb11)

names(xgb11.full)
colnames(xgb11.full)<-c("Patient_ID","xgb11Diabetes")
names(xgb11.full)
rm(xgb11.fold1,xgb11.fold2,xgb11.fold3,xgb11.fold4,xgb11.fold5); gc()

xgb21.fold1 <- read_csv("./submissions/prav.xgb21.fold1.csv")
xgb21.fold2 <- read_csv("./submissions/prav.xgb21.fold2.csv")
xgb21.fold3 <- read_csv("./submissions/prav.xgb21.fold3.csv")
xgb21.fold4 <- read_csv("./submissions/prav.xgb21.fold4.csv")
xgb21.fold5 <- read_csv("./submissions/prav.xgb21.fold5.csv")
xgb21.full  <- read_csv("./submissions/prav.xgb21.full.csv")

xgb21 <- rbind(xgb21.fold1,xgb21.fold2,xgb21.fold3,xgb21.fold4,xgb21.fold5)
names(xgb21)
colnames(xgb21)<-c("Patient_ID","xgb21Diabetes")
names(xgb21)

names(xgb21.full)
colnames(xgb21.full)<-c("Patient_ID","xgb21Diabetes")
names(xgb21.full)
rm(xgb21.fold1,xgb21.fold2,xgb21.fold3,xgb21.fold4,xgb21.fold5); gc()

lgbm02.fold1 <- read_csv("./submissions/prav.lgbm02.fold1.csv")
lgbm02.fold2 <- read_csv("./submissions/prav.lgbm02.fold2.csv")
lgbm02.fold3 <- read_csv("./submissions/prav.lgbm02.fold3.csv")
lgbm02.fold4 <- read_csv("./submissions/prav.lgbm02.fold4.csv")
lgbm02.fold5 <- read_csv("./submissions/prav.lgbm02.fold5.csv")
lgbm02.full  <- read_csv("./submissions/prav.lgbm02.full.csv")

lgbm02 <- rbind(lgbm02.fold1,lgbm02.fold2,lgbm02.fold3,lgbm02.fold4,lgbm02.fold5)
names(lgbm02)
colnames(lgbm02)<-c("Patient_ID","lgbm02Diabetes")
names(lgbm02)

names(lgbm02.full)
colnames(lgbm02.full)<-c("Patient_ID","lgbm02Diabetes")
names(lgbm02.full)
rm(lgbm02.fold1,lgbm02.fold2,lgbm02.fold3,lgbm02.fold4,lgbm02.fold5); gc()

et21.fold1 <- read_csv("./submissions/prav.et21.fold1.csv")
et21.fold2 <- read_csv("./submissions/prav.et21.fold2.csv")
et21.fold3 <- read_csv("./submissions/prav.et21.fold3.csv")
et21.fold4 <- read_csv("./submissions/prav.et21.fold4.csv")
et21.fold5 <- read_csv("./submissions/prav.et21.fold5.csv")
et21.full  <- read_csv("./submissions/prav.et21.full.csv")

et21 <- rbind(et21.fold1,et21.fold2,et21.fold3,et21.fold4,et21.fold5)
names(et21)
colnames(et21)<-c("Patient_ID","et21Diabetes")
names(et21)

names(et21.full)
colnames(et21.full)<-c("Patient_ID","et21Diabetes")
names(et21.full)
rm(et21.fold1,et21.fold2,et21.fold3,et21.fold4,et21.fold5); gc()

rf21.fold1 <- read_csv("./submissions/prav.rf21.fold1.csv")
rf21.fold2 <- read_csv("./submissions/prav.rf21.fold2.csv")
rf21.fold3 <- read_csv("./submissions/prav.rf21.fold3.csv")
rf21.fold4 <- read_csv("./submissions/prav.rf21.fold4.csv")
rf21.fold5 <- read_csv("./submissions/prav.rf21.fold5.csv")
rf21.full  <- read_csv("./submissions/prav.rf21.full.csv")

rf21 <- rbind(rf21.fold1,rf21.fold2,rf21.fold3,rf21.fold4,rf21.fold5)
names(rf21)
colnames(rf21)<-c("Patient_ID","rf21Diabetes")
names(rf21)

names(rf21.full)
colnames(rf21.full)<-c("Patient_ID","rf21Diabetes")
names(rf21.full)
rm(rf21.fold1,rf21.fold2,rf21.fold3,rf21.fold4,rf21.fold5); gc()

nn21.fold1 <- read_csv("./submissions/prav.nn21.fold1.csv")
nn21.fold2 <- read_csv("./submissions/prav.nn21.fold2.csv")
nn21.fold3 <- read_csv("./submissions/prav.nn21.fold3.csv")
nn21.fold4 <- read_csv("./submissions/prav.nn21.fold4.csv")
nn21.fold5 <- read_csv("./submissions/prav.nn21.fold5.csv")
nn21.full  <- read_csv("./submissions/prav.nn21.full.csv")

nn21 <- rbind(nn21.fold1,nn21.fold2,nn21.fold3,nn21.fold4,nn21.fold5)
names(nn21)
colnames(nn21)<-c("Patient_ID","nn21Diabetes")
names(nn21)

names(nn21.full)
colnames(nn21.full)<-c("Patient_ID","nn21Diabetes")
names(nn21.full)
rm(nn21.fold1,nn21.fold2,nn21.fold3,nn21.fold4,nn21.fold5); gc()

########################################################################################
train <- left_join(train, xgb12,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , xgb12.full, by="Patient_ID", all.X = TRUE)
rm(xgb12,xgb12.full); gc()

train <- left_join(train, et01,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , et01.full, by="Patient_ID", all.X = TRUE)
rm(et01,et01.full); gc()

train <- left_join(train, rf01,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , rf01.full, by="Patient_ID", all.X = TRUE)
rm(rf01,rf01.full); gc()


train <- left_join(train, lgbm01,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , lgbm01.full, by="Patient_ID", all.X = TRUE)
rm(lgbm01,lgbm01.full); gc()

train <- left_join(train, rf20,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , rf20.full, by="Patient_ID", all.X = TRUE)
rm(rf20,rf20.full); gc()

train <- left_join(train, et20,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , et20.full, by="Patient_ID", all.X = TRUE)
rm(et20,et20.full); gc()


train <- left_join(train, lr20,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , lr20.full, by="Patient_ID", all.X = TRUE)
rm(lr20,lr20.full); gc()

train <- left_join(train, nn20,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , nn20.full, by="Patient_ID", all.X = TRUE)
rm(nn20,nn20.full); gc()


train <- left_join(train, xgb10,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , xgb10.full, by="Patient_ID", all.X = TRUE)
rm(xgb10,xgb10.full); gc()

train <- left_join(train, xgb11,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , xgb11.full, by="Patient_ID", all.X = TRUE)
rm(xgb11,xgb11.full); gc()

train <- left_join(train, xgb21,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , xgb21.full, by="Patient_ID", all.X = TRUE)
rm(xgb21,xgb21.full); gc()

train <- left_join(train, lgbm02,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , lgbm02.full, by="Patient_ID", all.X = TRUE)
rm(lgbm02,lgbm02.full); gc()

train <- left_join(train, et21,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , et21.full, by="Patient_ID", all.X = TRUE)
rm(et21,et21.full); gc()

train <- left_join(train, rf21,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , rf21.full, by="Patient_ID", all.X = TRUE)
rm(rf21,rf21.full); gc()

train <- left_join(train, nn21,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , nn21.full, by="Patient_ID", all.X = TRUE)
rm(nn21,nn21.full); gc()

########################################################################################
names(train)
names(test)

trainingSet <- train
testingSet  <- test


rm(train, test); gc()

feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("Patient_ID","DiabetesDispense", "CVindices","lr20Diabetes" ,"xgb11Diabetes" ))])


#######################################################################

cor(trainingSet[,feature.names],method = "pearson")
cor(trainingSet[,feature.names],method = "spearman")
cor(testingSet[,feature.names],method = "pearson")
cor(testingSet[,feature.names],method = "spearman")
#######################################################################
# Ensembling of high cor models
head(trainingSet)
head(testingSet)
##################################################################################

# order of columns are matching 
##################################################################################

# Check for all columns order, all columns orders are same after removing id columns


cv          = 5
bags        = 5
nround.cv   = 210
printeveryn = 100
seed        = 2017

param <- list(  "objective"        = "reg:linear", 
                "booster"          = "gbtree",
                "eval_metric"      = "auc",
                "nthread"          = 20,     
                "max_depth"        = 4,     
                "eta"              = 0.01, 
                "subsample"        = 0.7,  
                "colsample_bytree" = 0.7,  
                "min_child_weight" = 1     
                
)


set.seed(seed)
for (i in 1:cv)
  
{
  cat(i ,"fold Processing\n")
  
  cat("X_build fold Processing\n")
  X_build <- subset(trainingSet, CVindices != i, select = -c( CVindices))
  cat("X_val fold Processing\n")
  X_val   <- subset(trainingSet, CVindices == i, select = -c( CVindices)) 
  
  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=X_build$DiabetesDispense)
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=X_val$DiabetesDispense)
  watchlist <- list( val = dval,train = dtrain)
  pred_cv_bags   <- rep(0, nrow(X_val[, feature.names]))
  pred_test_bags <- rep(0, nrow(testingSet[,feature.names]))
  for (b in 1:bags) 
  {
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
    pred_cv  <- predict(XGModel, data.matrix(X_val[,feature.names]))
    
    
    cat("CV TestingSet prediction Processing\n")
    pred_test  <- predict(XGModel, data.matrix(testingSet[,feature.names]))
    
    pred_cv_bags   <- pred_cv_bags + pred_cv
    pred_test_bags <- pred_test_bags + pred_test
  }
  pred_cv_bags   <- pred_cv_bags / bags
  pred_test_bags <- pred_test_bags / bags
  cat("CV Fold-", i, " ", metric, ": ", score(X_val$DiabetesDispense, pred_cv_bags, metric), "\n", sep = "")
  val_predictions <- data.frame(Patient_ID=X_val$Patient_ID, Diabetes = pred_cv_bags)
  test_predictions <- data.frame(Patient_ID=testingSet$Patient_ID, Diabetes = pred_test_bags)
  
  # if(i == 1)
  # {
  #   write.csv(val_predictions,  './submissions/prav.L2_xgb01.fold1.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.L2_xgb01.fold1-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 2)
  # {
  #   write.csv(val_predictions,  './submissions/prav.L2_xgb01.fold2.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.L2_xgb01.fold2-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 3)
  # {
  #   write.csv(val_predictions,  './submissions/prav.L2_xgb01.fold3.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.L2_xgb01.fold3-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 4)
  # {
  #   write.csv(val_predictions,  './submissions/prav.L2_xgb01.fold4.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.L2_xgb01.fold4-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 5)
  # {
  #   write.csv(val_predictions,  './submissions/prav.L2_xgb01.fold5.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.L2_xgb01.fold5-test.csv', row.names=FALSE, quote = FALSE)
  # }
  
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
write.csv(testfull_predictions, './submissions/prav.L2_xgb03.full.csv', row.names=FALSE, quote = FALSE)


# #########################################################################################################
# Bagging - Full train
# #########################################################################################################
##########################################################################################################

BenchMark <- read.csv("./submissions/Prav_benchmark_01.csv")
val_sub <- read.csv("./submissions/prav.L2_xgb03.full.csv")

MergeSub <- left_join(BenchMark, val_sub, by ="Patient_ID")

head(MergeSub,10)
tail(MergeSub,10)

summary(MergeSub$Diabetes.y)

MergeSub <- arrange(MergeSub,Diabetes.y)
head(MergeSub,10)
tail(MergeSub,10)
MergeSub$Diabetes_rank <-  rank(MergeSub$Diabetes.y)
MergeSub$Diabetes <- MergeSub$Diabetes_rank / max(MergeSub$Diabetes_rank)

Final_sub <- MergeSub[,c("Patient_ID","Diabetes")]
head(Final_sub,10)
write.csv(Final_sub,  './submissions/prav.L2_xgb03_rank.csv', row.names=FALSE, quote = FALSE)
#########################################################################################################

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

# L2 ET

##########################################################################################################

options( java.parameters = "-Xmx50g" )
require("caret")
require("rJava")
require("extraTrees")



names(trainingSet)
col_idx <- grep("loss", names(trainingSet))
trainingSet <- trainingSet[, c(col_idx, (1:ncol(trainingSet))[-col_idx])]
names(trainingSet)

testingSet$loss <- 0
names(testingSet)
col_idx <- grep("loss", names(testingSet))
testingSet <- testingSet[, c(col_idx, (1:ncol(testingSet))[-col_idx])]
names(testingSet)

trainingSet$loss <- log(trainingSet$loss)

feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("id", "CVindices" ))])
testfeature.names <- names(testingSet[,-which(names(testingSet) %in% c("id" ))])

cv                 = 5
bags               = 2
seed               = 2018
#metric             = "auc"
Parammtry          = 2  # Regression 1/3 of variables
ParamnumThreads    = 7
Paramntree         = 200
ParamnumRandomCuts = 2


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
    
    
    pred_cv_bags   <- pred_cv_bags + exp(pred_cv)
    pred_test_bags <- pred_test_bags + exp(pred_test)
  }
  pred_cv_bags   <- pred_cv_bags / bags
  pred_test_bags <- pred_test_bags / bags
  
  cat("CV Fold-", i, " ", metric, ": ", score(exp(X_val$loss), pred_cv_bags, metric), "\n", sep = "")
  val_predictions <- data.frame(id=X_val$id, loss = pred_cv_bags)
  test_predictions <- data.frame(id=testingSet$id, loss = pred_test_bags)
  
#   if(i == 1)
#   {
#     write.csv(val_predictions,  'prav.et01.fold1.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.et01.fold1-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 2)
#   {
#     write.csv(val_predictions,  'prav.et01.fold2.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.et01.fold2-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 3)
#   {
#     write.csv(val_predictions,  'prav.et01.fold3.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.et01.fold3-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 4)
#   {
#     write.csv(val_predictions,  'prav.et01.fold4.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.et01.fold4-test.csv', row.names=FALSE, quote = FALSE)
#   }
#   if(i == 5)
#   {
#     write.csv(val_predictions,  'prav.et01.fold5.csv', row.names=FALSE, quote = FALSE)
#     write.csv(test_predictions, 'prav.et01.fold5-test.csv', row.names=FALSE, quote = FALSE)
#   }
  
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
  
  fulltest_ensemble <- fulltest_ensemble + exp(predfull_test)
  
}

fulltest_ensemble <- fulltest_ensemble / bags
testfull_predictions  <- data.frame(id=testingSet$id, loss = fulltest_ensemble)
write.csv(testfull_predictions, './submissions/prav.et01.full.csv', row.names=FALSE, quote = FALSE)



rf20.fold1 <- read_csv("./submissions/prav.rf30.fold1.csv")
rf20.fold2 <- read_csv("./submissions/prav.rf30.fold2.csv")
rf20.fold3 <- read_csv("./submissions/prav.rf30.fold3.csv")
rf20.fold4 <- read_csv("./submissions/prav.rf30.fold4.csv")
rf20.fold5 <- read_csv("./submissions/prav.rf30.fold5.csv")
rf20.full  <- read_csv("./submissions/prav.rf30.full.csv")

rf20 <- rbind(rf20.fold1,rf20.fold2,rf20.fold3,rf20.fold4,rf20.fold5)
names(rf20)
colnames(rf20)<-c("Patient_ID","rf20Diabetes")
names(rf20)

names(rf20.full)
colnames(rf20.full)<-c("Patient_ID","rf20Diabetes")
names(rf20.full)
rm(rf20.fold1,rf20.fold2,rf20.fold3,rf20.fold4,rf20.fold5); gc()


train <- left_join(train, rf20,      by="Patient_ID", all.X = TRUE)
test  <- left_join(test , rf20.full, by="Patient_ID", all.X = TRUE)
rm(rf20,rf20.full); gc()

head(train,20)

cat("CV Fold-", , " ", metric, ": ", score(train$DiabetesDispense, train$rf20Diabetes, metric), "\n", sep = "")
###############################################################################################################################
# Sys.time()
# save.image(file = "DSM2017_03.RData" , safe = TRUE)
Sys.time()
load("DSM2017_03.RData")
Sys.time()
###############################################################################################################################


all_data_full_to2015   <- subset(all_data_full, year(DispenseDate) <= 2015)

all_data_full_to2015 <- as.data.table(all_data_full_to2015)

names(all_data_full_to2015)

all_data_full_to2015[ , DrugCount := .N, by = list(Patient_ID, Drug_ID)]

all_data_full_to2015 <- as.data.frame(all_data_full_to2015)

all_data_features <- all_data_full_to2015[,c("Patient_ID","Drug_ID","DrugCount")]

all_data_features <- unique(all_data_features)

sapply(all_data_features, class)

all_data_features$DrugCount <- as.numeric(all_data_features$DrugCount )

head(all_data_features$Drug_ID)
all_data_features$Drug_ID <- paste0("X_",as.character(all_data_features$Drug_ID))

all_data_features[is.na(all_data_features)] <- 0

all_data_features_tabular <- dcast(all_data_features, Patient_ID ~ Drug_ID)

all_data_features_tabular[is.na(all_data_features_tabular)] <- 0


sapply(all_data_features_tabular[,c(1:10)], class)
gc()

#write.csv(all_data_features_tabular, './input/Prav_FE_00.csv', row.names=FALSE, quote = FALSE)
################################################################################################################################


trainingSet <- subset(all_data_features_tabular, Patient_ID < 279201 )
testingSet  <- subset(all_data_features_tabular, Patient_ID >= 279201 )


##################################################################################################################
CVSchema  <- read_csv("./CVSchema/Prav_CVindices_5folds.csv") 

trainingSet <- left_join(trainingSet, CVSchema, by = "Patient_ID")
##################################################################################################################
feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("Patient_ID", "CVindices","DiabetesDispense"))])

#trainingSet[,feature.names] <- sapply(trainingSet[,feature.names], as.numeric)


cv          = 5
bags        = 10
nround.cv   = 510
printeveryn = 50
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
  
  cat("X_build fold Processing\n")
  X_build <- subset(trainingSet, CVindices != i, select = -c( CVindices))
  cat("X_val fold Processing\n")
  X_val   <- subset(trainingSet, CVindices == i, select = -c( CVindices)) 
  
  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=X_build$DiabetesDispense)
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=X_val$DiabetesDispense)
  watchlist <- list( val = dval,train = dtrain)
  
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
  val_predictions <- data.frame(Patient_ID=X_val$Patient_ID, Diabetes = pred_cv)
  
  cat("CV TestingSet prediction Processing\n")
  pred_test  <- predict(XGModel, data.matrix(testingSet[,feature.names]))
  test_predictions <- data.frame(Patient_ID=testingSet$Patient_ID, Diabetes = pred_test)
  
  # if(i == 1)
  # {
  #   write.csv(val_predictions,  './submissions/prav.xgb5.fold1.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.xgb5.fold1-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 2)
  # {
  #   write.csv(val_predictions,  './submissions/prav.xgb5.fold2.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.xgb5.fold2-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 3)
  # {
  #   write.csv(val_predictions,  './submissions/prav.xgb5.fold3.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.xgb5.fold3-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 4)
  # {
  #   write.csv(val_predictions,  './submissions/prav.xgb5.fold4.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.xgb5.fold4-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 5)
  # {
  #   write.csv(val_predictions,  './submissions/prav.xgb5.fold5.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.xgb5.fold5-test.csv', row.names=FALSE, quote = FALSE)
  # }
  
}

# [1]	val-auc:0.500795	train-auc:0.511908 
# [51]	val-auc:0.497824	train-auc:0.593622 
# [101]	val-auc:0.496764	train-auc:0.622183 
# [151]	val-auc:0.495757	train-auc:0.641929 
# [201]	val-auc:0.496395	train-auc:0.656519 
# [251]	val-auc:0.498260	train-auc:0.669596 
# [301]	val-auc:0.498171	train-auc:0.680982 

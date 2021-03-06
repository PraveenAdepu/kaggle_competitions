
require(h2o)

h2o.init(
  nthreads=20,              ## -1: use all available threads
  max_mem_size = "75G"     ## specify the memory size for the H2O cloud, used 47gb
)    
h2o.removeAll()            ## Clean slate - just in case the cluster was already running


###############################################################################################################################
# Sys.time()
# save.image(file = "DSM2017_03.RData" , safe = TRUE)
Sys.time()
load("DSM2017_20.RData")
Sys.time()
###############################################################################################################################


# all_data_full <- subset(all_data_full, IsDeferredScript == 0) # 59450785

all_data_full <- as.data.table(all_data_full) # 64009135

unique(all_data_full$ChronicIllness )
all_data_full$ChronicIllness <- as.integer(as.factor(all_data_full$ChronicIllness))
all_data_full$ChronicIllness[is.na(all_data_full$ChronicIllness)] <- 0
unique(all_data_full$ChronicIllness )

unique(all_data_full$StateCode )
all_data_full$StateCode <- as.integer(as.factor(all_data_full$StateCode))
all_data_full$StateCode[is.na(all_data_full$StateCode)] <- 0
unique(all_data_full$StateCode )

unique(all_data_full$gender )
all_data_full$gender <- as.integer(as.factor(all_data_full$gender))
all_data_full$gender[is.na(all_data_full$gender)] <- 0
unique(all_data_full$gender )

head(all_data_full$postcode.x)
all_data_full$PatientStateCode <- as.integer(strtrim(as.character(all_data_full$postcode.x),1))
head(all_data_full$PatientStateCode)

head(all_data_full$year_of_birth)
all_data_full$PatientAge   <- year(all_data_full$DispenseDate) - all_data_full$year_of_birth
head(all_data_full$PatientAge)
# sapply(all_data_full, class)


all_data_full <- all_data_full[ order(all_data_full$Patient_ID, all_data_full$DispenseDate , decreasing = FALSE ),]

cols = c("Drug_ID","ChronicIllness","Script_Qty") #,"DispenseDate" ,"RepeatsLeft_Qty"

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

all_data_full_build <- as.data.table(all_data_full_build)
all_data_full_build <- all_data_full_build[ order(all_data_full_build$Patient_ID, all_data_full_build$DispenseDate , decreasing = TRUE ),]

all_data_full_build[, OrderRank := 1:.N, by = c("Patient_ID")]
all_data_full_build <- as.data.frame(all_data_full_build)
build_set <- subset(all_data_full_build, OrderRank == 1)


##################################################################################################################

build_set <- as.data.frame(build_set)

build_set[is.na(build_set)]  <- 0
##################################################################################################################

features_20  <- read_csv("./input/Prav_FE_20.csv")
build_set    <- left_join(build_set, features_20, by = "Patient_ID")

features_21  <- read_csv("./input/Prav_FE_21.csv")
build_set    <- left_join(build_set, features_21, by = "Patient_ID")

features_22  <- read_csv("./input/Prav_FE_22.csv")
build_set    <- left_join(build_set, features_22, by = "Patient_ID")

features_24  <- read_csv("./input/Prav_FE_24.csv")
build_set    <- left_join(build_set, features_24, by = "Patient_ID")

build_set[is.na(build_set)] <- 0

#############################################################################################################################
# save(build_set,file="build_set.Rda")
# load("build_set.Rda")

#############################################################################################################################


trainingSet <- subset(build_set, Patient_ID < 279201 )
testingSet  <- subset(build_set, Patient_ID >= 279201 )

##################################################################################################################
CVSchema  <- read_csv("./CVSchema/Prav_CVindices_5folds_10.csv") 

names(CVSchema)

trainingSet <- left_join(trainingSet, CVSchema, by = "Patient_ID")

trainingSet$DispenseDate <- NULL
testingSet$DispenseDate  <- NULL


trainingSet <- as.h2o(trainingSet)
testingSet  <- as.h2o(testingSet)


##################################################################################
#testingSet$DiabetesDispense <- 0

# names(trainingSet)
# ## arange column names to match train and test
# names(testingSet)
# col_idx <- grep("DiabetesDispense", names(testingSet))
# testingSet <- testingSet[, c(col_idx, (1:ncol(testingSet))[-col_idx])]
# names(testingSet)
##################################################################################


########################################################################################################################
# 264 features
feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("Patient_ID", "DispenseDate" , "CVindices","X13","IsDeferredScript","DiabetesDispense"
                                                                         
                                                                         
                                                                         ,"Prescription_Week"                         ,  "Dispense_Week"                               
                                                                         ,"Drug_Code"                                 ,  "NHS_Code"                                    
                                                                         ,"Script_Qty"                                ,  "Dispensed_Qty"                               
                                                                         , "MaxDispense_Qty"                          ,   "PatientPrice_Amt"                            
                                                                         , "WholeSalePrice_Amt"                       ,   "GovernmentReclaim_Amt"                       
                                                                         , "RepeatsTotal_Qty"                                                     
                                                                         , "StreamlinedApproval_Code"                                                      
                                                                         , "StateCode"                                                                              
                                                                         , "year_of_birth"                            ,   "postcode.y"                                  
                                                                         , "IsBannerGroup"                            ,   "MasterProductCode"                           
                                                                         , "MasterProductFullName"                    ,   "BrandName"                                   
                                                                         , "FormCode"                                 ,   "StrengthCode"                                
                                                                         , "PackSizeNumber"                           ,   "GenericIngredientName"                       
                                                                         , "EthicalSubCategoryName"                   ,   "EthicalCategoryName"                         
                                                                         , "ManufacturerCode"                         ,   "ManufacturerName"                            
                                                                         , "ManufacturerGroupID"                      ,   "ManufacturerGroupCode"                       
                                                                         , "ChemistListPrice"                                                        
                                                                         , "OrderRank","ATCLevel5Code", "ATCLevel4Code" , "ATCLevel3Code","ATCLevel2Code"   ,"ATCLevel1Code" ,"SourceSystem_Code"
                                                                         
))])


trainingSet[, feature.names][is.na(trainingSet[, feature.names])] <- 0 
testingSet[, feature.names][is.na(testingSet[, feature.names])]   <- 0 


cv          = 5
bags        = 5
seed        = 2017
ntrees      = 250 
maxdepth    = 20
mtry = max(floor(length(feature.names)/3), 1) # regression

set.seed(seed)
for (i in 1:cv)
  
{
  cat(i ,"fold Processing\n")
  
  cat("X_build fold Processing\n")
  X_build <- trainingSet[trainingSet$CVindices != i, colnames(trainingSet)]
  cat("X_val fold Processing\n")
  X_val   <- trainingSet[trainingSet$CVindices == i, colnames(trainingSet)]
  
  pred_cv_bags   <- rep(0, nrow(X_val[, feature.names]))
  pred_test_bags <- rep(0, nrow(testingSet[,feature.names]))
  for (b in 1:bags) 
  {
    cat(b ," - bag Processing\n")
    cat("X_build training Processing\n")
    set.seed(seed)
    rf <- h2o.randomForest(         ##
            training_frame   = X_build ,  ##
            validation_frame = X_val,     ##
            x=feature.names,              ##
            y="DiabetesDispense",         ##
            mtries = mtry,
            ntrees = ntrees,              ##
            max_depth = maxdepth,         ## Increase depth, from 20
            seed=seed)
    cat("X_val prediction Processing\n")
    pred_cv    <- h2o.predict(rf, X_val[,feature.names])
    
    cat("CV TestingSet prediction Processing\n")
    pred_test  <- h2o.predict(rf, testingSet[,feature.names])
    
    pred_cv_bags   <- pred_cv_bags + as.data.frame(pred_cv)
    pred_test_bags <- pred_test_bags + as.data.frame(pred_test)
  }
  pred_cv_bags   <- pred_cv_bags / bags
  pred_test_bags <- pred_test_bags / bags
  
  
  
  val_predictions <- data.frame(Patient_ID=as.data.frame(X_val$Patient_ID), Diabetes = as.data.frame(pred_cv_bags), DiabetesDispense = as.data.frame(X_val$DiabetesDispense))
  cat("CV Fold-", i, " ", metric, ": ", score(val_predictions$DiabetesDispense, val_predictions$Diabetes, metric), "\n", sep = "")
  val_predictions$DiabetesDispense <- NULL
  test_predictions <- data.frame(Patient_ID=as.data.frame(testingSet$Patient_ID), Diabetes = as.data.frame(pred_test_bags))
  
  if(i == 1)
  {
    write.csv(val_predictions,  './submissions/prav.rf01.fold1.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.rf01.fold1-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 2)
  {
    write.csv(val_predictions,  './submissions/prav.rf01.fold2.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.rf01.fold2-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 3)
  {
    write.csv(val_predictions,  './submissions/prav.rf01.fold3.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.rf01.fold3-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 4)
  {
    write.csv(val_predictions,  './submissions/prav.rf01.fold4.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.rf01.fold4-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 5)
  {
    write.csv(val_predictions,  './submissions/prav.rf01.fold5.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.rf01.fold5-test.csv', row.names=FALSE, quote = FALSE)
  }
  gc()
}

##########################################################################################################
# Full training

fulltrainnrounds = 1.2 * ntrees

# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

fulltest_ensemble <- rep(0, nrow(testingSet[,feature.names]))


for (b in 1:bags) {
  seed = seed + b
  cat(seed, "- seed")
  cat(b ," - bag Processing\n")
  cat("Bagging Full TrainingSet training\n")
  set.seed(seed)
  rfFulltrain <- h2o.randomForest(       
                              training_frame   = trainingSet ,  
                              validation_frame = trainingSet,     
                              x=feature.names,              
                              y="DiabetesDispense",              
                              model_id  = "prav.rfmodel",
                              mtries    = mtry,
                              ntrees    = fulltrainnrounds,                   
                              max_depth = maxdepth,               
                              seed=seed)
  cat("Bagging Full Model prediction Processing\n")
  
  predfull_test         <- h2o.predict(rf, testingSet[,feature.names])
  fulltest_ensemble     <- fulltest_ensemble + as.data.frame(predfull_test)
  
}

fulltest_ensemble     <- fulltest_ensemble / bags
testfull_predictions  <- data.frame(Patient_ID=as.data.frame(testingSet$Patient_ID), Diabetes = as.data.frame(fulltest_ensemble))

write.csv(testfull_predictions, './submissions/prav.rf01.full.csv', row.names=FALSE, quote = FALSE)


# #########################################################################################################
# Bagging - Full train
# #########################################################################################################
impMatrix <- as.data.frame(h2o.varimp(rfFulltrain))

##########################################################################################################

BenchMark <- read.csv("./submissions/Prav_benchmark_01.csv")
val_sub <- read.csv("./submissions/prav.rf01.full.csv")

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
write.csv(Final_sub,  './submissions/Prav_rf01_rank.csv', row.names=FALSE, quote = FALSE)
#########################################################################################################

############################################################################################
model = xgb.dump(XGModelFulltrain, with_stats=TRUE)

names = dimnames(trainingSet[,feature.names])[[2]]
importance_matrix = xgb.importance( names , model = XGModel)
#xgb.plot.importance(importance_matrix)
impMatrix <- as.data.frame(importance_matrix)
# impMatrix[1:12,1]
# impMatrix[,1]
ImpFeature <- impMatrix[,1]


# #########################################################################################################


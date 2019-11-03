###############################################################################################################################
# Sys.time()
# save.image(file = "DSM2017_03.RData" , safe = TRUE)
Sys.time()
load("DSM2017_03.RData")
Sys.time()
###############################################################################################################################


# all_data_full_build <- subset(all_data_full, year(DispenseDate) <= 2014)
# all_data_full_valid <- subset(all_data_full, year(DispenseDate) == 2015)
# 
# 
# ####################################################################################################################################
# set.seed(20)
# features <- c("year_of_birth","Drug_ID")
# head(all_data_full_valid[,features])
# all_data_full_valid[,features] <- apply(all_data_full_valid[,features], 2, normalit)
# 
# Patient_Chronic_Clusters <- kmeans(all_data_full_valid[, features], 12, nstart = 20)
# Patient_Chronic_Clusters
# 
# table(Patient_Chronic_Clusters$cluster, all_data_full_valid$ChronicIllness)
# 
# Patient_Chronic_Clusters$cluster <- as.factor(Patient_Chronic_Clusters$cluster)
# ggplot(all_data_full_valid, aes(year_of_birth, DispenseDate, color = Patient_Chronic_Clusters$cluster)) + geom_point()
# 
# ####################################################################################################################################
# 
# tail(all_data_full,5)
# 
# trans$DispenseDate <- as.POSIXct(trans$Dispense_Week)
# trans <- trans[ order(trans$Patient_ID, trans$DispenseDate , decreasing = FALSE ),]
# trans <- as.data.table(trans)
# cols = c("Drug_ID","DispenseDate")
# anscols = paste("lag1", cols, sep="_")
# trans[, (anscols) := shift(.SD, 1, 0, "lag"), .SDcols=cols, by=Patient_ID]

all_data_full <- as.data.table(all_data_full)
all_data_full <- all_data_full[ order(all_data_full$Patient_ID, all_data_full$DispenseDate , decreasing = FALSE ),]

cols = c("Drug_ID","ChronicIllness") #,"DispenseDate"

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


all_data_full_build <- subset(all_data_full, year(DispenseDate) <= 2014)
all_data_full_valid <- subset(all_data_full, year(DispenseDate) == 2015)

##################################################################################################################
all_data_full_build <- as.data.frame(all_data_full_build)
build_DiabetesDispense <- unique(all_data_full_build[,c("Patient_ID","ChronicIllness")])

head(build_DiabetesDispense)
build_DiabetesDispense <- subset(build_DiabetesDispense, ChronicIllness == "Diabetes")
build_DiabetesDispense$DiabetesDispense <- 1
build_DiabetesDispense$ChronicIllness   <- NULL
##################################################################################################################

all_data_full_valid <- as.data.frame(all_data_full_valid)
valid_DiabetesDispense <- unique(all_data_full_valid[,c("Patient_ID","ChronicIllness")])

head(valid_DiabetesDispense)
valid_DiabetesDispense <- subset(valid_DiabetesDispense, ChronicIllness == "Diabetes")
valid_DiabetesDispense$DiabetesDispense <- 1
valid_DiabetesDispense$ChronicIllness   <- NULL

valid_DiabetesDispense <- unique(rbind(build_DiabetesDispense, valid_DiabetesDispense))
##################################################################################################################

features <- grep("lag", names(all_data_full_build), value=TRUE)

all_data_full_build <- as.data.table(all_data_full_build)
head(all_data_full_build,2)
all_data_full_build <- all_data_full_build[ order(all_data_full_build$Patient_ID, all_data_full_build$DispenseDate , decreasing = TRUE ),]

all_data_full_build[, OrderRank := 1:.N, by = c("Patient_ID")]
all_data_full_build <- as.data.frame(all_data_full_build)
build_set <- subset(all_data_full_build, OrderRank == 1)

all_data_full_valid <- as.data.table(all_data_full_valid)
head(all_data_full_valid,2)
all_data_full_valid <- all_data_full_valid[ order(all_data_full_valid$Patient_ID, all_data_full_valid$DispenseDate , decreasing = TRUE ),]

all_data_full_valid[, OrderRank := 1:.N, by = c("Patient_ID")]
valid_set <- subset(all_data_full_valid, OrderRank == 1)
##################################################################################################################
OriginalFeatures <- c("Patient_ID","Drug_ID","ChronicIllness", "DispenseDate")

model.feature <- union(OriginalFeatures ,features)

head(valid_set)

build_set <- as.data.frame(build_set)
valid_set <- as.data.frame(valid_set)

building   <- build_set[, model.feature]
validation <- valid_set[, model.feature]

building <- left_join(building,     build_DiabetesDispense , by="Patient_ID")
validation <- left_join(validation, valid_DiabetesDispense , by="Patient_ID")

all_data <- rbind(building, validation)

factor.features     <- grep("ChronicIllness", names(all_data_full_build), value=TRUE)
sapply(all_data[,factor.features], class)
head(all_data[,factor.features])
for (f in factor.features) {
  #if (class(train_test[[f]])=="character") {
    cat("VARIABLE : ",f,"\n")
    levels <- unique(all_data[[f]])
    all_data[[f]] <- as.integer(factor(all_data[[f]], levels=levels))
  #}
}

all_data[is.na(all_data)] <- 0

building <- subset(all_data, year(DispenseDate) <= 2014)
validation <- subset(all_data, year(DispenseDate) == 2015)


OriginalFeatures <- c("Patient_ID","Drug_ID","ChronicIllness", "DispenseDate")

########################################################################################################################
#names(testingSet)
feature.names     <- names(building[,-which(names(building) %in% c("Patient_ID", "DispenseDate" ,"DiabetesDispense"))])



cv          = 5
bags        = 1
nround.cv   = 155 
printeveryn = 50
seed        = 2017

param <- list(  "objective"        = "reg:linear", 
                "booster"          = "gbtree",
                "eval_metric"      = "auc",
                "nthread"          = 25,     
                "max_depth"        = 7,     
                "eta"              = 0.05, 
                "subsample"        = 0.7,  
                "colsample_bytree" = 0.7,  
                "min_child_weight" = 1     
                
)

dtrain <-xgb.DMatrix(data=data.matrix(building[, feature.names]),label=building$DiabetesDispense)
dval   <-xgb.DMatrix(data=data.matrix(validation[, feature.names])   ,label=validation$DiabetesDispense)
watchlist <- list( val = dval,train = dtrain)
pred_cv_bags   <- rep(0, nrow(validation[, feature.names]))

for (b in 1:bags) 
{
  seed = seed + b
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
pred_cv    <- predict(XGModel, data.matrix(validation[,feature.names]))
cat("CV bag- ", b ," ", metric, ": ", score(validation$DiabetesDispense, pred_cv, metric), "\n", sep = "")

pred_cv_bags   <- pred_cv_bags + pred_cv
}
pred_cv_bags   <- pred_cv_bags / bags
cat("CV Fold- 2015 ", metric, ": ", score(validation$DiabetesDispense, pred_cv_bags, metric), "\n", sep = "")

head(validation$DiabetesDispense)
head(pred_cv)
head(pred_cv_bags)

val_predictions <- data.frame(Patient_ID=validation$Patient_ID, Diabetes = pred_cv_bags)

write.csv(val_predictions,  './submissions/Prav_validation2015_01.csv', row.names=FALSE, quote = FALSE)
#################################################################################################################

BenchMark <- read.csv("./submissions/Prav_benchmark_01.csv")

MergeSub <- left_join(BenchMark, val_predictions, by ="Patient_ID")

head(MergeSub,100)

MergeSub$Diabetes <- ifelse(MergeSub$Diabetes.x == 1, 1, MergeSub$Diabetes.y)

MergeSub$Diabetes[is.na(MergeSub$Diabetes) ] <- 0

MergeSub$Diabetes <- ifelse(MergeSub$Diabetes < 0, 0, MergeSub$Diabetes)

Final_sub <- MergeSub[,c("Patient_ID","Diabetes")]

write.csv(Final_sub,  './submissions/Prav_xgb01.csv', row.names=FALSE, quote = FALSE)
#################################################################################################################
MergeSub[is.na(MergeSub)] <- 0 
score(MergeSub$Diabetes.x, MergeSub$Diabetes.y, metric)
score(MergeSub$Diabetes.x, MergeSub$Diabetes, metric)
score(MergeSub$Diabetes.y, MergeSub$Diabetes, metric)


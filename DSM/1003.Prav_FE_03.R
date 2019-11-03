###############################################################################################################################
# Sys.time()
# save.image(file = "DSM2017_03.RData" , safe = TRUE)
Sys.time()
load("DSM2017_03.RData")
Sys.time()
###############################################################################################################################


# all_data_full <- subset(all_data_full, IsDeferredScript == 0) # 59450785

all_data_full <- as.data.table(all_data_full) # 59313254
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

features_00  <- read_csv("./input/Prav_FE_00.csv") 

build_set <- left_join(build_set, features_00, by = "Patient_ID")

write.csv(build_set, './input/Prav_FE_01.csv', row.names=FALSE, quote = FALSE)
##################################################################################################################
# save(build_set,file="build_set.Rda")
# load("build_set.Rda")


##################################################################################################################
trainingSet <- subset(build_set, Patient_ID < 279201 )
testingSet  <- subset(build_set, Patient_ID >= 279201 )


##################################################################################################################
CVSchema  <- read_csv("./CVSchema/Prav_CVindices_5folds.csv") 

names(CVSchema)

trainingSet <- left_join(trainingSet, CVSchema, by = "Patient_ID")

########################################################################################################################

feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("Patient_ID", "DispenseDate" ,"DiabetesDispense", "CVindices","X13"))])

########################################################################################################################


###############################################################################################################################
# Sys.time()
# save.image(file = "DSM2017_03.RData" , safe = TRUE)
Sys.time()
load("DSM2017_20.RData")
Sys.time()
###############################################################################################################################


# all_data_full <- subset(all_data_full, IsDeferredScript == 0) # 59450785

all_data_full <- as.data.table(all_data_full) # 64009135


all_data_full <- all_data_full[ order(all_data_full$Patient_ID, all_data_full$DispenseDate , decreasing = FALSE ),]

cols = c("DispenseDate") 

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

head(build_set,2)

build_set$lag1_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag1_DispenseDate , units = c("days")))
build_set$lag2_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag2_DispenseDate , units = c("days")))
build_set$lag3_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag3_DispenseDate , units = c("days")))
build_set$lag4_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag4_DispenseDate , units = c("days")))
build_set$lag5_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag5_DispenseDate , units = c("days")))
build_set$lag6_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag6_DispenseDate , units = c("days")))
build_set$lag7_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag7_DispenseDate , units = c("days")))
build_set$lag8_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag8_DispenseDate , units = c("days")))
build_set$lag9_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag9_DispenseDate , units = c("days")))

build_set$lag10_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag10_DispenseDate , units = c("days")))
build_set$lag11_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag11_DispenseDate , units = c("days")))
build_set$lag12_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag12_DispenseDate , units = c("days")))
build_set$lag13_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag13_DispenseDate , units = c("days")))
build_set$lag14_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag14_DispenseDate , units = c("days")))
build_set$lag15_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag15_DispenseDate , units = c("days")))
build_set$lag16_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag16_DispenseDate , units = c("days")))
build_set$lag17_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag17_DispenseDate , units = c("days")))
build_set$lag18_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag18_DispenseDate , units = c("days")))
build_set$lag19_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag19_DispenseDate , units = c("days")))

build_set$lag20_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag20_DispenseDate , units = c("days")))
build_set$lag21_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag21_DispenseDate , units = c("days")))
build_set$lag22_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag22_DispenseDate , units = c("days")))
build_set$lag23_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag23_DispenseDate , units = c("days")))
build_set$lag24_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag24_DispenseDate , units = c("days")))
build_set$lag25_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag25_DispenseDate , units = c("days")))
build_set$lag26_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag26_DispenseDate , units = c("days")))
build_set$lag27_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag27_DispenseDate , units = c("days")))
build_set$lag28_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag28_DispenseDate , units = c("days")))
build_set$lag29_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag29_DispenseDate , units = c("days")))

build_set$lag30_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag30_DispenseDate , units = c("days")))
build_set$lag31_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag31_DispenseDate , units = c("days")))
build_set$lag32_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag32_DispenseDate , units = c("days")))
build_set$lag33_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag33_DispenseDate , units = c("days")))
build_set$lag34_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag34_DispenseDate , units = c("days")))
build_set$lag35_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag35_DispenseDate , units = c("days")))
build_set$lag36_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag36_DispenseDate , units = c("days")))
build_set$lag37_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag37_DispenseDate , units = c("days")))
build_set$lag38_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag38_DispenseDate , units = c("days")))
build_set$lag39_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag39_DispenseDate , units = c("days")))

build_set$lag40_DispenseDays <-  as.double(difftime(build_set$DispenseDate ,build_set$lag40_DispenseDate , units = c("days")))


lag_days <- grep("DispenseDays", names(build_set), value = TRUE)

FE_01.features <- union("Patient_ID",lag_days)

build_set_features <- build_set[,FE_01.features]

##################################################################################################################

build_set_features <- as.data.frame(build_set_features)

build_set_features[is.na(build_set_features)]  <- 0
##################################################################################################################

write.csv(build_set_features, "./input/build_set_FE_30.csv", row.names=FALSE, quote = FALSE)
##################################################################################################################

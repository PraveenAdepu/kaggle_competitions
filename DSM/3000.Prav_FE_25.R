###############################################################################################################################
# Sys.time()
# save.image(file = "DSM2017_20.RData" , safe = TRUE)
Sys.time()
load("DSM2017_20.RData")
Sys.time()
###############################################################################################################################
# length(unique(all_data_full$IsDeferredScript))

all_data_full_to2015   <- subset(all_data_full, year(DispenseDate) <= 2015)
tail(all_data_full_to2015)

UniqueColumns <- c("Patient_ID","DispenseDate")
all_data_full_to2015 <- unique(all_data_full_to2015[,UniqueColumns])
all_data_full_to2015 <- as.data.table(all_data_full_to2015)


all_data_full_to2015[ , VisitCount := .N, by = list(Patient_ID)]

all_data_full_to2015 <- as.data.frame(all_data_full_to2015)

all_data_features <- all_data_full_to2015[,c("Patient_ID","VisitCount")]

all_data_features <- unique(all_data_features)

write.csv(all_data_features, './input/Prav_FE_25.csv', row.names=FALSE, quote = FALSE)
################################################################################################################################

###############################################################################################################################
# Sys.time()
# save.image(file = "DSM2017_20.RData" , safe = TRUE)
Sys.time()
load("DSM2017_20.RData")
Sys.time()
###############################################################################################################################
#all_data_full$DispenseDate <- as.POSIXct(all_data_full$Dispense_Week)

all_data_full_to2015   <- subset(all_data_full, year(DispenseDate) <= 2015)

all_data_full_to2015 <- as.data.table(all_data_full_to2015)

all_data_full_to2015[ , ChronicCount := .N, by = list(Patient_ID, ChronicIllness)]

all_data_full_to2015 <- as.data.frame(all_data_full_to2015)

all_data_features <- all_data_full_to2015[,c("Patient_ID","ChronicIllness","ChronicCount")]

all_data_features <- unique(all_data_features)

all_data_features_tabular <- cast(all_data_features, Patient_ID ~ ChronicIllness)

all_data_features_tabular[is.na(all_data_features_tabular)] <- 0

head(all_data_features_tabular,1)
write.csv(all_data_features_tabular, './input/Prav_FE_20.csv', row.names=FALSE, quote = FALSE)
################################################################################################################################

###############################################################################################################################
#all_data_full$DispenseDate <- as.POSIXct(all_data_full$Dispense_Week)


all_data_full_to2015   <- subset(all_data_full, year(DispenseDate) <= 2015)
all_data_full_to2015   <- as.data.table(all_data_full_to2015)

length(unique(all_data_full_to2015$ATCLevel1Code))

all_data_full_to2015[ , ACT1Count := .N, by = list(Patient_ID, ATCLevel1Code)]

all_data_full_to2015   <- as.data.frame(all_data_full_to2015)
all_data_features      <- all_data_full_to2015[,c("Patient_ID","ATCLevel1Code","ACT1Count")]
all_data_features      <- unique(all_data_features)

head(all_data_features)

all_data_features$ATCLevel1Code <- paste0("ATc1_", all_data_features$ATCLevel1Code)

all_data_features_tabular <- cast(all_data_features, Patient_ID ~ ATCLevel1Code)

all_data_features_tabular[is.na(all_data_features_tabular)] <- 0

head(all_data_features_tabular,1)
write.csv(all_data_features_tabular, './input/Prav_FE_21.csv', row.names=FALSE, quote = FALSE)
################################################################################################################################

###############################################################################################################################
#all_data_full$DispenseDate <- as.POSIXct(all_data_full$Dispense_Week)


all_data_full_to2015   <- subset(all_data_full, year(DispenseDate) <= 2015)
all_data_full_to2015   <- as.data.table(all_data_full_to2015)

length(unique(all_data_full_to2015$ATCLevel2Code))

all_data_full_to2015[ , ACT2Count := .N, by = list(Patient_ID, ATCLevel2Code)]

all_data_full_to2015   <- as.data.frame(all_data_full_to2015)
all_data_features      <- all_data_full_to2015[,c("Patient_ID","ATCLevel2Code","ACT2Count")]
all_data_features      <- unique(all_data_features)

head(all_data_features)

all_data_features$ATCLevel2Code <- paste0("ATc2_", all_data_features$ATCLevel2Code)

all_data_features_tabular <- cast(all_data_features, Patient_ID ~ ATCLevel2Code)


all_data_features_tabular[is.na(all_data_features_tabular)] <- 0

head(all_data_features_tabular,1)
write.csv(all_data_features_tabular, './input/Prav_FE_22.csv', row.names=FALSE, quote = FALSE)
################################################################################################################################

###############################################################################################################################
#all_data_full$DispenseDate <- as.POSIXct(all_data_full$Dispense_Week)


all_data_full_to2015   <- subset(all_data_full, year(DispenseDate) <= 2015)
all_data_full_to2015   <- as.data.table(all_data_full_to2015)

length(unique(all_data_full_to2015$ATCLevel3Code))

all_data_full_to2015[ , ACT3Count := .N, by = list(Patient_ID, ATCLevel3Code)]

all_data_full_to2015   <- as.data.frame(all_data_full_to2015)
all_data_features      <- all_data_full_to2015[,c("Patient_ID","ATCLevel3Code","ACT3Count")]
all_data_features      <- unique(all_data_features)

head(all_data_features)

all_data_features$ATCLevel3Code <- paste0("ATc3_", all_data_features$ATCLevel3Code)

all_data_features_tabular <- cast(all_data_features, Patient_ID ~ ATCLevel3Code)


all_data_features_tabular[is.na(all_data_features_tabular)] <- 0

head(all_data_features_tabular,1)
write.csv(all_data_features_tabular, './input/Prav_FE_23.csv', row.names=FALSE, quote = FALSE)
################################################################################################################################

###############################################################################################################################
#all_data_full$DispenseDate <- as.POSIXct(all_data_full$Dispense_Week)


all_data_full_to2015   <- subset(all_data_full, year(DispenseDate) <= 2015)
all_data_full_to2015   <- as.data.table(all_data_full_to2015)

length(unique(all_data_full_to2015$SourceSystem_Code))

all_data_full_to2015[ , SSCount := .N, by = list(Patient_ID, SourceSystem_Code)]

all_data_full_to2015   <- as.data.frame(all_data_full_to2015)
all_data_features      <- all_data_full_to2015[,c("Patient_ID","SourceSystem_Code","SSCount")]
all_data_features      <- unique(all_data_features)

head(all_data_features)

all_data_features$SourceSystem_Code <- paste0("SS_", all_data_features$SourceSystem_Code)

all_data_features_tabular <- cast(all_data_features, Patient_ID ~ SourceSystem_Code)


all_data_features_tabular[is.na(all_data_features_tabular)] <- 0

head(all_data_features_tabular,1)
write.csv(all_data_features_tabular, './input/Prav_FE_24.csv', row.names=FALSE, quote = FALSE)
################################################################################################################################


###############################################################################################################################
#all_data_full$DispenseDate <- as.POSIXct(all_data_full$Dispense_Week)


all_data_full_to2015   <- subset(all_data_full, year(DispenseDate) <= 2015)
all_data_full_to2015   <- as.data.table(all_data_full_to2015)

length(unique(all_data_full_to2015$NHS_Code))

all_data_full_to2015[ , SSCount := .N, by = list(Patient_ID, SourceSystem_Code)]

all_data_full_to2015   <- as.data.frame(all_data_full_to2015)
all_data_features      <- all_data_full_to2015[,c("Patient_ID","SourceSystem_Code","SSCount")]
all_data_features      <- unique(all_data_features)

head(all_data_features)

all_data_features$SourceSystem_Code <- paste0("SS_", all_data_features$SourceSystem_Code)

all_data_features_tabular <- cast(all_data_features, Patient_ID ~ SourceSystem_Code)


all_data_features_tabular[is.na(all_data_features_tabular)] <- 0

head(all_data_features_tabular,1)
write.csv(all_data_features_tabular, './input/Prav_FE_24.csv', row.names=FALSE, quote = FALSE)
################################################################################################################################


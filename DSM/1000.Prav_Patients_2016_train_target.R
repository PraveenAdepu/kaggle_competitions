################################################################################
Sys.time()
load("DSM2017_20.RData")
Sys.time()
################################################################################

all_data_full_2016  <- subset(all_data_full, year(DispenseDate) == 2016)

min(all_data_full_2016$DispenseDate)
max(all_data_full_2016$DispenseDate)


all_data_full_2016 <- unique(all_data_full_2016[,c("Patient_ID","ChronicIllness")])
Patients_target <- subset(Patients,  Patient_ID < 279201)


DiabetesDispense_2016 <- subset(all_data_full_2016, ChronicIllness == "Diabetes")
DiabetesDispense_2016$DiabetesDispense <- 1
DiabetesDispense_2016$ChronicIllness   <- NULL

Patients_target <- left_join(Patients_target, DiabetesDispense_2016, by = "Patient_ID")

Patients_target <- Patients_target[,c("Patient_ID","DiabetesDispense")]

Patients_target$DiabetesDispense[is.na(Patients_target$DiabetesDispense)] <- 0

head(Patients_target)
table(Patients_target$DiabetesDispense)

##############################################################################

write.csv(Patients_target,  './input/Prav_Patients_2016_train_target_20.csv', row.names=FALSE, quote = FALSE)
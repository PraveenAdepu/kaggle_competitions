###############################################################################################################################
# Sys.time()
# save.image(file = "DSM2017_03.RData" , safe = TRUE)
Sys.time()
load("DSM2017_03.RData")
Sys.time()
###############################################################################################################################


all_data_full_build <- subset(all_data_full, year(DispenseDate) <= 2014)
all_data_full_valid <- subset(all_data_full, year(DispenseDate) == 2015)

ValidationSet <- unique(all_data_full_valid[,c("Patient_ID","ChronicIllness")])

head(ValidationSet)

ValidationSet$Diabetes <- ifelse(ValidationSet$ChronicIllness == "Diabetes",1,0)

ValidationSet$Diabetes[is.na(ValidationSet$Diabetes)] <- 0
head(ValidationSet,20)

ValidationSet <- unique(ValidationSet[,c("Patient_ID","Diabetes")])

head(ValidationSet,20)

ValidationSet1 <- ValidationSet %>% 
                    group_by(Patient_ID) %>% 
                    summarise(Frequency = sum(Diabetes))

head(ValidationSet1,20)
ValidationSet1$Diabetes <- ifelse(ValidationSet1$Frequency >= 1,1,0)

length(ValidationSet1$Patient_ID)
length(unique(ValidationSet1$Patient_ID))
write.csv(ValidationSet1[,c("Patient_ID","Diabetes")],  './input/Prav_ValidationSet_2015.csv', row.names=FALSE, quote = FALSE)
###################################################################################################################################
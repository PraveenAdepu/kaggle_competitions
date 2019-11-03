###############################################################################################################################
Sys.time()
load("DSM2017_10.RData")
Sys.time()
###############################################################################################################################

names(Stores)[1]   <- "Store_ID"
names(Patients)[1] <- "Patient_ID"
names(Drug)[1]     <- "Drug_ID" # "MasterProductID"


Chronic <- ChronicIllness[,1:2]

names(Chronic)
names(Chronic)[2] = "Drug_ID"
head(Chronic)

# head(trans,1)
# head(Patients,1)
# head(Stores,1)
# head(Drug,1)
# head(ATC,1)
# head(ChronicIllness,1)

unique(length(Patients$Patient_ID))           # 558352
unique(length(Stores$Store_ID))               #   2822
unique(length(Drug$Drug_ID))                  #  13792
unique(length(ChronicIllness$MasterProductID))#   2207

names(trans)

all_data <- left_join(trans, Patients, by="Patient_ID")
all_data <- left_join(all_data, Stores, by = "Store_ID")
all_data <- left_join(all_data, Drug, by = "Drug_ID")
all_data <- left_join(all_data, Chronic, by = "Drug_ID")

###################################################################################################################

all_data_full <- left_join(Transactions, Patients, by="Patient_ID")
all_data_full <- left_join(all_data_full, Stores, by = "Store_ID")
all_data_full <- left_join(all_data_full, Drug, by = "Drug_ID")
all_data_full <- left_join(all_data_full, Chronic, by = "Drug_ID")
###############################################################################################################################

table(all_data_full$IsDeferredScript)

###############################################################################################################################
gc()
Sys.time()
save.image(file = "DSM2017_11.RData" , safe = TRUE)
Sys.time()
# load("DSM2017_11.RData")
# Sys.time()
###############################################################################################################################


###############################################################################################################################
Sys.time()
load("DSM2017_11.RData")
Sys.time()
###############################################################################################################################

######################################################################################################

all_data_full$DispenseDate <- as.POSIXct(all_data_full$Dispense_Week)

###############################################################################################################################
# Sys.time()
# save.image(file = "DSM2017_12.RData" , safe = TRUE)
# Sys.time()
# load("DSM2017_12.RData")
# Sys.time()
###############################################################################################################################


unique(all_data_full$IsDeferredScript)
table(all_data_full$IsDeferredScript)

# False       True 
# 64009135    16429 

all_data_full$IsDeferredScript[is.na(all_data_full$IsDeferredScript)] <- "False"

all_data_full_NonDeferred <- subset(all_data_full, IsDeferredScript == "False")

unique(all_data_full_NonDeferred$IsDeferredScript)
table(all_data_full_NonDeferred$IsDeferredScript)

all_data_full <- all_data_full_NonDeferred

rm(all_data_full_NonDeferred)

# 
# all_data_fullPost2015 <- subset(all_data_full, year(DispenseDate) > 2015)
# all_data_fullTill2015 <- subset(all_data_full, year(DispenseDate) <= 2015)
# ######################################################################################################
# # temp <- subset(all_data_full, Patient_ID == "558351") # 279201 ,558352
# # 
# # temp <- temp[order(temp$DispenseDate),]
# # 
# # max(temp$DispenseDate)
# # min(temp$DispenseDate)
# # 
# # unique(temp$ChronicIllness)
# 
# ######################################################################################################
# 
# sample_sub <- read_csv("./submissions/diabetes_submission_example.csv")
# names(sample_sub)[1] <- "Patient_ID"
# head(sample_sub)
# 
# # No patient records exists in post 2015
# TestingSet <- inner_join(all_data_fullPost2015, sample_sub, by = "Patient_ID")
# 
# DiabeticKnowPre2015 <- subset(all_data_fullTill2015, ChronicIllness == "Diabetes")
# 
# DiabeticKnowTill2015 <- DiabeticKnowPre2015[,c("Patient_ID","ChronicIllness")]
# 
# DiabeticKnowTill2015 <- unique(DiabeticKnowTill2015)
# 
# TestingSetLeak <- inner_join(DiabeticKnowTill2015,sample_sub, by = "Patient_ID")
# 
# head(TestingSetLeak)
# 
# TestingSetLeak$Diabetes <- 1
# 
# TestingLeakSub <- TestingSetLeak[,c("Patient_ID","Diabetes")]
# 
# write.csv(TestingLeakSub,  './submissions/TestingSet_LeakPatients.csv', row.names=FALSE, quote = FALSE)
# 
# First_sub <- left_join(sample_sub, TestingLeakSub,by="Patient_ID")
# 
# First_sub$Diabetes.y[is.na(First_sub$Diabetes.y)] <- 0
# 
# head(First_sub,25)
# 
# First_sub$Diabetes <- ifelse(First_sub$Diabetes.y == 1, 1, First_sub$Diabetes.x)
# 
# submission.full <- First_sub[,c("Patient_ID","Diabetes")]
# 
# write.csv(submission.full,  './submissions/Prav_benchmark_01.csv', row.names=FALSE, quote = FALSE)
# ######################################################################################################
# rm(temp, temp22); gc()
# rm(all_data_fullPost2015,all_data_fullTill2015,DiabeticKnowPre2015,DiabeticKnowTill2015) ; gc()
# rm(First_sub,sample_sub,submission.full,tempPost2015, tempTill2015,TestingLeakSub,TestingSet,TestingSetLeak); gc()

gc()
###############################################################################################################################
Sys.time()
save.image(file = "DSM2017_20.RData" , safe = TRUE)
Sys.time()
# load("DSM2017_12.RData")
# Sys.time()
###############################################################################################################################



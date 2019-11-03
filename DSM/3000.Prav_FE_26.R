###############################################################################################################################
# Sys.time()
# save.image(file = "DSM2017_20.RData" , safe = TRUE)
Sys.time()
load("DSM2017_20.RData")
Sys.time()
###############################################################################################################################
# length(unique(all_data_full$IsDeferredScript))

all_data_full_to2015   <- subset(all_data_full, year(DispenseDate) <= 2015)
head(all_data_full_to2015)
gc()
all_data_features <-  sqldf("SELECT Patient_ID, 
         sum(Script_Qty)[ScriptQty_sum], min(Script_Qty)[ScriptQty_min], max(Script_Qty) [ScriptQty_max] ,avg(Script_Qty) [ScriptQty_mean],
         sum(Dispensed_Qty)[DispensedQty_sum], min(Dispensed_Qty)[DispensedQty_min], max(Dispensed_Qty) [DispensedQty_max] ,avg(Dispensed_Qty) [DispensedQty_mean],
         sum(PatientPrice_Amt)[PatientPrice_sum], min(PatientPrice_Amt)[PatientPrice_min], max(PatientPrice_Amt) [PatientPrice_max] ,avg(PatientPrice_Amt) [PatientPrice_mean],
         sum(WholeSalePrice_Amt)[WholeSalePrice_sum], min(WholeSalePrice_Amt)[WholeSalePrice_min], max(WholeSalePrice_Amt) [WholeSalePrice_max] ,avg(WholeSalePrice_Amt) [WholeSalePrice_mean],
         sum(GovernmentReclaim_Amt)[GovernmentReclaim_sum], min(GovernmentReclaim_Amt)[GovernmentReclaim_min], max(GovernmentReclaim_Amt) [GovernmentReclaim_max] ,avg(GovernmentReclaim_Amt) [GovernmentReclaim_mean], 
         sum(ChemistListPrice)[ChemistListPrice_sum], min(ChemistListPrice)[ChemistListPrice_min], max(ChemistListPrice) [ChemistListPrice_max] ,avg(ChemistListPrice) [ChemistListPrice_mean] 
       FROM all_data_full_to2015 Group by Patient_ID")

head(all_data_features)

write.csv(all_data_features, './input/Prav_FE_26.csv', row.names=FALSE, quote = FALSE)
################################################################################################################################

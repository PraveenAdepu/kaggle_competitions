###############################################################################################################################
# Sys.time()
# save.image(file = "DSM2017_03.RData" , safe = TRUE)
Sys.time()
load("DSM2017_03.RData")
Sys.time()
###############################################################################################################################


all_data_full_build <- subset(all_data_full, year(DispenseDate) <= 2014)
all_data_full_valid <- subset(all_data_full, year(DispenseDate) == 2015)


ValidationSet <- read_csv("./input/Prav_ValidationSet_2015.csv")

Build_DiabeticKnown <- subset(all_data_full_build[,c("Patient_ID","ChronicIllness")], ChronicIllness == "Diabetes")
Build_DiabeticKnown <- unique(Build_DiabeticKnown)

ValidationSetLeak <- inner_join(Build_DiabeticKnown,ValidationSet, by = "Patient_ID")

# head(ValidationSetLeak)
ValidationSetLeak$Diabetes_build <- 1
ValidationSetLeak$Diabetes <- NULL

validation_Check <- left_join(ValidationSet, ValidationSetLeak,by="Patient_ID")
validation_Check$Diabetes_build[is.na(validation_Check$Diabetes_build)] <- 0
validation_Check$ChronicIllness <- NULL

validation_Check$Diabetes_build <- as.integer(validation_Check$Diabetes_build)
# head(validation_Check,25)

validation_Check <- subset(validation_Check, Patient_ID >= 279201)

cat("CV Fold- 2015 ", " ", metric, ": ", score(validation_Check$Diabetes_build, validation_Check$Diabetes, metric), "\n", sep = "")
# CV Fold- 2015  auc: 0.9231684

####################################################################################################################################
unique(Build_DiabeticKnown$ChronicIllness)
Build_DiabeticKnown$ChronicIllness <- NULL

Build_Diabetes <- inner_join(all_data_full_build,Build_DiabeticKnown,by="Patient_ID")

unique(Build_Diabetes$ChronicIllness)

Build_NoDiabetic <- subset(all_data_full_build[,c("Patient_ID","ChronicIllness")], ChronicIllness != "Diabetes")
Build_NoDiabetic <- unique(Build_NoDiabetic)
unique(Build_NoDiabetic$ChronicIllness)

length(unique(all_data_full$ChronicIllness))

ggplot(all_data_full_valid, aes(year_of_birth, DispenseDate, color = ChronicIllness)) + geom_point()

####################################################################################################################################
set.seed(20)
features <- c("year_of_birth")
Patient_Chronic_Clusters <- kmeans(all_data_full_build[, features], 12, nstart = 20)
Patient_Chronic_Clusters

table(Patient_Chronic_Clusters$cluster, all_data_full_build$ChronicIllness)

Patient_Chronic_Clusters$cluster <- as.factor(Patient_Chronic_Clusters$cluster)
ggplot(all_data_full_build, aes(year_of_birth, DispenseDate, color = Patient_Chronic_Clusters$cluster)) + geom_point()


names(all_data_full_build)

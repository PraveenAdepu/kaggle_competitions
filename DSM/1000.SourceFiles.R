


###############################################################################################################################
# Start - Read all transaction files in a loop
# https://inclass.kaggle.com/c/dsm2017
###############################################################################################################################
file_list <- list.files("./input/Transactions/")

Transactions<- read.table(paste0("./input/Transactions/",file_list[1]), header=TRUE, sep="\t") 

for (file in file_list[2:100]){
  
    print(paste0("./input/Transactions/",file))
 
    temp_dataset <-read.table(paste0("./input/Transactions/",file), header=TRUE, sep="\t")
    Transactions<-rbind(Transactions, temp_dataset)
    rm(temp_dataset)
  
}

names(Transactions)



###############################################################################################################################
# Start - Real all transaction files in a loop
###############################################################################################################################
trans <- read.table("./input/Transactions/patients_1.txt", sep="\t", header=TRUE)

ATC            <- read.table("./input/Lookups/ATC_LookUp.txt"           , sep="\t", header=TRUE)
ChronicIllness <- read.table("./input/Lookups/ChronicIllness_LookUp.txt", sep="\t", header=TRUE)
Drug           <- read.table("./input/Lookups/Drug_LookUp.txt"          , sep="\t", header=TRUE)
Patients       <- read.table("./input/Lookups/patients.txt"             , sep="\t", header=TRUE)
Stores         <- read.table("./input/Lookups/stores.txt"               , sep="\t", header=TRUE)

gc()
###############################################################################################################################
Sys.time()
save.image(file = "DSM2017_10.RData" , safe = TRUE)
Sys.time()
# load("DSM2017_10.RData")
# Sys.time()
###############################################################################################################################

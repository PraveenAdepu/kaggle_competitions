
###############################################################################################################################
# Sys.time()
# save.image(file = "DSM2017_03.RData" , safe = TRUE)
Sys.time()
load("DSM2017_20.RData")
Sys.time()
###############################################################################################################################


# all_data_full <- subset(all_data_full, IsDeferredScript == 0) # 59450785

all_data_full <- as.data.table(all_data_full) # 64009135

unique(all_data_full$ChronicIllness )
all_data_full$ChronicIllness <- as.integer(as.factor(all_data_full$ChronicIllness))
all_data_full$ChronicIllness[is.na(all_data_full$ChronicIllness)] <- 0
unique(all_data_full$ChronicIllness )

unique(all_data_full$StateCode )
all_data_full$StateCode <- as.integer(as.factor(all_data_full$StateCode))
all_data_full$StateCode[is.na(all_data_full$StateCode)] <- 0
unique(all_data_full$StateCode )

unique(all_data_full$gender )
all_data_full$gender <- as.integer(as.factor(all_data_full$gender))
all_data_full$gender[is.na(all_data_full$gender)] <- 0
unique(all_data_full$gender )

head(all_data_full$postcode.x)
all_data_full$PatientStateCode <- as.integer(strtrim(as.character(all_data_full$postcode.x),1))
head(all_data_full$PatientStateCode)

head(all_data_full$year_of_birth)
all_data_full$PatientAge   <- year(all_data_full$DispenseDate) - all_data_full$year_of_birth
head(all_data_full$PatientAge)
# sapply(all_data_full, class)


all_data_full <- all_data_full[ order(all_data_full$Patient_ID, all_data_full$DispenseDate , decreasing = FALSE ),]

cols = c("Drug_ID","ChronicIllness","Script_Qty") #,"DispenseDate" ,"RepeatsLeft_Qty"

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


##################################################################################################################

build_set <- as.data.frame(build_set)

build_set[is.na(build_set)]  <- 0
##################################################################################################################

features_20  <- read_csv("./input/Prav_FE_20.csv")
build_set    <- left_join(build_set, features_20, by = "Patient_ID")

features_21  <- read_csv("./input/Prav_FE_21.csv")
build_set    <- left_join(build_set, features_21, by = "Patient_ID")

features_22  <- read_csv("./input/Prav_FE_22.csv")
build_set    <- left_join(build_set, features_22, by = "Patient_ID")

features_24  <- read_csv("./input/Prav_FE_24.csv")
build_set    <- left_join(build_set, features_24, by = "Patient_ID")

build_set[is.na(build_set)] <- 0

#############################################################################################################################
# save(build_set,file="build_set.Rda")
# load("build_set.Rda")

factorFeatures <- c( "Prescription_Week"  ,  "Dispense_Week"                               
                     ,"Drug_Code"  ,  "NHS_Code"                                    
                     , "postcode.y"  , "MasterProductCode"                           
                     , "MasterProductFullName" ,   "BrandName"                                   
                     , "FormCode"  ,   "StrengthCode"                                
                     ,   "GenericIngredientName"                       
                     , "EthicalSubCategoryName",   "EthicalCategoryName"                         
                     , "ManufacturerCode" ,   "ManufacturerName"                            
                     ,   "ManufacturerGroupCode"                   
                     ,"ATCLevel5Code", "ATCLevel4Code" , "ATCLevel3Code","ATCLevel2Code"   ,"ATCLevel1Code" ,"SourceSystem_Code")

sapply(build_set[,factorFeatures], class)

head(build_set[,factorFeatures])

for (f in factorFeatures) {
  cat("VARIABLE : ",f,"\n")
  levels <- unique(build_set[[f]])
  build_set[[f]] <- as.integer(factor(build_set[[f]], levels=levels))
  
}

#############################################################################################################################


trainingSet <- subset(build_set, Patient_ID < 279201 )
testingSet  <- subset(build_set, Patient_ID >= 279201 )

##################################################################################################################
CVSchema  <- read_csv("./CVSchema/Prav_CVindices_5folds_10.csv") 

names(CVSchema)

trainingSet <- left_join(trainingSet, CVSchema, by = "Patient_ID")

trainingSet$DispenseDate <- NULL
testingSet$DispenseDate  <- NULL


########################################################################################################################
# 282 features
feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("Patient_ID", "DispenseDate" , "CVindices","X13","IsDeferredScript","postcode.x","postcode.y"
                                                                         , "OrderRank"
                                                                         
))])

trainingSet[, feature.names][is.na(trainingSet[, feature.names])] <- 0 
testingSet[, feature.names][is.na(testingSet[, feature.names])]   <- 0 

########################################################################################################################
require(libFMexe)

formulas = paste("DiabetesDispense ~ ",paste(feature.names, collapse="+"))
(fmla <- as.formula(paste("y ~ ", paste(xnam, collapse= "+"))))



feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("Patient_ID", "DispenseDate" , "CVindices","X13","IsDeferredScript","postcode.x","postcode.y"
                                                                         , "OrderRank"
                                                                         
))])

pred_cv = libFM(X_build[, feature.names],
                X_val[, feature.names], 
                DiabetesDispense ~  Store_ID  +  Prescriber_ID          +               Drug_ID                                     
                + SourceSystem_Code        +  Prescription_Week         +               Dispense_Week                               
                + Drug_Code                +  NHS_Code                  +               Script_Qty                                  
                + Dispensed_Qty            +  MaxDispense_Qty           +               PatientPrice_Amt                            
                + WholeSalePrice_Amt       +  GovernmentReclaim_Amt     +               RepeatsTotal_Qty                            
                + RepeatsLeft_Qty          +  StreamlinedApproval_Code  +               gender                                      
                + year_of_birth            +  StateCode                 +               IsBannerGroup                               
                + MasterProductCode        +  MasterProductFullName     +               BrandName                                   
                + FormCode                 +  StrengthCode              +               PackSizeNumber                              
                + GenericIngredientName    +  EthicalSubCategoryName    +               EthicalCategoryName                         
                + ManufacturerCode         +  ManufacturerName          +               ManufacturerGroupID                         
                + ManufacturerGroupCode    +  ChemistListPrice          +               ATCLevel5Code                               
                + ATCLevel4Code            +  ATCLevel3Code             +               ATCLevel2Code                               
                + ATCLevel1Code            +  ChronicIllness            +               PatientStateCode                            
                + PatientAge               +  lag1_Drug_ID              +               lag1_ChronicIllness                         
                + lag1_Script_Qty          +  lag2_Drug_ID              +               lag2_ChronicIllness                         
                + lag2_Script_Qty          +  lag3_Drug_ID              +               lag3_ChronicIllness                         
                + lag3_Script_Qty          +  lag4_Drug_ID              +               lag4_ChronicIllness                         
                + lag4_Script_Qty          +  lag5_Drug_ID              +               lag5_ChronicIllness                         
                + lag5_Script_Qty          +  lag6_Drug_ID              +               lag6_ChronicIllness                         
                + lag6_Script_Qty          +  lag7_Drug_ID              +               lag7_ChronicIllness                         
                + lag7_Script_Qty          +  lag8_Drug_ID              +               lag8_ChronicIllness                         
                + lag8_Script_Qty          +  lag9_Drug_ID              +               lag9_ChronicIllness                         
                + lag9_Script_Qty          +  lag10_Drug_ID             +               lag10_ChronicIllness                        
                + lag10_Script_Qty         +  lag11_Drug_ID             +               lag11_ChronicIllness                        
                + lag11_Script_Qty         +  lag12_Drug_ID             +               lag12_ChronicIllness                        
                + lag12_Script_Qty         +  lag13_Drug_ID             +               lag13_ChronicIllness                        
                + lag13_Script_Qty         +  lag14_Drug_ID             +               lag14_ChronicIllness                        
                + lag14_Script_Qty         +  lag15_Drug_ID             +               lag15_ChronicIllness                        
                + lag15_Script_Qty         +  lag16_Drug_ID             +               lag16_ChronicIllness                        
                + lag16_Script_Qty         +  lag17_Drug_ID             +               lag17_ChronicIllness                        
                + lag17_Script_Qty         +  lag18_Drug_ID             +               lag18_ChronicIllness                        
                + lag18_Script_Qty         +  lag19_Drug_ID             +               lag19_ChronicIllness                        
                + lag19_Script_Qty         +  lag20_Drug_ID             +               lag20_ChronicIllness                        
                + lag20_Script_Qty         +  lag21_Drug_ID             +               lag21_ChronicIllness                        
                + lag21_Script_Qty         +  lag22_Drug_ID             +               lag22_ChronicIllness                        
                + lag22_Script_Qty         +  lag23_Drug_ID             +               lag23_ChronicIllness                        
                + lag23_Script_Qty         +  lag24_Drug_ID             +               lag24_ChronicIllness                        
                + lag24_Script_Qty         +  lag25_Drug_ID             +               lag25_ChronicIllness                        
                + lag25_Script_Qty         +  lag26_Drug_ID             +               lag26_ChronicIllness                        
                + lag26_Script_Qty         +  lag27_Drug_ID             +               lag27_ChronicIllness                        
                + lag27_Script_Qty         +  lag28_Drug_ID             +               lag28_ChronicIllness                        
                + lag28_Script_Qty         +  lag29_Drug_ID             +               lag29_ChronicIllness                        
                + lag29_Script_Qty         +  lag30_Drug_ID             +               lag30_ChronicIllness                        
                + lag30_Script_Qty         +  lag31_Drug_ID             +               lag31_ChronicIllness                        
                + lag31_Script_Qty         +  lag32_Drug_ID             +               lag32_ChronicIllness                        
                + lag32_Script_Qty         +  lag33_Drug_ID             +               lag33_ChronicIllness                        
                + lag33_Script_Qty         +  lag34_Drug_ID             +               lag34_ChronicIllness                        
                + lag34_Script_Qty         +  lag35_Drug_ID             +               lag35_ChronicIllness                        
                + lag35_Script_Qty         +  lag36_Drug_ID             +               lag36_ChronicIllness                        
                + lag36_Script_Qty         +  lag37_Drug_ID             +               lag37_ChronicIllness                        
                + lag37_Script_Qty         +  lag38_Drug_ID             +               lag38_ChronicIllness                        
                + lag38_Script_Qty         +  lag39_Drug_ID             +               lag39_ChronicIllness                        
                + lag39_Script_Qty         +  lag40_Drug_ID             +               lag40_ChronicIllness                        
                + lag40_Script_Qty         
                + Depression               +  Diabetes                  +               Epilepsy                                    
                           +  Hypertension              +               Immunology                                  
                + Lipids                   +  Osteoporosis              +               Urology                                     
                + ATc1_A                   +  ATc1_B                    +               ATc1_C                                      
                + ATc1_D                   +  ATc1_G                    +               ATc1_H                                      
                + ATc1_J                   +  ATc1_L                    +               ATc1_M                                      
                + ATc1_N                   +  ATc1_NA                   +               ATc1_P                                      
                + ATc1_R                   +  ATc1_S                    +               ATc1_V                                      
                + ATc2_A01                 +  ATc2_A02                  +               ATc2_A03                                    
                + ATc2_A04                 +  ATc2_A05                  +               ATc2_A06                                    
                + ATc2_A07                 +  ATc2_A08                  +               ATc2_A09                                    
                + ATc2_A10                 +  ATc2_A11                  +               ATc2_A12                                    
                + ATc2_A14                 +  ATc2_A16                  +               ATc2_B01                                    
                + ATc2_B02                 +  ATc2_B03                  +               ATc2_B05                                    
                + ATc2_B06                 +  ATc2_C01                  +               ATc2_C02                                    
                + ATc2_C03                 +  ATc2_C04                  +               ATc2_C05                                    
                + ATc2_C07                 +  ATc2_C08                  +               ATc2_C09                                    
                + ATc2_C10                 +      ATc2_D01              +               ATc2_D02                                    
                + ATc2_D04                 +      ATc2_D05              +               ATc2_D06                                    
                + ATc2_D07                 +      ATc2_D08              +               ATc2_D09                                    
                + ATc2_D10                 +      ATc2_D11              +               ATc2_G01                                    
                + ATc2_G02                 +      ATc2_G03              +               ATc2_G04                                    
                + ATc2_H01                 +      ATc2_H02              +               ATc2_H03                                    
                + ATc2_H04                 +      ATc2_H05              +               ATc2_J01                                    
                + ATc2_J02                 +      ATc2_J04              +               ATc2_J05                                    
                + ATc2_J06                 +      ATc2_J07              +               ATc2_L01                                    
                + ATc2_L02                 +      ATc2_L03              +               ATc2_L04                                    
                + ATc2_M01                 +      ATc2_M02              +               ATc2_M03                                    
                + ATc2_M04                 +      ATc2_M05              +               ATc2_N01                                    
                + ATc2_N02                 +      ATc2_N03              +               ATc2_N04                                    
                + ATc2_N05                 +      ATc2_N06              +               ATc2_N07                                    
                + ATc2_NA                  +      ATc2_P01              +               ATc2_P02                                    
                + ATc2_P03                 +      ATc2_R01              +               ATc2_R02                                    
                + ATc2_R03                 +      ATc2_R05              +               ATc2_R06                                    
                + ATc2_S01                 +      ATc2_S02              +               ATc2_V03                                    
                + ATc2_V04                 +      ATc2_V06              +               ATc2_V07                                    
                + SS_A                     +      SS_C                  +               SS_F                                        
                + SS_L                     +      SS_M                  +               SS_P                                        
                + SS_Z                                                                         
                  ,
                task = "r", dim = 1, iter = 100
                , exe_loc = "C:\\Users\\SriPrav\\Documents\\R\\24DSM\\LibFM")

cat("CV Fold-", i, " ", metric, ": ", score(X_val$DiabetesDispense, pred_cv, metric), "\n", sep = "")




RGF.fit = function(Input, Target, prefix = "",Trees=500,L2=0.1,sL2=-1,loss="Log",min_pop=10,ignoreStdout = T)
{
  if (class(Target) == "factor")
    Target = (as.numeric(Target)-1.5)*2
  
  if (sL2 == -1)
    sL2 = L2
  
  trainData = paste(root_directory,"/RGF/RGF_data/full.train.data.x_",prefix, sep="")
  trainY    = paste(root_directory,"/RGF/RGF_data/full.train.data.y_",prefix, sep="")
  
  write.table(Input,  paste(trainData, sep=""), row.names = F, col.names = F)
  write.table(Target, paste(trainY, sep=""), row.names = F, col.names = F)
  
  base = paste("./RGF/rgf1.2/bin/rgf train train_x_fn=",trainData,",train_y_fn=",trainY,",algorithm=RGF",sep="")
  
  strMdl = paste("model_fn_prefix=RGF/RGF_data/out/mdl_",prefix,sep="")
  strL2 = paste("reg_L2",L2,sep="=")
  strsL2 = paste("reg_sL2",sL2,sep="=")
  strTrees = paste("max_leaf_forest",Trees,sep="=")
  strSave = paste("test_interval",Trees,sep="=")
  strLoss = paste("loss",loss,sep="=")
  strPop = paste("min_pop",min_pop,sep="=")
  cmd.rgf = paste(base,strMdl,strL2,strsL2,strTrees,strSave,strLoss,strPop, sep = ",")
  
  system(cmd.rgf,ignore.stdout=ignoreStdout)
  
  return(1)
}

# use data frame from RGFSourceData
# factor target to get classification



RGF.pred = function(Input,prefix = "",ignoreStdout = T)
{
  testData = paste(root_directory,"/RGF/RGF_data/full.test.data.x_",prefix, sep="")
  write.table(Input, paste(testData, sep=""), row.names = F, col.names = F)
  
  base = paste("./RGF/rgf1.2/bin/rgf predict test_x_fn=",testData,sep="")
  
  strPred = paste("prediction_fn=RGF/RGF_data/out/full.pred_",prefix,sep="")
  strMdl = paste("model_fn=RGF/RGF_data/out/mdl_",prefix,"-01",sep="")
  
  cmd.rgf = paste(base,strPred,strMdl,sep=",")
  
  system(cmd.rgf,ignore.stdout=ignoreStdout)
  
  pred = read.table(paste(root_directory,"/RGF/RGF_data/out/full.pred_", prefix, sep=""))$V1
  
  return(pred)
}
# X_build$DiabetesDispense <- as.factor(X_build$DiabetesDispense)
# RGF.fit(X_build[,feature.names],X_build$DiabetesDispense)
# val_pred <- RGF.pred(X_val[, feature.names])
# cat("CV Fold-", i, " ", metric, ": ", score(X_val$DiabetesDispense, val_pred, metric), "\n", sep = "")

########################################################################################################################
cv          = 5
bags        = 1
seed        = 2017
i = 1
########################################################################################################################
set.seed(seed)
for (i in 1:cv)
  
{
  cat(i ,"fold Processing\n")
  seed        = 2017
  cat("X_build fold Processing\n")
  X_build <- subset(trainingSet, CVindices != i, select = -c( CVindices))
  cat("X_val fold Processing\n")
  X_val   <- subset(trainingSet, CVindices == i, select = -c( CVindices)) 
  X_build$DiabetesDispense <- as.factor(X_build$DiabetesDispense)
  
  val_ensemble <- rep(0, nrow(X_val[,feature.names]))
  test_ensemble <- rep(0, nrow(testingSet[,feature.names]))
  
  for (b in 1:bags) {
    seed = seed + b
    set.seed(seed)
    cat(seed, "- seed")
    cat(b ," - bag Processing\n")
    cat("X_build training Processing\n")
    
    RGF.fit(X_build[,feature.names],X_build$DiabetesDispense)
    val_pred <- RGF.pred(X_val[, feature.names])
    
    cat("X_val prediction Processing\n")
    pred_cv          <- RGF.pred(X_val[, feature.names])
    cat("CV Fold-", i, "bag - ",b, " ", metric, ": ", score(X_val$DiabetesDispense, pred_cv, metric), "\n", sep = "")
    val_ensemble     <- val_ensemble + pred_cv
    
    
    cat("CV TestingSet prediction Processing\n")
    pred_test         <- RGF.pred(testingSet[,feature.names])
    test_ensemble     <- test_ensemble + pred_test
  }
  val_ensemble     <- val_ensemble / bags
  test_ensemble     <- test_ensemble / bags
  
  cat("CV Fold-", i, " ", metric, ": ", score(X_val$DiabetesDispense, val_ensemble, metric), "\n", sep = "")
  
  val_predictions <- data.frame(Patient_ID=X_val$Patient_ID, Diabetes = val_ensemble)
  test_predictions <- data.frame(Patient_ID=testingSet$Patient_ID, Diabetes = test_ensemble)
  
  if(i == 1)
  {
    write.csv(val_predictions,  './submissions/prav.rgf30.fold1.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.rgf30.fold1-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 2)
  {
    write.csv(val_predictions,  './submissions/prav.rgf30.fold2.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.rgf30.fold2-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 3)
  {
    write.csv(val_predictions,  './submissions/prav.rgf30.fold3.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.rgf30.fold3-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 4)
  {
    write.csv(val_predictions,  './submissions/prav.rgf30.fold4.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.rgf30.fold4-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 5)
  {
    write.csv(val_predictions,  './submissions/prav.rgf30.fold5.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.rgf30.fold5-test.csv', row.names=FALSE, quote = FALSE)
  }
  
}

##########################################################################################################
# Full training


# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

fulltest_ensemble <- rep(0, nrow(testingSet[,feature.names]))
trainingSet$DiabetesDispense <- as.factor(trainingSet$DiabetesDispense)
seed        = 2017
for (b in 1:bags) {
  seed = seed + b
  cat(seed, "- seed")
  set.seed(seed)
  cat(b ," - bag Processing\n")
  cat("Bagging Full TrainingSet training\n")
  RGF.fit(trainingSet[,feature.names],trainingSet$DiabetesDispense)
  cat("Bagging Full Model prediction Processing\n")
  
  predfull_test         <- RGF.pred(testingSet[,feature.names])
  fulltest_ensemble     <- fulltest_ensemble + predfull_test
  
}

fulltest_ensemble     <- fulltest_ensemble / bags
testfull_predictions  <- data.frame(Patient_ID=testingSet$Patient_ID, Diabetes = fulltest_ensemble)
write.csv(testfull_predictions, './submissions/prav.rgf30.full.csv', row.names=FALSE, quote = FALSE)

##########################################################################################################


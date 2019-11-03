#############################################################################################################################
# save(build_set,file="build_set.Rda")
# load("build_set.Rda")
#############################################################################################################################



feature.names     <- names(build_set[,-which(names(build_set) %in% c( "DispenseDate", "X13","IsDeferredScript"
                                                                      
                                                                      ,"Prescription_Week"                         ,  "Dispense_Week"                               
                                                                      ,"Drug_Code"                                 ,  "NHS_Code"                                    
                                                                      ,"Script_Qty"                                ,  "Dispensed_Qty"                               
                                                                      , "MaxDispense_Qty"                          ,   "PatientPrice_Amt"                            
                                                                      , "WholeSalePrice_Amt"                       ,   "GovernmentReclaim_Amt"                       
                                                                      , "RepeatsTotal_Qty"                                                     
                                                                      , "StreamlinedApproval_Code"                                                      
                                                                      , "StateCode"                                                                              
                                                                      , "year_of_birth"                            ,   "postcode.y"                                  
                                                                      , "IsBannerGroup"                            ,   "MasterProductCode"                           
                                                                      , "MasterProductFullName"                    ,   "BrandName"                                   
                                                                      , "FormCode"                                 ,   "StrengthCode"                                
                                                                      , "PackSizeNumber"                           ,   "GenericIngredientName"                       
                                                                      , "EthicalSubCategoryName"                   ,   "EthicalCategoryName"                         
                                                                      , "ManufacturerCode"                         ,   "ManufacturerName"                            
                                                                      , "ManufacturerGroupID"                      ,   "ManufacturerGroupCode"                       
                                                                      , "ChemistListPrice"                                                        
                                                                      , "OrderRank","ATCLevel5Code", "ATCLevel4Code" , "ATCLevel3Code","ATCLevel2Code"   ,"ATCLevel1Code" ,"SourceSystem_Code"
                                                                      
))])

write.csv(build_set[,feature.names], "./input/build_set_FE_01.csv", row.names=FALSE, quote = FALSE)
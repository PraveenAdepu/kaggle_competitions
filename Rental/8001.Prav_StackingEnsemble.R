


pred  <- read.csv("./ToShipping/sigma_stack_pred56.csv", header=FALSE)
test  <- read.csv("./ToShipping/test_stacknet56.csv", header=FALSE)

Pred_file <- test[1]
names(train)
names(test)
head(pred)
head(Pred_file)

Pred_file <- cbind(Pred_file, pred)

head()
names(Pred_file) <- c("listing_id","high","medium","low")

write.csv(Pred_file, "./submissions/Prav_Stacking56.csv", row.names = FALSE)

##################################################################################################

StackNet01    <- read_csv( "./submissions/Prav_Stacking55.csv")
StackNet02    <- read_csv( "./submissions/Prav_Stacking56.csv")


head(StackNet01)
head(StackNet02)
head(StackNetData)


Ensemble <- inner_join(StackNet01, StackNet02, by = "listing_id")

head(Ensemble)

features <- setdiff(names(Ensemble),"listing_id")

cor(Ensemble[, features])

Ensemble$high   <- (Ensemble$high.x   + Ensemble$high.y )   /2
Ensemble$medium <- (Ensemble$medium.x + Ensemble$medium.y ) /2
Ensemble$low    <- (Ensemble$low.x    + Ensemble$low.y )    /2

final.features <- c("listing_id","high","medium","low")

final.ensemble <- Ensemble[, final.features]

head(final.ensemble)

write.csv(final.ensemble, "./submissions/Prav_SN5556.csv", row.names = FALSE)

########################################################################################################

StackNet01    <- read_csv( "./submissions/Prav_SN5152.csv")
StackNet02    <- read_csv( "./submissions/Prav_SN5556.csv")


Ensemble <- inner_join(StackNet01, StackNet02, by = "listing_id")

head(Ensemble)

features <- setdiff(names(Ensemble),"listing_id")

cor(Ensemble[, features])

Ensemble$high   <- (Ensemble$high.x   + Ensemble$high.y )   /2
Ensemble$medium <- (Ensemble$medium.x + Ensemble$medium.y ) /2
Ensemble$low    <- (Ensemble$low.x    + Ensemble$low.y )    /2

final.features <- c("listing_id","high","medium","low")

final.ensemble <- Ensemble[, final.features]

head(final.ensemble)

write.csv(final.ensemble, "./submissions/Prav_SN5152_SN5556.csv", row.names = FALSE)
##########################################################################################################

StackNet01    <- read_csv( "./submissions/Prav_Ensemble_Stacking01_RFNN_xgb1601.csv")
StackNet02    <- read_csv( "./submissions/Prav_EnsembleStack01and02FeatureSets.csv")

StackNetData <- StackNet01#[,2:5]
head(StackNet01)
head(StackNet02)
head(StackNetData)


Ensemble <- inner_join(StackNet01, StackNet02, by = "listing_id")

head(Ensemble)

features <- setdiff(names(Ensemble),"listing_id")

cor(Ensemble[, features])

Ensemble$high   <- (Ensemble$high.x   + Ensemble$high.y )   /2
Ensemble$medium <- (Ensemble$medium.x + Ensemble$medium.y ) /2
Ensemble$low    <- (Ensemble$low.x    + Ensemble$low.y )    /2

final.features <- c("listing_id","high","medium","low")

final.ensemble <- Ensemble[, final.features]

head(final.ensemble)

write.csv(final.ensemble, "./submissions/Prav_EnsembleStack0102_Stacking01_RFNN_xgb1601.csv", row.names = FALSE)

########################################################################################################
########################################################################################################

########################################################################################################

StackNet01    <- read_csv( "./submissions/Prav_Ensemble_Stacking01_RFNN.csv") #0.53705
StackNet02    <- read_csv( "./submissions/Prav.xgb17.full.csv") #0.53597

StackNetData <- StackNet01#[,2:5]
head(StackNet01)
head(StackNet02)
head(StackNetData)


Ensemble <- inner_join(StackNet01, StackNet02, by = "listing_id")

head(Ensemble)

features <- setdiff(names(Ensemble),"listing_id")

cor(Ensemble[, features])

Ensemble$high   <- (Ensemble$high.x   + Ensemble$high.y )   /2
Ensemble$medium <- (Ensemble$medium.x + Ensemble$medium.y ) /2
Ensemble$low    <- (Ensemble$low.x    + Ensemble$low.y )    /2

final.features <- c("listing_id","high","medium","low")

final.ensemble <- Ensemble[, final.features]

head(final.ensemble)

write.csv(final.ensemble, "./submissions/Prav_Ensemble_Stacking01_RFNN_xgb17.csv", row.names = FALSE)
##########################################################################################################

StackNet01    <- read_csv( "./submissions/Prav_Ensemble_Stacking01_RFNN_xgb17.csv") #0.52391
StackNet02    <- read_csv( "./submissions/Prav_EnsembleStack01and02FeatureSets.csv") #0.52556

StackNetData <- StackNet01#[,2:5]
head(StackNet01)
head(StackNet02)
head(StackNetData)


Ensemble <- inner_join(StackNet01, StackNet02, by = "listing_id")

head(Ensemble)

features <- setdiff(names(Ensemble),"listing_id")

cor(Ensemble[, features])

Ensemble$high   <- (Ensemble$high.x   + Ensemble$high.y )   /2
Ensemble$medium <- (Ensemble$medium.x + Ensemble$medium.y ) /2
Ensemble$low    <- (Ensemble$low.x    + Ensemble$low.y )    /2

final.features <- c("listing_id","high","medium","low")

final.ensemble <- Ensemble[, final.features]

head(final.ensemble)

write.csv(final.ensemble, "./submissions/Prav_EnsembleStack0102_Stacking01_RFNN_xgb17.csv", row.names = FALSE)

########################################################################################################

########################################################################################################

StackNet01    <- read_csv( "./submissions/Prav_Ensemble_Stacking01_RFNN.csv") # 0.537
StackNet02    <- read_csv( "./submissions/Prav_stacknet04.csv") # 0.532

StackNetData <- StackNet01#[,2:5]
head(StackNet01)
head(StackNet02)
head(StackNetData)


Ensemble <- inner_join(StackNet01, StackNet02, by = "listing_id")

head(Ensemble)

features <- setdiff(names(Ensemble),"listing_id")

cor(Ensemble[, features])

Ensemble$high   <- (Ensemble$high.x   + Ensemble$high.y )   /2
Ensemble$medium <- (Ensemble$medium.x + Ensemble$medium.y ) /2
Ensemble$low    <- (Ensemble$low.x    + Ensemble$low.y )    /2

final.features <- c("listing_id","high","medium","low")

final.ensemble <- Ensemble[, final.features]

head(final.ensemble)

write.csv(final.ensemble, "./submissions/Prav_Ensemble_Stacking01_RFNN_stacknet04.csv", row.names = FALSE)
##########################################################################################################

StackNet01    <- read_csv( "./submissions/Prav_Ensemble_Stacking01_RFNN_stacknet04.csv")
StackNet02    <- read_csv( "./submissions/Prav_EnsembleStack01and02FeatureSets.csv") # 0.532

StackNetData <- StackNet01#[,2:5]
head(StackNet01)
head(StackNet02)
head(StackNetData)


Ensemble <- inner_join(StackNet01, StackNet02, by = "listing_id")

head(Ensemble)

features <- setdiff(names(Ensemble),"listing_id")

cor(Ensemble[, features])

Ensemble$high   <- (Ensemble$high.x   + Ensemble$high.y )   /2
Ensemble$medium <- (Ensemble$medium.x + Ensemble$medium.y ) /2
Ensemble$low    <- (Ensemble$low.x    + Ensemble$low.y )    /2

final.features <- c("listing_id","high","medium","low")

final.ensemble <- Ensemble[, final.features]

head(final.ensemble)

write.csv(final.ensemble, "./submissions/Prav_EnsembleStack0102_Stacking01_RFNN_stacknet04.csv", row.names = FALSE)

########################################################################################################

########################################################################################################

StackNet01    <- read_csv( "./submissions/Prav_Ensemble_Stacking01_RFNN.csv") # 0.537
StackNet02    <- read_csv( "./submissions/Prav_Stacking02.csv") # 0.534

StackNetData <- StackNet01#[,2:5]
head(StackNet01)
head(StackNet02)
head(StackNetData)


Ensemble <- inner_join(StackNet01, StackNet02, by = "listing_id")

head(Ensemble)

features <- setdiff(names(Ensemble),"listing_id")

cor(Ensemble[, features])

Ensemble$high   <- (Ensemble$high.x   + Ensemble$high.y )   /2
Ensemble$medium <- (Ensemble$medium.x + Ensemble$medium.y ) /2
Ensemble$low    <- (Ensemble$low.x    + Ensemble$low.y )    /2

final.features <- c("listing_id","high","medium","low")

final.ensemble <- Ensemble[, final.features]

head(final.ensemble)

write.csv(final.ensemble, "./submissions/Prav_Ensemble_Stacking01_RFNN_stacknet02.csv", row.names = FALSE) # 0.525
##########################################################################################################

StackNet01    <- read_csv( "./submissions/Prav_Ensemble_Stacking01_RFNN_stacknet02.csv") # 0.525
StackNet02    <- read_csv( "./submissions/Prav_stacknet04.csv") # 0.532

StackNetData <- StackNet01#[,2:5]
head(StackNet01)
head(StackNet02)
head(StackNetData)


Ensemble <- inner_join(StackNet01, StackNet02, by = "listing_id")

head(Ensemble)

features <- setdiff(names(Ensemble),"listing_id")

cor(Ensemble[, features])

Ensemble$high   <- (Ensemble$high.x   + Ensemble$high.y )   /2
Ensemble$medium <- (Ensemble$medium.x + Ensemble$medium.y ) /2
Ensemble$low    <- (Ensemble$low.x    + Ensemble$low.y )    /2

final.features <- c("listing_id","high","medium","low")

final.ensemble <- Ensemble[, final.features]

head(final.ensemble)

write.csv(final.ensemble, "./submissions/Prav_Stack0102Final_stacknet04.csv", row.names = FALSE)

########################################################################################################
# xgb1601 and xgb20 -- different features
########################################################################################################
StackNet01    <- read_csv( "./submissions/Prav.xgb20.full.csv") # 0.538
StackNet02    <- read_csv( "./submissions/Prav.xgb1601.full.csv") # 0.536


Ensemble <- inner_join(StackNet01, StackNet02, by = "listing_id")
head(Ensemble)

features <- setdiff(names(Ensemble),"listing_id")
cor(Ensemble[, features])
Ensemble$high   <- (Ensemble$high.x   + Ensemble$high.y )   /2
Ensemble$medium <- (Ensemble$medium.x + Ensemble$medium.y ) /2
Ensemble$low    <- (Ensemble$low.x    + Ensemble$low.y )    /2

final.features <- c("listing_id","high","medium","low")

final.ensemble <- Ensemble[, final.features]

head(final.ensemble)

write.csv(final.ensemble, "./submissions/Prav_xgb1601_20.csv", row.names = FALSE)
########################################################################################################
# xgb1601 and xgb20 -- different features
########################################################################################################
########################################################################################################

StackNet01    <- read_csv( "./submissions/Prav_Ensemble_Stacking01_RFNN.csv") #0.53705
StackNet02    <- read_csv( "./submissions/Prav_xgb1601_20.csv") #0.533



Ensemble <- inner_join(StackNet01, StackNet02, by = "listing_id")

head(Ensemble)

features <- setdiff(names(Ensemble),"listing_id")

cor(Ensemble[, features])

Ensemble$high   <- (Ensemble$high.x   + Ensemble$high.y )   /2
Ensemble$medium <- (Ensemble$medium.x + Ensemble$medium.y ) /2
Ensemble$low    <- (Ensemble$low.x    + Ensemble$low.y )    /2

final.features <- c("listing_id","high","medium","low")

final.ensemble <- Ensemble[, final.features]

head(final.ensemble)

write.csv(final.ensemble, "./submissions/Prav_SN01_xgb1601_20.csv", row.names = FALSE)
##########################################################################################################

StackNet01    <- read_csv( "./submissions/Prav_SN01_xgb1601_20.csv") #0.52391
StackNet02    <- read_csv( "./submissions/Prav_EnsembleStack01and02FeatureSets.csv") #0.52556



Ensemble <- inner_join(StackNet01, StackNet02, by = "listing_id")

head(Ensemble)

features <- setdiff(names(Ensemble),"listing_id")

cor(Ensemble[, features])

Ensemble$high   <- (Ensemble$high.x   + Ensemble$high.y )   /2
Ensemble$medium <- (Ensemble$medium.x + Ensemble$medium.y ) /2
Ensemble$low    <- (Ensemble$low.x    + Ensemble$low.y )    /2

final.features <- c("listing_id","high","medium","low")

final.ensemble <- Ensemble[, final.features]

head(final.ensemble)

write.csv(final.ensemble, "./submissions/Prav_SN0102_SN01_xgb1601_20.csv", row.names = FALSE)

########################################################################################################

##########################################################################################################

StackNet01    <- read_csv( "./submissions/Prav_Ensemble_Stacking01_RFNN_stacknet02.csv") # 0.525
StackNet02    <- read_csv( "./submissions/Prav_stacknet05.csv") # 0.532

StackNetData <- StackNet01#[,2:5]
head(StackNet01)
head(StackNet02)
head(StackNetData)


Ensemble <- inner_join(StackNet01, StackNet02, by = "listing_id")

head(Ensemble)

features <- setdiff(names(Ensemble),"listing_id")

cor(Ensemble[, features])

Ensemble$high   <- (Ensemble$high.x   + Ensemble$high.y )   /2
Ensemble$medium <- (Ensemble$medium.x + Ensemble$medium.y ) /2
Ensemble$low    <- (Ensemble$low.x    + Ensemble$low.y )    /2

final.features <- c("listing_id","high","medium","low")

final.ensemble <- Ensemble[, final.features]

head(final.ensemble)

write.csv(final.ensemble, "./submissions/Prav_Stack0102Final_stacknet05.csv", row.names = FALSE)

########################################################################################################


model1 <- read_csv("./submissions/Prav_Stack0102Final_stacknet05.csv") # 0.52360
model2 <- read_csv("./submissions/prav.SN02_xgb21.csv") # 0.52810
model3 <- read_csv("./submissions/Prav_SN0102_SN01_xgb1601_20.csv") # 0.52412 
model4 <- read_csv("./submissions/Prav_Stack0102Final_stacknet04.csv") # 0.52426
model5 <- read_csv("./submissions/Prav_Ensemble_Stacking01_RFNN_stacknet02.csv") # 0.52568
model6 <- read_csv("./submissions/Prav_EnsembleStack0102_Stacking01_RFNN_stacknet04.csv") # 0.52417
model7 <- read_csv("./submissions/Prav_EnsembleStack0102_Stacking01_RFNN_xgb17.csv") # 0.52391
model8 <- read_csv("./submissions/Prav_EnsembleStack0102_Stacking01_RFNN_xgb1601.csv") # 0.52440
model9 <- read_csv("./submissions/Oldsubs/Prav_EnsembleStack0102_01RFNN.csv") # 0.52483
model10 <- read_csv("./submissions/Oldsubs/Prav_EnsembleStack02and03FeatureSets.csv") # 0.528
model11 <- read_csv("./submissions/Prav_Ensemble_Stacking01_RFNN_xgb15.csv") # 0.52600
model12 <- read_csv("./submissions/Prav_EnsembleStack01and02FeatureSets.csv") # 0.52556
model13 <- read_csv("./submissions/xgb15andRefmodelsStackingV2.csv") # 0.52556


names(model1)  <- c("listing_id","m1_high","m1_medium","m1_low")
names(model2)  <- c("listing_id","m2_high","m2_medium","m2_low")
names(model3)  <- c("listing_id","m3_high","m3_medium","m3_low")
names(model4)  <- c("listing_id","m4_high","m4_medium","m4_low")
names(model5)  <- c("listing_id","m5_high","m5_medium","m5_low")
names(model6)  <- c("listing_id","m6_high","m6_medium","m6_low")
names(model7)  <- c("listing_id","m7_high","m7_medium","m7_low")
names(model8)  <- c("listing_id","m8_high","m8_medium","m8_low")
names(model9)  <- c("listing_id","m9_high","m9_medium","m9_low")
names(model10) <- c("listing_id","m10_high","m10_medium","m10_low")
names(model11) <- c("listing_id","m11_high","m11_medium","m11_low")
names(model12) <- c("listing_id","m12_high","m12_medium","m12_low")
names(model13) <- c("listing_id","m13_high","m13_medium","m13_low")



Ensemble <- inner_join(model1, model2, by = "listing_id")
Ensemble <- inner_join(Ensemble, model3, by = "listing_id")
Ensemble <- inner_join(Ensemble, model4, by = "listing_id")
Ensemble <- inner_join(Ensemble, model5, by = "listing_id")
Ensemble <- inner_join(Ensemble, model6, by = "listing_id")
Ensemble <- inner_join(Ensemble, model7, by = "listing_id")
Ensemble <- inner_join(Ensemble, model8, by = "listing_id")
Ensemble <- inner_join(Ensemble, model9, by = "listing_id")
Ensemble <- inner_join(Ensemble, model10, by = "listing_id")
Ensemble <- inner_join(Ensemble, model11, by = "listing_id")
Ensemble <- inner_join(Ensemble, model12, by = "listing_id")
Ensemble <- inner_join(Ensemble, model13, by = "listing_id")


features <- setdiff(names(Ensemble),"listing_id")

cor(Ensemble[, features])

Ensemble$high   <- (Ensemble$m1_high  + 
                    Ensemble$m2_high  + 
                    Ensemble$m3_high  + 
                    Ensemble$m4_high  + 
                    Ensemble$m5_high  + 
                    Ensemble$m6_high  + 
                    Ensemble$m7_high  + 
                    Ensemble$m8_high  +
                    Ensemble$m9_high  + 
                    Ensemble$m10_high  + 
                    Ensemble$m11_high  + 
                    Ensemble$m12_high  + 
                    Ensemble$m13_high  )   /13

Ensemble$medium   <- (Ensemble$m1_medium  + 
                        Ensemble$m2_medium  + 
                        Ensemble$m3_medium  + 
                        Ensemble$m4_medium  + 
                        Ensemble$m5_medium  + 
                        Ensemble$m6_medium  + 
                        Ensemble$m7_medium  + 
                        Ensemble$m8_medium  +
                        Ensemble$m9_medium  + 
                        Ensemble$m10_medium  + 
                        Ensemble$m11_medium  + 
                        Ensemble$m12_medium  + 
                        Ensemble$m13_medium  )   /13

Ensemble$low   <- (Ensemble$m1_low  + 
                     Ensemble$m2_low  + 
                     Ensemble$m3_low  + 
                     Ensemble$m4_low  + 
                     Ensemble$m5_low  + 
                     Ensemble$m6_low  + 
                     Ensemble$m7_low  + 
                     Ensemble$m8_low  +
                     Ensemble$m9_low  + 
                     Ensemble$m10_low  + 
                     Ensemble$m11_low  + 
                     Ensemble$m12_low  + 
                     Ensemble$m13_low  )   /13
head(Ensemble)

final.features <- c("listing_id","high","medium","low")

final.ensemble <- Ensemble[, final.features]

head(final.ensemble)

write.csv(final.ensemble, "./submissions/Prav_all052models.csv", row.names = FALSE)

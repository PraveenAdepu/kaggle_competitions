testfile <- read_csv("./input/testing40.csv")

idcols <- c("display_id","ad_id")

testingIdCols <- testfile[, idcols]

prob_ffm <- read_csv("./submissions/prav_fullmodel40_ffm.csv", col_names =FALSE)

names(prob_ffm)[1] <- "clicked"

submission_proba_ffm <- cbind(testingIdCols, prob_ffm)
head(submission_proba_ffm,10)
# write_csv(submission_proba_ffm,   './submissions/Prav_FFM40_fulltest_probs.csv')


# submission_proba  <- fread("./submissions/Prav_sub_proba_01.csv") 
# names(submission_proba)
# head(submission_proba,10)

submission_proba_ffm <- as.data.table(submission_proba_ffm)

setorderv( submission_proba_ffm, c("display_id","clicked"), c(1,-1)  );gc() #Sort by -prob
head(submission_proba_ffm,10)

submission_ffm <- submission_proba_ffm[,.(ad_id=paste0(ad_id,collapse=" ")), keyby="display_id" ];gc()#Build submission
#6,245,533
head(submission_ffm,10)

write.csv(submission_ffm,   './submissions/Prav_FFM20_newCV_25features.csv', row.names=FALSE, quote = FALSE)

###############################################################################################################################

# Get the validation file to maintain the columns order with respect to FFM prediction file
# Check same validation file used -- Yes
testfile <- read_csv("./input/training40_valid.csv")

idcols <- c("display_id","ad_id","clicked")

testingIdCols <- testfile[, idcols]

# Check FFM validation prediction file
prob_ffm <- read_csv("./submissions/prav_validationmodel40_ffm.csv", col_names =FALSE)
# Check Validation source and FFM predicitons rowcounts -- Yes
# update column name -- Yes
names(prob_ffm)[1] <- "Prob_clicked"

# column bind of validation source and FFM predictions files, FFM predictions will get in the same order as validation source
submission_proba_ffm <- cbind(testingIdCols, prob_ffm)
# Test the sample header
head(submission_proba_ffm,10)
#submission_proba_ffm$clicked <- NULL
# Write to file to test Ash python MAP score 
# MAP got 0.71 -- ARE YOU PAYING ATTENTION
# Too good to be true, but did all tests

#write_csv(submission_proba_ffm,   './submissions/Prav_FFM40_validationSet_probs.csv')

# Test with Giba R MAP function
# Lets get MAP from Giba function
MAP12 <- function( display_id, clicked, prob  ){
  map12 <- data.table( display_id=display_id, clicked=clicked, prob=prob  )
  map12[ is.na(prob) , prob := mean(prob, na.rm=T)    ]
  setorderv( map12, c("display_id","prob"), c(1,-1)  )
  map12[, count := 1:.N , by="display_id" ]
  return(mean( map12[, sum(clicked/count) , by="display_id" ]$V1 ))
}

# previous sub score LB = 0.684
cat("MAP12 score using Ash CV and Prav 25 raw features without hashing - ", MAP12(submission_proba_ffm$display_id, submission_proba_ffm$clicked, submission_proba_ffm$Prob_clicked))

##################################################################################################################################################################


# Get the validation file to maintain the columns order with respect to FFM prediction file
# Check same validation file used -- Yes
testfile <- read_csv("./input/training20_valid.csv")

idcols <- c("display_id","ad_id","clicked")

testingIdCols <- testfile[, idcols]

# Check FFM validation prediction file
prob_ffm <- read_csv("./submissions/prav_validationmodel22_ffm.csv", col_names =FALSE)
head(prob_ffm)
#prob_ffm1 <- read_csv("./submissions/prav_validationmodel202_ffm.csv", col_names =FALSE)
# Check Validation source and FFM predicitons rowcounts -- Yes
# update column name -- Yes
#names(prob_ffm)[1] <- "clicked"
names(prob_ffm)[1] <- "Prob_clicked"

# column bind of validation source and FFM predictions files, FFM predictions will get in the same order as validation source
submission_proba_ffm <- cbind(testingIdCols, prob_ffm)
#submission_proba_ffm <- cbind(submission_proba_ffm, prob_ffm1)
# Test the sample header
head(submission_proba_ffm,20)
# Write to file to test Ash python MAP score 
# MAP got 0.71 -- ARE YOU PAYING ATTENTION
# Too good to be true, but did all tests
#write_csv(submission_proba_ffm,   './submissions/Prav_FFM22_validationSet_probs.csv')

# Test with Giba R MAP function
# Lets get MAP from Giba function
MAP12 <- function( display_id, clicked, prob  ){
  map12 <- data.table( display_id=display_id, clicked=clicked, prob=prob  )
  map12[ is.na(prob) , prob := mean(prob, na.rm=T)    ]
  setorderv( map12, c("display_id","prob"), c(1,-1)  )
  map12[, count := 1:.N , by="display_id" ]
  return(mean( map12[, sum(clicked/count) , by="display_id" ]$V1 ))
}

cat("MAP12 score using Ash CV and Prav 30  Raw features - FFM22 no hash - ", MAP12(submission_proba_ffm$display_id, submission_proba_ffm$clicked, submission_proba_ffm$Prob_clicked))


##################################################################################################################################


# Get the validation file to maintain the columns order with respect to FFM prediction file
# Check same validation file used -- Yes
testfile <- read_csv("./input/trainingSet20_valid.csv")

idcols <- c("display_id","ad_id","clicked")

testingIdCols <- testfile[, idcols]

# Check FFM validation prediction file
prob_ftrl <- read_csv("./submissions/Prav_FTRL_proba20.csv")
head(prob_ftrl)
#prob_ffm1 <- read_csv("./submissions/prav_validationmodel202_ffm.csv", col_names =FALSE)
# Check Validation source and FFM predicitons rowcounts -- Yes
# update column name -- Yes

#names(prob_ftrl)[3] <- "Prob_clicked1"

# column bind of validation source and FFM predictions files, FFM predictions will get in the same order as validation source
submission_proba_ftrl <- left_join(testingIdCols, prob_ftrl, by =c("display_id","ad_id"))

#submission_proba_ffm <- cbind(submission_proba_ffm, prob_ffm1)
# Test the sample header
head(submission_proba_ftrl,20)
# Write to file to test Ash python MAP score 
# MAP got 0.71 -- ARE YOU PAYING ATTENTION
# Too good to be true, but did all tests
#write_csv(submission_proba_ffm,   './submissions/Prav_FFM20_newCV_25features_validationSet_probs.csv')

# Test with Giba R MAP function
# Lets get MAP from Giba function
MAP12 <- function( display_id, clicked, prob  ){
  map12 <- data.table( display_id=display_id, clicked=clicked, prob=prob  )
  map12[ is.na(prob) , prob := mean(prob, na.rm=T)    ]
  setorderv( map12, c("display_id","prob"), c(1,-1)  )
  map12[, count := 1:.N , by="display_id" ]
  return(mean( map12[, sum(clicked/count) , by="display_id" ]$V1 ))
}

cat("MAP12 score using Ash CV and Prav 25 features - ", MAP12(submission_proba_ftrl$display_id, submission_proba_ftrl$clicked, submission_proba_ftrl$Prob_clicked1))

####################################################################################################################################################################
# Prepare Prob files


# Get the validation file to maintain the columns order with respect to FFM prediction file
# Check same validation file used -- Yes
testfile <- read_csv("./input/testingSet301.csv")

idcols <- c("display_id","ad_id")

testingIdCols <- testfile[, idcols]

# Check FFM validation prediction file
prob_ffm <- read_csv("./submissions/prav_fullmodel31_ffm.csv", col_names =FALSE)
head(prob_ffm)
#prob_ffm1 <- read_csv("./submissions/prav_validationmodel202_ffm.csv", col_names =FALSE)
# Check Validation source and FFM predicitons rowcounts -- Yes
# update column name -- Yes
names(prob_ffm)[1] <- "clicked"
#names(prob_ffm)[1] <- "Prob_clicked"

# column bind of validation source and FFM predictions files, FFM predictions will get in the same order as validation source
submission_proba_ffm <- cbind(testingIdCols, prob_ffm)
#submission_proba_ffm <- cbind(submission_proba_ffm, prob_ffm1)
# Test the sample header
head(submission_proba_ffm,20)
# Write to file to test Ash python MAP score 
# MAP got 0.71 -- ARE YOU PAYING ATTENTION
# Too good to be true, but did all tests
write_csv(submission_proba_ffm,   './submissions/Prav_FFM31_fulltest_probs.csv')







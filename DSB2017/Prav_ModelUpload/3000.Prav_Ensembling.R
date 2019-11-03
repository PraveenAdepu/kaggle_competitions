

setwd("C:/Users/SriPrav/Documents/R/19DSB2017")
root_directory = "C:/Users/SriPrav/Documents/R/19DSB2017"

# rm(list=ls())

require(data.table)
require(Matrix)
require(xgboost)
require(sqldf)
require(plyr)
require(dplyr)
require(ROCR)
require(Metrics)
require(pROC)
require(caret)
require(readr)
require(moments)
require(forecast)

#####################################################################################################


sample_sub <- read_csv("./submissions/stage1_sample_submission.csv")

CNN02.fold1.test <- read_csv("./submissions/prav.bl02.fold1-test.csv")
CNN02.fold2.test <- read_csv("./submissions/prav.bl02.fold2-test.csv")
CNN02.fold3.test <- read_csv("./submissions/prav.bl02.fold3-test.csv")
CNN02.fold4.test <- read_csv("./submissions/prav.bl02.fold4-test.csv")
CNN02.fold5.test <- read_csv("./submissions/prav.bl02.fold5-test.csv")


names(CNN02.fold1.test)[2]<- "fold1cancer"
names(CNN02.fold2.test)[2]<- "fold2cancer"
names(CNN02.fold3.test)[2]<- "fold3cancer"
names(CNN02.fold4.test)[2]<- "fold4cancer"
names(CNN02.fold5.test)[2]<- "fold5cancer"


test_sub <- left_join(sample_sub, CNN02.fold1.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold2.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold3.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold4.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold5.test, by="id")


head(test_sub)

test_sub$cancer <- (test_sub$fold1cancer + test_sub$fold2cancer +test_sub$fold3cancer +test_sub$fold4cancer +test_sub$fold5cancer )/5

cols <- c("id","cancer")

submission <- test_sub[, cols]

write_csv(submission,"./submissions/Prav.FE01.bl02_Meanfolds12345.csv")

head(submission)

##################################################################################################################################################


sample_sub <- read_csv("./submissions/stage1_sample_submission.csv")

CNN02.fold1.test <- read_csv("./submissions/Prav.FE01.nn01.fold1-test.csv")
CNN02.fold2.test <- read_csv("./submissions/Prav.FE01.nn01.fold2-test.csv")
CNN02.fold3.test <- read_csv("./submissions/Prav.FE01.nn01.fold3-test.csv")
CNN02.fold4.test <- read_csv("./submissions/Prav.FE01.nn01.fold4-test.csv")
CNN02.fold5.test <- read_csv("./submissions/Prav.FE01.nn01.fold5-test.csv")


names(CNN02.fold1.test)[2]<- "fold1cancer"
names(CNN02.fold2.test)[2]<- "fold2cancer"
names(CNN02.fold3.test)[2]<- "fold3cancer"
names(CNN02.fold4.test)[2]<- "fold4cancer"
names(CNN02.fold5.test)[2]<- "fold5cancer"


test_sub <- left_join(sample_sub, CNN02.fold1.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold2.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold3.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold4.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold5.test, by="id")


head(test_sub)

test_sub$cancer <- (test_sub$fold1cancer + test_sub$fold2cancer +test_sub$fold3cancer +test_sub$fold4cancer +test_sub$fold5cancer )/5

cols <- c("id","cancer")

submission <- test_sub[, cols]

write_csv(submission,"./submissions/Prav.FE01.nn01_Meanfolds12345.csv")

head(submission)

########################################################################################################################################


sample_sub <- read_csv("./submissions/stage1_sample_submission.csv")

CNN02.fold1.test <- read_csv("./submissions/Prav.FE00.nn01.fold1-test.csv")
CNN02.fold2.test <- read_csv("./submissions/Prav.FE00.nn01.fold2-test.csv")
CNN02.fold3.test <- read_csv("./submissions/Prav.FE00.nn01.fold3-test.csv")
CNN02.fold4.test <- read_csv("./submissions/Prav.FE00.nn01.fold4-test.csv")
CNN02.fold5.test <- read_csv("./submissions/Prav.FE00.nn01.fold5-test.csv")


names(CNN02.fold1.test)[2]<- "fold1cancer"
names(CNN02.fold2.test)[2]<- "fold2cancer"
names(CNN02.fold3.test)[2]<- "fold3cancer"
names(CNN02.fold4.test)[2]<- "fold4cancer"
names(CNN02.fold5.test)[2]<- "fold5cancer"


test_sub <- left_join(sample_sub, CNN02.fold1.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold2.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold3.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold4.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold5.test, by="id")


head(test_sub)

test_sub$cancer <- (test_sub$fold1cancer + test_sub$fold2cancer +test_sub$fold3cancer +test_sub$fold4cancer +test_sub$fold5cancer )/5

cols <- c("id","cancer")

submission <- test_sub[, cols]

write_csv(submission,"./submissions/Prav.FE00.nn01_Meanfolds12345.csv")

head(submission)

########################################################################################################################################


sample_sub <- read_csv("./submissions/stage1_sample_submission.csv")

CNN02.fold1.test <- read_csv("./submissions/prav.FE00.bl02.fold1-test.csv")
CNN02.fold2.test <- read_csv("./submissions/prav.FE00.bl02.fold2-test.csv")
CNN02.fold3.test <- read_csv("./submissions/prav.FE00.bl02.fold3-test.csv")
CNN02.fold4.test <- read_csv("./submissions/prav.FE00.bl02.fold4-test.csv")
CNN02.fold5.test <- read_csv("./submissions/prav.FE00.bl02.fold5-test.csv")


names(CNN02.fold1.test)[2]<- "fold1cancer"
names(CNN02.fold2.test)[2]<- "fold2cancer"
names(CNN02.fold3.test)[2]<- "fold3cancer"
names(CNN02.fold4.test)[2]<- "fold4cancer"
names(CNN02.fold5.test)[2]<- "fold5cancer"


test_sub <- left_join(sample_sub, CNN02.fold1.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold2.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold3.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold4.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold5.test, by="id")


head(test_sub)

test_sub$cancer <- (test_sub$fold1cancer + test_sub$fold2cancer +test_sub$fold3cancer +test_sub$fold4cancer +test_sub$fold5cancer )/5

cols <- c("id","cancer")

submission <- test_sub[, cols]

write_csv(submission,"./submissions/Prav.FE00.bl02_Meanfolds12345.csv")

head(submission)

##################################################################################################################################################

sample_sub <- read_csv("./submissions/stage1_sample_submission.csv")

model1 <- read_csv("./submissions/Prav.FE01.bl02_Meanfolds12345.csv")
model2 <- read_csv("./submissions/Prav.FE01.nn01_Meanfolds12345.csv")

test_sub <- left_join(sample_sub, model1, by="id")
test_sub <- left_join(test_sub,   model2, by="id")
head(test_sub)

names(test_sub)[3] <- "model1.cancer"
names(test_sub)[4] <- "model2.cancer"
names(test_sub)[2] <- "cancer"

test_sub$cancer <- (test_sub$model1.cancer + test_sub$model2.cancer  )/2

cols <- c("id","cancer")

submission <- test_sub[, cols]

write_csv(submission,"./submissions/Prav.FE01.bl02_nn01_meanfolds12345_Mean.csv")
####################################################################################################################################################
## Final submission - Prav select this as final sub
####################################################################################################################################################
sample_sub <- read_csv("./submissions/stage1_sample_submission.csv")

model1 <- read_csv("./submissions/Prav.FE01.bl02_Meanfolds12345.csv")
model2 <- read_csv("./submissions/Prav.FE00.bl02_Meanfolds12345.csv")

test_sub <- left_join(sample_sub, model1, by="id")
test_sub <- left_join(test_sub,   model2, by="id")
head(test_sub)

names(test_sub)[3] <- "model1.cancer"
names(test_sub)[4] <- "model2.cancer"
names(test_sub)[2] <- "cancer"

test_sub$cancer <- (test_sub$model1.cancer + test_sub$model2.cancer  )/2

cols <- c("id","cancer")

submission <- test_sub[, cols]

write_csv(submission,"./submissions/Prav.FE01.bl02_FE00.bl02_meanfolds_Mean.csv") 

####################################################################################################################################################

sample_sub <- read_csv("./submissions/stage1_sample_submission.csv")

model1 <- read_csv("./submissions/Prav.FE01.nn01_Meanfolds12345.csv")
model2 <- read_csv("./submissions/Prav.FE00.nn01_Meanfolds12345.csv")

test_sub <- left_join(sample_sub, model1, by="id")
test_sub <- left_join(test_sub,   model2, by="id")
head(test_sub)

names(test_sub)[3] <- "model1.cancer"
names(test_sub)[4] <- "model2.cancer"
names(test_sub)[2] <- "cancer"

test_sub$cancer <- (test_sub$model1.cancer + test_sub$model2.cancer  )/2

cols <- c("id","cancer")

submission <- test_sub[, cols]

write_csv(submission,"./submissions/Prav.FE01.nn01_FE00.nn01_meanfolds_Mean.csv")

####################################################################################################################################################
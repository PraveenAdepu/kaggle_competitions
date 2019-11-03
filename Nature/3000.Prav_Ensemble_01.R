

setwd("C:/Users/SriPrav/Documents/R/18Nature")
root_directory = "C:/Users/SriPrav/Documents/R/18Nature"


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

#########################################################################################################
sub1 <- read_csv("./submissions/Prav_VGGAugmentation01.csv")
sub2 <- read_csv("./submissions/Prav_CNN01.csv")


test_sub <- left_join(sub1, sub2, by="image")


head(test_sub)

feature.names     <- names(test_sub[,-which(names(test_sub) %in% c("image"))])

cor(test_sub[,feature.names])



test_sub$ALB   <- (test_sub$ALB.x + test_sub$ALB.y)/2
test_sub$BET   <- (test_sub$BET.x + test_sub$BET.y)/2
test_sub$DOL   <- (test_sub$DOL.x + test_sub$DOL.y)/2
test_sub$LAG   <- (test_sub$LAG.x + test_sub$LAG.y)/2
test_sub$NoF   <- (test_sub$NoF.x + test_sub$NoF.y)/2
test_sub$OTHER <- (test_sub$OTHER.x + test_sub$OTHER.y)/2
test_sub$SHARK <- (test_sub$SHARK.x + test_sub$SHARK.y)/2
test_sub$YFT   <- (test_sub$YFT.x + test_sub$YFT.y)/2


cols <- c("image","ALB","BET","DOL","LAG","NoF","OTHER","SHARK","YFT")

submission <- test_sub[, cols]
write_csv(submission,"./submissions/Prav_VGGAug01_CNN01_Ensemble.csv")
####################################################################################################################
## Final submission - Prav use this as final submission
####################################################################################################################

sub1 <- read_csv("./submissions/Prav_VGGAugmentation01.csv")
sub2 <- read_csv("./submissions/Prav_CNN02.csv")


test_sub <- left_join(sub1, sub2, by="image")


head(test_sub)

feature.names     <- names(test_sub[,-which(names(test_sub) %in% c("image"))])

cor(test_sub[,feature.names])



test_sub$ALB   <- (test_sub$ALB.x + test_sub$ALB.y)/2
test_sub$BET   <- (test_sub$BET.x + test_sub$BET.y)/2
test_sub$DOL   <- (test_sub$DOL.x + test_sub$DOL.y)/2
test_sub$LAG   <- (test_sub$LAG.x + test_sub$LAG.y)/2
test_sub$NoF   <- (test_sub$NoF.x + test_sub$NoF.y)/2
test_sub$OTHER <- (test_sub$OTHER.x + test_sub$OTHER.y)/2
test_sub$SHARK <- (test_sub$SHARK.x + test_sub$SHARK.y)/2
test_sub$YFT   <- (test_sub$YFT.x + test_sub$YFT.y)/2


cols <- c("image","ALB","BET","DOL","LAG","NoF","OTHER","SHARK","YFT")

submission <- test_sub[, cols]
write_csv(submission,"./submissions/Prav_VGGAug01_CNN02_Ensemble.csv")
####################################################################################################################

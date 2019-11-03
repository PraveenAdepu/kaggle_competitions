X1 <- read_csv("./submissions/Prav_xgb_baseline01.csv")
X2 <- read_csv("./submissions/Prav_xgb_linear01.csv") 
X3 <- read_csv("./submissions/Prav_xgb_dense01.csv")
X4 <- read_csv("./submissions/Prav_xgb_linear02.csv")
X5 <- read_csv("./submissions/Prav_h2o_rf01.csv")
X6 <- read_csv("./submissions/Prav_xgb_tree02.csv")
X7 <- read_csv("./submissions/Prav_h2o_rf02.csv")
X8 <- read_csv("./submissions/Prav_et_01.csv")
X9 <- read_csv("./submissions/Prav_xgb_tree03.csv")
X10 <- read_csv("./submissions/Prav_et_02.csv")
X11 <- read_csv("./submissions/Prav_et_03.csv")
X12 <- read_csv("./submissions/Prav_h2o_rf03.csv")


X1 <- arrange(X1,activity_id) 
X2 <- arrange(X2,activity_id) 
X3 <- arrange(X3,activity_id) 
X4 <- arrange(X4,activity_id)
X5 <- arrange(X5,activity_id)
X6 <- arrange(X6,activity_id)
X7 <- arrange(X7,activity_id)
X8 <- arrange(X8,activity_id)
X9 <- arrange(X9,activity_id)
X10 <- arrange(X10,activity_id)
X11 <- arrange(X11,activity_id)
X12 <- arrange(X12,activity_id)

# summary(X5$outcome)


# Normalise
#DP1$probability <- (DP1$probability - min(DP1$probability))/(max(DP1$probability) - min(DP1$probability))

# summary(DP1$probability)
cor(X1$outcome, X2$outcome,  method = "pearson") 
cor(X1$outcome, X3$outcome,  method = "pearson") 
cor(X1$outcome, X4$outcome,  method = "pearson")
cor(X2$outcome, X3$outcome,  method = "pearson") 
cor(X2$outcome, X4$outcome,  method = "pearson") 
cor(X1$outcome, X5$outcome,  method = "pearson")
cor(X1$outcome, X6$outcome,  method = "pearson")
cor(X5$outcome, X7$outcome,  method = "pearson")
cor(X9$outcome, X6$outcome,  method = "pearson")
cor(X8$outcome, X10$outcome,  method = "pearson")
cor(X7$outcome, X11$outcome,  method = "pearson")

cor(cbind(rank(X6$outcome),rank(X9$outcome),cbind(rank(X7$outcome),rank(X12$outcome))))

cor(cbind(rank(X1$outcome)
          ,rank(X2$outcome)
          ,rank(X3$outcome)
          ,rank(X4$outcome)
          ,rank(X5$outcome)
          ,rank(X6$outcome)
          ,rank(X7$outcome)
          ,rank(X8$outcome)
          ,rank(X9$outcome)
          ,rank(X10$outcome) ))

#, rank(X3$probability), rank(X4$probability), rank(X5$probability),rank(X6$probability), rank(X7$probability))) 

#submission <- data.frame(id = X1$id, probability=X1$probability * 0.6 + X2$probability * 0.4)

############################################################################################################
## Rank Average ############################################################################################
############################################################################################################
X1 <- read_csv("./submissions/Prav_xgb_baseline01.csv")
X2 <- read_csv("./submissions/Prav_xgb_linear01.csv") 
X3 <- read_csv("./submissions/Prav_xgb_dense01.csv")
X4 <- read_csv("./submissions/Prav_xgb_linear02.csv")
X5 <- read_csv("./submissions/Prav_h2o_rf01.csv")
X6 <- read_csv("./submissions/Prav_xgb_tree02.csv")


X1 <- arrange(X1,activity_id) 
X2 <- arrange(X2,activity_id) 
X3 <- arrange(X3,activity_id) 
X4 <- arrange(X4,activity_id)
X5 <- arrange(X5,activity_id)
X6 <- arrange(X6,activity_id) 

head(X1)
head(X2)
head(X3)
# X1 <- X1[ order(activity_id) ] 
# X2 <- X2[ order(activity_id) ] 
# X3 <- X3[ order(activity_id) ] 
# X4 <- X4[ order(id) ] 
# X5 <- X5[ order(id) ] 

#tmp        <- 100*rank(X4$outcome) + 100*rank(X3$outcome)+ 95*rank(X8$outcome) + 90*rank(X2$outcome) + 90*rank(X6$outcome) + 85*rank(X7$outcome) + 75*rank(X5$outcome) + 50*rank(X1$outcome)
#tmp         <- 100*rank(X4$outcome) + 100*rank(X3$outcome)+ 100*rank(X8$outcome)  + 90*rank(X7$outcome) + 95*rank(X9$outcome)
#tmp         <- 100*rank(X4$outcome) + 100*rank(X3$outcome)+ 100*rank(X10$outcome)  + 90*rank(X7$outcome) + 95*rank(X9$outcome)
tmp         <- 100*rank(X4$outcome) + 100*rank(X3$outcome)+ 90*rank(X10$outcome)  + 90*rank(X12$outcome) + 95*rank(X9$outcome)

tmp         <- tmp/max(tmp) 
submission <- data.frame(activity_id = X1$activity_id, outcome= tmp) 
write.csv(submission, file="./submissions/Prav_rank_5baselinemodels_xgbeth2onewfeatures_02.csv",row.names=FALSE)
############################################################################################################
############################################################################################################
X4 <-  read_csv("./submissions/Prav_rank_tree_linear_dense.csv")

X4 <- arrange(X4,activity_id)

cor(X4$outcome, X3$outcome,  method = "pearson")
cor(cbind(rank(X4$outcome),rank(X3$outcome) ))

###########################################################################################################


BenchMark <- read.csv("./submissions/Prav_benchmark_01.csv")
val_sub <- read.csv("./submissions/prav.L2_xgb04.full.csv")

MergeSub <- left_join(BenchMark, val_sub, by ="Patient_ID")

head(MergeSub,10)
tail(MergeSub,10)

summary(MergeSub$Diabetes.y)

MergeSub <- arrange(MergeSub,Diabetes.y)
head(MergeSub,10)
tail(MergeSub,10)
MergeSub$Diabetes_rank <-  rank(MergeSub$Diabetes.y)
MergeSub$Diabetes <- MergeSub$Diabetes_rank / max(MergeSub$Diabetes_rank)

Final_sub <- MergeSub[,c("Patient_ID","Diabetes")]
head(Final_sub,10)
write.csv(Final_sub,  './submissions/prav.L2_xgb04_rank.csv', row.names=FALSE, quote = FALSE)
###########################################################################################################
X1 <- read_csv("./submissions/prav.L2_xgb04.full.csv")
X2 <- read_csv("./submissions/prav.L2_nn01.full.csv")
X3 <- read_csv("./submissions/Prav.L2_xgb03.full.csv")

X1 <- arrange(X1,Patient_ID) 
X2 <- arrange(X2,Patient_ID)
X3 <- arrange(X3,Patient_ID)

head(X1); head(X2); head(X3)

cor(X1$Diabetes, X2$Diabetes ,  method = "pearson") 

cor(cbind(rank(X1$Diabetes)
          ,rank(X2$Diabetes)
          #,rank(X3$Diabetes)
          )
    )

tmp         <- 100*rank(X1$Diabetes) + 100*rank(X2$Diabetes) + 100*rank(X3$Diabetes)

tmp         <- tmp/max(tmp) 
submission <- data.frame(Patient_ID = X1$Patient_ID, Diabetes= tmp) 
head(submission,10)
write.csv(submission, file="./submissions/prav.L2_xgb4nn01_rank.csv",row.names=FALSE)

# X11 <- read_csv("./submissions/Prav_xgb10_rank.csv")
# X12 <- read_csv("./submissions/Prav_xgb11_rank.csv")
# 
# X11 <- arrange(X11,Patient_ID) 
# X12 <- arrange(X12,Patient_ID)
# 
# head(X11,10); head(X12,10); head(MergeSub,10)
###########################################################################################################


BenchMark <- read.csv("./submissions/Prav_benchmark_01.csv")
val_sub <- read.csv("./submissions/prav.L2_xgb08.full.csv")

MergeSub <- left_join(BenchMark, val_sub, by ="Patient_ID")

head(MergeSub,10)
tail(MergeSub,10)

summary(MergeSub$Diabetes.y)

MergeSub <- arrange(MergeSub,Diabetes.y)
head(MergeSub,10)
tail(MergeSub,10)
# write.csv(MergeSub,  './submissions/prav.L2_xgb08_merge.csv', row.names=FALSE, quote = FALSE)

MergeSub$Diabetes_rank <-  rank(MergeSub$Diabetes.y)
MergeSub$Diabetes <- MergeSub$Diabetes_rank / max(MergeSub$Diabetes_rank)

Final_sub <- MergeSub[,c("Patient_ID","Diabetes")]
head(Final_sub,10)
write.csv(Final_sub,  './submissions/prav.L2_xgb08_rank.csv', row.names=FALSE, quote = FALSE)
###########################################################################################################
###########################################################################################################


BenchMark <- read.csv("./submissions/Prav_benchmark_01.csv")
val_sub <- read.csv("./submissions/prav.rf30.full.csv")

MergeSub <- left_join(BenchMark, val_sub, by ="Patient_ID")

head(MergeSub,10)
tail(MergeSub,10)

summary(MergeSub$predict)

MergeSub <- arrange(MergeSub,predict)
head(MergeSub,10)
tail(MergeSub,10)
# write.csv(MergeSub,  './submissions/prav.L2_xgb08_merge.csv', row.names=FALSE, quote = FALSE)

MergeSub$Diabetes_rank <-  rank(MergeSub$Diabetes.y)
MergeSub$Diabetes <- MergeSub$Diabetes_rank / max(MergeSub$Diabetes_rank)

Final_sub <- MergeSub[,c("Patient_ID","Diabetes")]
head(Final_sub,10)
write.csv(Final_sub,  './submissions/prav.L2_xgb08_rank.csv', row.names=FALSE, quote = FALSE)
###########################################################################################################

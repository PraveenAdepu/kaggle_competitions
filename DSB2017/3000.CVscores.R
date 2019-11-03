####################################################################################
# L2 Stacking
####################################################################################

CVindices5folds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")

CNN02.fold1 <- read_csv("./submissions/prav.mxnet.fold1.csv")
CNN02.fold2 <- read_csv("./submissions/prav.mxnet.fold2.csv")
CNN02.fold3 <- read_csv("./submissions/prav.mxnet.fold3.csv")
CNN02.fold4 <- read_csv("./submissions/prav.mxnet.fold4.csv")
CNN02.fold5 <- read_csv("./submissions/prav.mxnet.fold5.csv")
CNN02.full  <- read_csv("./submissions/prav.mxnet.full.csv")
head(CNN02.fold4)

CNN02 <- rbind(CNN02.fold1,CNN02.fold2,CNN02.fold3,CNN02.fold4,CNN02.fold5)

names(CNN02)[1] <- "pred_cancer"

CNN02.folds <- left_join(CVindices5folds, CNN02, by="id")

names(CNN02.folds)
head(CNN02.folds)
cv = 5
Meanscore = 0
for (i in 1:cv)
  
{
  cat(i ,"fold Processing\n")
  
  cat("X_build fold Processing\n")
  fold <- subset(CNN02.folds, CVindices == i)
  cat("CV Fold-", i, " ", metric, ": ", score(fold$cancer, fold$pred_cancer, metric), "\n", sep = "")
  Meanscore = Meanscore + score(fold$cancer, fold$pred_cancer, metric)
}
Meanscore = Meanscore/cv
cat("CV Mean", i, " ", metric, ": ",Meanscore, "\n", sep = "")

# 1 fold Processing
# X_build fold Processing
# CV Fold-1 logloss: 0.5812888
# 2 fold Processing
# X_build fold Processing
# CV Fold-2 logloss: 0.5530259
# 3 fold Processing
# X_build fold Processing
# CV Fold-3 logloss: 0.5763374
# 4 fold Processing
# X_build fold Processing
# CV Fold-4 logloss: 0.6097515
# 5 fold Processing
# X_build fold Processing
# CV Fold-5 logloss: 0.5471109
# > Meanscore = Meanscore/cv
# > cat("CV Mean", i, " ", metric, ": ",Meanscore, "\n", sep = "")
# CV Mean5 logloss: 0.5735029


sample_sub <- read_csv("./submissions/stage1_sample_submission.csv")

CNN02.fold1.test <- read_csv("./submissions/prav.bl02.fold1-test.csv")
CNN02.fold2.test <- read_csv("./submissions/prav.bl02.fold2-test.csv")
CNN02.fold3.test <- read_csv("./submissions/prav.bl02.fold3-test.csv")
CNN02.fold4.test <- read_csv("./submissions/prav.bl02.fold4-test.csv")
CNN02.fold5.test <- read_csv("./submissions/prav.bl02.fold5-test.csv")
CNN02.full  <- read_csv("./submissions/prav.CNN02.full.csv")

names(CNN02.fold1.test)[2]<- "fold1cancer"
names(CNN02.fold2.test)[2]<- "fold2cancer"
names(CNN02.fold3.test)[2]<- "fold3cancer"
names(CNN02.fold4.test)[2]<- "fold4cancer"
names(CNN02.fold5.test)[2]<- "fold5cancer"
names(CNN02.full)[2]<- "fullcancer"


test_sub <- left_join(sample_sub, CNN02.fold1.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold2.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold3.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold4.test, by="id")
test_sub <- left_join(test_sub,   CNN02.fold5.test, by="id")
test_sub <- left_join(test_sub,   CNN02.full, by="id")

head(test_sub)

test_sub$cancer <- (test_sub$fold1cancer + test_sub$fold2cancer +test_sub$fold3cancer +test_sub$fold4cancer +test_sub$fold5cancer )/5

cols <- c("id","cancer")

submission <- test_sub[, cols]

write_csv(submission,"./submissions/Prav_bl02_Meanfolds12345.csv")

head(submission)

CNN02.fold5.full <- read_csv("./submissions/prav.CNN02.full.csv")
head(CNN02.fold5.full)

names(submission)[2]<- "foldscancer"
names(CNN02.fold5.full)[2]<- "fullcancer"

submission <- left_join(submission, CNN02.fold5.full, by="id")
head(submission)

submission$cancer <- (submission$foldscancer + submission$fullcancer) / 2
head(submission)

submission <- submission[, cols]

write_csv(submission,"./submissions/Prav_CNN02_0.5Meanfolds1to5test_0.5full.csv")
#########################################################################################################################################################

sample_sub <- read_csv("./submissions/stage1_sample_submission.csv")

mxnet01.test <- read_csv("./submissions/Prav_baseline02model.csv")
mxnet02.test <- read_csv("./submissions/Prav_baseline02model2.csv")

names(mxnet01.test)[2]<- "fold1cancer"
names(mxnet02.test)[2]<- "fold2cancer"

test_sub <- left_join(sample_sub, mxnet01.test, by="id")
test_sub <- left_join(test_sub,   mxnet02.test, by="id")

head(test_sub)

test_sub$cancer <- (test_sub$fold1cancer + test_sub$fold2cancer) / 2


cols <- c("id","cancer")

submission <- test_sub[, cols]
write_csv(submission,"./submissions/Prav_baseline02.csv")
##########################################################################################################################################################
sample_sub <- read_csv("./submissions/stage1_sample_submission.csv")
xgb <- read_csv("./submissions/Prav_mxnet_mean2bags.csv")
nnet<- read_csv("./submissions/submission_20170208.csv")

head(xgb)
head(nnet)
names(xgb)[2]  <- "xgb_cancer"
names(nnet)[2] <- "nnet_cancer"

test_sub <- left_join(sample_sub, xgb, by="id")
test_sub <- left_join(test_sub,   nnet, by="id")



feature.names     <- names(test_sub[,-which(names(test_sub) %in% c("id","cancer"))])

cor(test_sub[,feature.names])

head(test_sub)
test_sub$cancer <- (test_sub$xgb_cancer + test_sub$nnet_cancer) / 2

cols <- c("id","cancer")

submission <- test_sub[, cols]
write_csv(submission,"./submissions/Prav_xgbnnetfull5bags_Ensemble.csv")

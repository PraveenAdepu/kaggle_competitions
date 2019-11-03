
##########################################################################################
# xgb04 - Start
##########################################################################################

xgb04.fold1      <- read_csv("./submissions/prav.xgb04.fold1-test.csv")
xgb04.fold2      <- read_csv("./submissions/prav.xgb04.fold2-test.csv")
xgb04.fold3      <- read_csv("./submissions/prav.xgb04.fold3-test.csv")
xgb04.fold4      <- read_csv("./submissions/prav.xgb04.fold4-test.csv")
xgb04.fold5      <- read_csv("./submissions/prav.xgb04.fold5-test.csv")
xgb04.full       <- read_csv("./submissions/prav.xgb04.full.csv")

xgb04 <- left_join(xgb04.fold1, xgb04.fold2,      by="id", all.X = TRUE)
xgb04 <- left_join(xgb04      , xgb04.fold3,      by="id", all.X = TRUE)
xgb04 <- left_join(xgb04      , xgb04.fold4,      by="id", all.X = TRUE)
xgb04 <- left_join(xgb04      , xgb04.fold5,      by="id", all.X = TRUE)

xgb04.test <- data.frame(id=xgb04[,1], loss.test=rowMeans(xgb04[,-1]))

xgb04.all <- left_join(xgb04.test, xgb04.full, by = "id", all.x = TRUE)

xgb04.all$Avgloss <- (xgb04.all$loss.test + xgb04.all$loss ) /2
xgb04.all$loss.test <- NULL
xgb04.all$loss      <- NULL

names(xgb04.all)
colnames(xgb04.all)<-c("id","xgb04loss")
names(xgb04.all)

rm(xgb04.fold1, xgb04.fold2, xgb04.fold3, xgb04.fold4, xgb04.fold5, xgb04.full, 
     xgb04.test, xgb04); gc()

##########################################################################################
# xgb04 - End
##########################################################################################



##########################################################################################
# xgb05 - Start
##########################################################################################

xgb05.fold1      <- read_csv("./submissions/prav.xgb05.fold1-test.csv")
xgb05.fold2      <- read_csv("./submissions/prav.xgb05.fold2-test.csv")
xgb05.fold3      <- read_csv("./submissions/prav.xgb05.fold3-test.csv")
xgb05.fold4      <- read_csv("./submissions/prav.xgb05.fold4-test.csv")
xgb05.fold5      <- read_csv("./submissions/prav.xgb05.fold5-test.csv")
xgb05.full       <- read_csv("./submissions/prav.xgb05.full.csv")

xgb05 <- left_join(xgb05.fold1, xgb05.fold2,      by="id", all.X = TRUE)
xgb05 <- left_join(xgb05      , xgb05.fold3,      by="id", all.X = TRUE)
xgb05 <- left_join(xgb05      , xgb05.fold4,      by="id", all.X = TRUE)
xgb05 <- left_join(xgb05      , xgb05.fold5,      by="id", all.X = TRUE)

xgb05.test <- data.frame(id=xgb05[,1], loss.test=rowMeans(xgb05[,-1]))

xgb05.all <- left_join(xgb05.test, xgb05.full, by = "id", all.x = TRUE)

xgb05.all$Avgloss <- (xgb05.all$loss.test + xgb05.all$loss ) /2
xgb05.all$loss.test <- NULL
xgb05.all$loss      <- NULL

names(xgb05.all)
colnames(xgb05.all)<-c("id","xgb05loss")
names(xgb05.all)

rm(xgb05.fold1, xgb05.fold2, xgb05.fold3, xgb05.fold4, xgb05.fold5, xgb05.full, 
   xgb05.test, xgb05); gc()

##########################################################################################
# xgb05 - End
##########################################################################################


##########################################################################################
# xgb06 - Start
##########################################################################################

xgb06.fold1      <- read_csv("./submissions/prav.xgb06.fold1-test.csv")
xgb06.fold2      <- read_csv("./submissions/prav.xgb06.fold2-test.csv")
xgb06.fold3      <- read_csv("./submissions/prav.xgb06.fold3-test.csv")
xgb06.fold4      <- read_csv("./submissions/prav.xgb06.fold4-test.csv")
xgb06.fold5      <- read_csv("./submissions/prav.xgb06.fold5-test.csv")
xgb06.full       <- read_csv("./submissions/prav.xgb06.full.csv")

xgb06 <- left_join(xgb06.fold1, xgb06.fold2,      by="id", all.X = TRUE)
xgb06 <- left_join(xgb06      , xgb06.fold3,      by="id", all.X = TRUE)
xgb06 <- left_join(xgb06      , xgb06.fold4,      by="id", all.X = TRUE)
xgb06 <- left_join(xgb06      , xgb06.fold5,      by="id", all.X = TRUE)

xgb06.test <- data.frame(id=xgb06[,1], loss.test=rowMeans(xgb06[,-1]))

xgb06.all <- left_join(xgb06.test, xgb06.full, by = "id", all.x = TRUE)

xgb06.all$Avgloss <- (xgb06.all$loss.test + xgb06.all$loss ) /2
xgb06.all$loss.test <- NULL
xgb06.all$loss      <- NULL

names(xgb06.all)
colnames(xgb06.all)<-c("id","xgb06loss")
names(xgb06.all)

rm(xgb06.fold1, xgb06.fold2, xgb06.fold3, xgb06.fold4, xgb06.fold5, xgb06.full, 
   xgb06.test, xgb06); gc()

##########################################################################################
# xgb06 - End
##########################################################################################


##########################################################################################
# kerasnnet - Start
##########################################################################################

kerasnnet.fold1      <- read_csv("./submissions/prav.kerasnnet.fold1-test.csv")
kerasnnet.fold2      <- read_csv("./submissions/prav.kerasnnet.fold2-test.csv")
kerasnnet.fold3      <- read_csv("./submissions/prav.kerasnnet.fold3-test.csv")
kerasnnet.fold4      <- read_csv("./submissions/prav.kerasnnet.fold4-test.csv")
kerasnnet.fold5      <- read_csv("./submissions/prav.kerasnnet.fold5-test.csv")
kerasnnet.full       <- read_csv("./submissions/prav.kerasnnet.full.csv")

kerasnnet <- left_join(kerasnnet.fold1, kerasnnet.fold2,      by="id", all.X = TRUE)
kerasnnet <- left_join(kerasnnet      , kerasnnet.fold3,      by="id", all.X = TRUE)
kerasnnet <- left_join(kerasnnet      , kerasnnet.fold4,      by="id", all.X = TRUE)
kerasnnet <- left_join(kerasnnet      , kerasnnet.fold5,      by="id", all.X = TRUE)

kerasnnet.test <- data.frame(id=kerasnnet[,1], loss.test=rowMeans(kerasnnet[,-1]))

kerasnnet.all <- left_join(kerasnnet.test, kerasnnet.full, by = "id", all.x = TRUE)

kerasnnet.all$Avgloss <- (kerasnnet.all$loss.test + kerasnnet.all$loss ) /2
kerasnnet.all$loss.test <- NULL
kerasnnet.all$loss      <- NULL

names(kerasnnet.all)
colnames(kerasnnet.all)<-c("id","kerasnnetloss")
names(kerasnnet.all)

rm(kerasnnet.fold1, kerasnnet.fold2, kerasnnet.fold3, kerasnnet.fold4, kerasnnet.fold5, kerasnnet.full, 
   kerasnnet.test, kerasnnet); gc()

##########################################################################################
# kerasnnet - End
##########################################################################################


##########################################################################################
# mxnet01 - Start
##########################################################################################

mxnet01.fold1      <- read_csv("./submissions/prav.mxnet01.fold1-test.csv")
mxnet01.fold2      <- read_csv("./submissions/prav.mxnet01.fold2-test.csv")
mxnet01.fold3      <- read_csv("./submissions/prav.mxnet01.fold3-test.csv")
mxnet01.fold4      <- read_csv("./submissions/prav.mxnet01.fold4-test.csv")
mxnet01.fold5      <- read_csv("./submissions/prav.mxnet01.fold5-test.csv")
mxnet01.full       <- read_csv("./submissions/prav.mxnet01.full.csv")

mxnet01 <- left_join(mxnet01.fold1, mxnet01.fold2,      by="id", all.X = TRUE)
mxnet01 <- left_join(mxnet01      , mxnet01.fold3,      by="id", all.X = TRUE)
mxnet01 <- left_join(mxnet01      , mxnet01.fold4,      by="id", all.X = TRUE)
mxnet01 <- left_join(mxnet01      , mxnet01.fold5,      by="id", all.X = TRUE)

mxnet01.test <- data.frame(id=mxnet01[,1], loss.test=rowMeans(mxnet01[,-1]))

mxnet01.all <- left_join(mxnet01.test, mxnet01.full, by = "id", all.x = TRUE)

mxnet01.all$Avgloss <- (mxnet01.all$loss.test + mxnet01.all$loss ) /2
mxnet01.all$loss.test <- NULL
mxnet01.all$loss      <- NULL

names(mxnet01.all)
colnames(mxnet01.all)<-c("id","mxnet01loss")
names(mxnet01.all)

rm(mxnet01.fold1, mxnet01.fold2, mxnet01.fold3, mxnet01.fold4, mxnet01.fold5, mxnet01.full, 
   mxnet01.test, mxnet01); gc()

##########################################################################################
# mxnet01 - End
##########################################################################################

##########################################################################################
# kerasnnet.all - Start
##########################################################################################

kerasnnet.all.full  <- read_csv("./submissions/submission_keras_01.csv")

##########################################################################################
# kerasnnet.all - End
##########################################################################################


xgb.Ensemble <- left_join(xgb04.all,    xgb05.all , by = "id", all.x = TRUE)
xgb.Ensemble <- left_join(xgb.Ensemble, xgb06.all , by = "id", all.x = TRUE)

NN.Ensemble <- left_join(kerasnnet.all,    mxnet01.all , by = "id", all.x = TRUE)
NN.Ensemble <- left_join(NN.Ensemble, kerasnnet.all.full , by = "id", all.x = TRUE)
head(xgb.Ensemble)
head(NN.Ensemble)
xgb.Ensemble.all <- data.frame(id=xgb.Ensemble[,1], loss.xgb=rowMeans(xgb.Ensemble[,-1]))
NN.Ensemble.all  <- data.frame(id=NN.Ensemble[,1],  loss.nn=rowMeans(NN.Ensemble[,-1]))
head(xgb.Ensemble.all)
head(NN.Ensemble.all)

rm(xgb04.all, xgb05.all, xgb06.all, kerasnnet.all, kerasnnet.all.full, 
   mxnet01.all ,xgb.Ensemble, NN.Ensemble); gc()

Ensemble.final <- left_join(xgb.Ensemble.all,    NN.Ensemble.all , by = "id", all.x = TRUE)

rm(xgb.Ensemble.all, NN.Ensemble.all); gc()
head(Ensemble.final)
Ensemble.final.all <- data.frame(id=Ensemble.final[,1], loss=rowMeans(Ensemble.final[,-1]))
head(Ensemble.final.all)

rm(Ensemble.final); gc()

write.csv(Ensemble.final.all, './submissions/StackingL2/prav.Ensemble04.csv', row.names=FALSE, quote = FALSE)


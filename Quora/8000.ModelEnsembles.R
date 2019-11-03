model01 <- read_csv("./submissions/Prav.xgb07.full.csv") 
model02 <- read_csv("./submissions/Prav.nn01.full.csv")

all_ensemble <- left_join(model01, model02, by = "test_id")
head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"test_id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate   <- 0.5 * all_ensemble$is_duplicate.x+ 0.5 * all_ensemble$is_duplicate.y

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_xgb07_nn01_Mean.csv', row.names=FALSE, quote = FALSE)


#######################################################################################################################
model01 <- read_csv("./submissions/Prav.et02.full.csv") 
model02 <- read_csv("./submissions/Prav_dn03dn05_Mean.csv")
model03 <- read_csv("./submissions/Prav.xgb07.full.csv")
model04 <- read_csv("./submissions/Prav.nn01.fold1-test.csv")
model05 <- read_csv("./submissions/Prav.nn01.full.csv")


all_ensemble <- left_join(model01, model02, by = "test_id")
all_ensemble <- left_join(all_ensemble, model03, by = "test_id")
all_ensemble <- left_join(all_ensemble, model04, by = "test_id")
all_ensemble <- left_join(all_ensemble, model05, by = "test_id")


ensemble.features <- setdiff(names(all_ensemble),"test_id")
head(all_ensemble)
mean(all_ensemble$is_duplicate.x)
mean(all_ensemble$is_duplicate.y)
mean(all_ensemble$is_duplicate.x.x)
mean(all_ensemble$is_duplicate.y.y)
mean(all_ensemble$is_duplicate)

cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate1    <- 0.5 * (0.5 * all_ensemble$is_duplicate.x+ 0.5 * all_ensemble$is_duplicate.y) + 0.5 * all_ensemble$is_duplicate

cols <- c("test_id","is_duplicate1")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)

names(Ensemble)[2] <- "is_duplicate"

write.csv(Ensemble, './submissions/Prav_xgb07_dn0305_et02_Mean_Mean.csv', row.names=FALSE, quote = FALSE)







######################################################################################################################
######################################################################################################################
# rm(list=ls())

model01 <- read_csv("./submissions/Prav.deepnet03.csv") 
model02 <- read_csv("./submissions/Prav.deepnet05.csv")
model03 <- read_csv("./submissions/Prav.xgb07.full.csv")
model04 <- read_csv("./submissions/Prav_xgb07_dn0305_Mean.csv")
# model05 <- read_csv("./submissions/Prav.deepnet071.csv")


all_ensemble <- left_join(model01, model02, by = "test_id")
all_ensemble <- left_join(all_ensemble, model03, by = "test_id")
all_ensemble <- left_join(all_ensemble, model04, by = "test_id")
all_ensemble <- left_join(all_ensemble, model05, by = "test_id")



ensemble.features <- setdiff(names(all_ensemble),"test_id")

# names(all_ensemble)
# 
# head(model01)
# head(model03)
# head(all_ensemble,5)

cor(all_ensemble[, ensemble.features])

mean(all_ensemble$is_duplicate.x)
mean(all_ensemble$is_duplicate.y)
mean(all_ensemble$is_duplicate)

all_ensemble$dn03_cali <- all_ensemble$is_duplicate.x * (  0.169/0.1804649)
all_ensemble$dn05_cali <- all_ensemble$is_duplicate.y * (  0.169/0.1831928)

ensemble.features <- setdiff(names(all_ensemble),"test_id")


all_ensemble$is_duplicate    <- 0.5 * (0.5 * all_ensemble$dn03_cali+ 0.5 * all_ensemble$dn05_cali) + 0.5 * (all_ensemble$is_duplicate.x.x)

head(all_ensemble)

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_xgb07_dn0305cali_Mean.csv', row.names=FALSE, quote = FALSE)

################################################################################################

xgb03 <- read_csv("./submissions/Prav.xgb08.full.csv")
head(xgb03)

xgb03 <- xgb03[,c(4,1,2,3)]
head(xgb03)
write.csv(xgb03, './submissions/prav.xgb08.full.ordered.csv', row.names=FALSE, quote = FALSE)
###############################################################################################

model01 <- read_csv("./submissions/Prav.dn50.fold1-test.csv") 
model02 <- read_csv("./submissions/Prav.dn50.fold2-test.csv")

all_ensemble <- left_join(model01, model02, by = "test_id")
head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"test_id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate   <- 0.5 * all_ensemble$is_duplicate.x+ 0.5 * all_ensemble$is_duplicate.y

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_dn50_folds12_Mean.csv', row.names=FALSE, quote = FALSE)

###############################################################################################

model01 <- read_csv("./submissions/Prav_dn50_folds12_Mean.csv") 
model02 <- read_csv("./submissions/Prav_xgb07_nn01_Mean.csv")

all_ensemble <- left_join(model01, model02, by = "test_id")
head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"test_id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate   <- 0.5 * all_ensemble$is_duplicate.x+ 0.5 * all_ensemble$is_duplicate.y

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_dn50_xgb07nn01_Means.csv', row.names=FALSE, quote = FALSE)

###############################################################################################

model01 <- read_csv("./submissions/Prav.dn50_1.full.csv") 
model02 <- read_csv("./submissions/Prav.dn50_2.full.csv")

all_ensemble <- left_join(model01, model02, by = "test_id")
head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"test_id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate   <- 0.5 * all_ensemble$is_duplicate.x+ 0.5 * all_ensemble$is_duplicate.y

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_dn50_12.full.Mean.csv', row.names=FALSE, quote = FALSE)

###############################################################################################

model01 <- read_csv("./submissions/Prav_dn50_12.full.Mean.csv") 
model02 <- read_csv("./submissions/Prav_xgb07_nn01_Mean.csv")

all_ensemble <- left_join(model01, model02, by = "test_id")
head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"test_id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate   <- 0.5 * all_ensemble$is_duplicate.x+ 0.5 * all_ensemble$is_duplicate.y

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_dn50_bags_xgb07nn01_Means.csv', row.names=FALSE, quote = FALSE)

###############################################################################################

###############################################################################################

model01 <- read_csv("./submissions/Prav_dn50_12.full.Mean.csv") 
model02 <- read_csv("./submissions/Prav.et50.full.csv")

all_ensemble <- left_join(model01, model02, by = "test_id")
head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"test_id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate   <- 0.5 * all_ensemble$is_duplicate.x+ 0.5 * all_ensemble$is_duplicate.y

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_dn50_12_et50.full.Mean.csv', row.names=FALSE, quote = FALSE)

###############################################################################################
###############################################################################################

model01 <- read_csv("./submissions/Prav_dn50_12_et50.full.Mean.csv") 
model02 <- read_csv("./submissions/Prav_xgb07_nn01_Mean.csv")

all_ensemble <- left_join(model01, model02, by = "test_id")
head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"test_id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate   <- 0.5 * all_ensemble$is_duplicate.x+ 0.5 * all_ensemble$is_duplicate.y

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_dn50et50_xgb07nn01_Means.csv', row.names=FALSE, quote = FALSE)

###############################################################################################

###############################################################################################

model01 <- read_csv("./submissions/Prav_fm01.csv") 
model02 <- read_csv("./submissions/Prav_dn50_12_et50.full.Mean.csv")

all_ensemble <- left_join(model01, model02, by = "test_id")
head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"test_id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate   <- 0.5 * all_ensemble$is_duplicate.x+ 0.5 * all_ensemble$is_duplicate.y

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_dn50et50_fm01_Mean.csv', row.names=FALSE, quote = FALSE)

###############################################################################################

###############################################################################################

model01 <- read_csv("./submissions/Prav.xgb53.full.csv") 
model02 <- read_csv("./submissions/Prav.nn502.full.csv")

all_ensemble <- left_join(model01, model02, by = "test_id")
head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"test_id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate   <- 0.5 * all_ensemble$is_duplicate.x+ 0.5 * all_ensemble$is_duplicate.y

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_xgb53_nn502.csv', row.names=FALSE, quote = FALSE)

###############################################################################################



###############################################################################################

model01 <- read_csv("./submissions/Prav.L2_xgb04.full.csv") 
model02 <- read_csv("./submissions/Prav.L2_xgb05.full.csv")

all_ensemble <- left_join(model01, model02, by = "test_id")
head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"test_id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate   <- 0.5 * all_ensemble$is_duplicate.x+ 0.5 * all_ensemble$is_duplicate.y

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_xgb53_nn502.csv', row.names=FALSE, quote = FALSE)

###############################################################################################

###############################################################################################

model01 <- read_csv("./submissions/Prav.xgb102.full.csv") 
model02 <- read_csv("./submissions/Prav.nn100.full.csv")

all_ensemble <- left_join(model01, model02, by = "test_id")
head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"test_id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate   <- 0.5 * all_ensemble$is_duplicate.x+ 0.5 * all_ensemble$is_duplicate.y

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_xgb102_nn100.csv', row.names=FALSE, quote = FALSE)

###############################################################################################

###############################################################################################

model01 <- read_csv("./submissions/Prav.L2_xgb07.full.csv") 
model02 <- read_csv("./submissions/Prav.deepnet59.csv")
cols <- c("test_id","is_duplicate")
Ensemble <- model02[, cols]
write.csv(Ensemble, './submissions/Prav_dn59.csv', row.names=FALSE, quote = FALSE)

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_xgb102_nn100.csv', row.names=FALSE, quote = FALSE)

all_ensemble <- left_join(model01, model02, by = "test_id")
head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"test_id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate   <- 0.5 * all_ensemble$is_duplicate.x+ 0.5 * all_ensemble$is_duplicate.y

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_xgb102_nn100.csv', row.names=FALSE, quote = FALSE)

###############################################################################################

model01 <- read_csv("./submissions/Prav.deepnet52.csv") 
model02 <- read_csv("./submissions/Prav.deepnet53.csv")
model03 <- read_csv("./submissions/Prav.deepnet54.csv")
model04 <- read_csv("./submissions/Prav.deepnet55.csv")
model05 <- read_csv("./submissions/Prav.deepnet56.csv")
model06 <- read_csv("./submissions/Prav.deepnet57.csv")
model07 <- read_csv("./submissions/Prav.deepnet59.csv")
model08 <- read_csv("./submissions/Prav.deepnet60.csv")
model09 <- read_csv("./submissions/Prav.deepnet61.csv")
model10 <- read_csv("./submissions/Prav.deepnet62.csv")

all_ensemble <- left_join(model01, model02, by = "test_id")
all_ensemble <- left_join(all_ensemble, model03, by = "test_id")
all_ensemble <- left_join(all_ensemble, model04, by = "test_id")
all_ensemble <- left_join(all_ensemble, model05, by = "test_id")
all_ensemble <- left_join(all_ensemble, model06, by = "test_id")
all_ensemble <- left_join(all_ensemble, model07, by = "test_id")
all_ensemble <- left_join(all_ensemble, model08, by = "test_id")
all_ensemble <- left_join(all_ensemble, model09, by = "test_id")
all_ensemble <- left_join(all_ensemble, model10, by = "test_id")

head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"test_id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate <- rowMeans(all_ensemble[, ensemble.features])

#all_ensemble$is_duplicate   <- (all_ensemble$is_duplicate.x+ all_ensemble$is_duplicate.y + all_ensemble$is_duplicate.x.x+ all_ensemble$is_duplicate.y.y+ all_ensemble$is_duplicate.x.x.x+all_ensemble$is_duplicate.y.y.y)/6

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)



write.csv(Ensemble, './submissions/Prav_dnall.csv', row.names=FALSE, quote = FALSE)

###############################################################################################

model01 <- read_csv("./submissions/Prav.L2_xgb11.full.csv") 
model02 <- read_csv("./submissions/Prav_dnall.csv")


all_ensemble <- left_join(model01, model02, by = "test_id")
head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"test_id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate   <- 0.5 * all_ensemble$is_duplicate.x+ 0.5 * all_ensemble$is_duplicate.y

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.L2_xgb11_dnall.csv', row.names=FALSE, quote = FALSE)
###############################################################################################
model01 <- read_csv("./submissions/Prav.L2_xgb08.full.csv") 
model02 <- read_csv("./submissions/Prav.L2_xgb10.full.csv")


all_ensemble <- left_join(model01, model02, by = "test_id")
head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"test_id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_duplicate   <- 0.5 * all_ensemble$is_duplicate.x+ 0.5 * all_ensemble$is_duplicate.y

cols <- c("test_id","is_duplicate")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.L2_xgb0810.csv', row.names=FALSE, quote = FALSE)
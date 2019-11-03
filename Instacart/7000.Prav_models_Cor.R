

xgb01.full           <- read_csv("./submissions/prav.xgb06.full.csv")
sample_submission    <- read_csv("./input/sample_submission.csv")


xgb.fold1           <- read_csv("./submissions/prav.xgb06.fold1-test.csv")
xgb.fold2           <- read_csv("./submissions/prav.xgb06.fold2-test.csv")
xgb.fold3           <- read_csv("./submissions/prav.xgb06.fold3-test.csv")
xgb.fold4           <- read_csv("./submissions/prav.xgb06.fold4-test.csv")
xgb.fold5           <- read_csv("./submissions/prav.xgb06.fold5-test.csv")


names(xgb.fold1)[4] <- "fold1.pred"
names(xgb.fold2)[4] <- "fold2.pred"
names(xgb.fold3)[4] <- "fold3.pred"
names(xgb.fold4)[4] <- "fold4.pred"
names(xgb.fold5)[4] <- "fold5.pred"

all.folds <- left_join(xgb.fold1, xgb.fold2, by = c("user_id","order_id","product_id"))
all.folds <- left_join(all.folds, xgb.fold3, by = c("user_id","order_id","product_id"))
all.folds <- left_join(all.folds, xgb.fold4, by = c("user_id","order_id","product_id"))
all.folds <- left_join(all.folds, xgb.fold5, by = c("user_id","order_id","product_id"))

head(all.folds)

cor(all.folds[,c("fold1.pred"
                 ,"fold2.pred"
                 ,"fold3.pred"
                 ,"fold4.pred"
                 ,"fold5.pred")])

all.folds$folds.pred <- (all.folds$fold1.pred + all.folds$fold2.pred + all.folds$fold3.pred + all.folds$fold4.pred + all.folds$fold5.pred )/5

all.folds <- left_join(all.folds, xgb01.full, by = c("user_id","order_id","product_id"))

cor(all.folds[,c("folds.pred"
                 ,"pred"
                 )])

##############################################################################################################################################
xgb06.full           <- read_csv("./submissions/prav.xgb07.full.csv")
xgb07.full           <- read_csv("./submissions/prediction_lgbm.csv")


all.folds <- left_join(xgb06.full, xgb07.full, by = c("order_id","product_id"))
all.folds$ensemble <- 0.5 * all.folds$pred + 0.5 * all.folds$prediction 

head(all.folds)
cor(all.folds[,c("pred"
                 ,"prediction"
                )])

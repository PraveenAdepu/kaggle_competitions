xgb01 <- read_csv("./submissions/prav.xgb0607091and11Mean.csv") #prav.xgb0607091Mean.csv
xgb02 <- read_csv("./submissions/Prav.xgb15.full.csv")

all_ensemble <- left_join(xgb01, xgb02, by = "listing_id")

ensemble.features <- setdiff(names(all_ensemble),"listing_id")
names(all_ensemble)

cor(all_ensemble[, ensemble.features])

cor(all_ensemble$high.x    ,all_ensemble$high.y     ,method = "pearson")
cor(all_ensemble$medium.x  ,all_ensemble$medium.y   ,method = "pearson")
cor(all_ensemble$low.x     ,all_ensemble$low.y      ,method = "pearson")


cor(all_ensemble$high.x    ,all_ensemble$high.y     ,method = "spearman")
cor(all_ensemble$medium.x  ,all_ensemble$medium.y   ,method = "spearman")
cor(all_ensemble$low.x     ,all_ensemble$low.y      ,method = "spearman")

all_ensemble$high   <- (all_ensemble$high.x+all_ensemble$high.y)/2
all_ensemble$medium <- (all_ensemble$medium.x+all_ensemble$medium.y)/2
all_ensemble$low    <- (all_ensemble$low.x+all_ensemble$low.y)/2


cols <- c("listing_id","high","medium","low")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/prav.xgb0607091and11MeanAndxgb15Mean.csv', row.names=FALSE, quote = FALSE)

################################################################################################

xgb03 <- read_csv("./submissions/Prav.xgb08.full.csv")
head(xgb03)

xgb03 <- xgb03[,c(4,1,2,3)]
head(xgb03)
write.csv(xgb03, './submissions/prav.xgb08.full.ordered.csv', row.names=FALSE, quote = FALSE)

################################################################################################

xgb01 <- read_csv("./submissions/Prav_xgb1601_20.csv") #Prav_xgb1601_20.7z
xgb02 <- read_csv("./submissions/prav.xgb0607091and11MeanAndRef03Mean.csv") # prav.xgb0607091and11MeanAndRef03Mean.csv.7z

all_ensemble <- left_join(xgb01, xgb02, by = "listing_id")

ensemble.features <- setdiff(names(all_ensemble),"listing_id")
names(all_ensemble)

cor(all_ensemble[, ensemble.features])

cor(all_ensemble$high.x    ,all_ensemble$high.y     ,method = "pearson")
cor(all_ensemble$medium.x  ,all_ensemble$medium.y   ,method = "pearson")
cor(all_ensemble$low.x     ,all_ensemble$low.y      ,method = "pearson")


cor(all_ensemble$high.x    ,all_ensemble$high.y     ,method = "spearman")
cor(all_ensemble$medium.x  ,all_ensemble$medium.y   ,method = "spearman")
cor(all_ensemble$low.x     ,all_ensemble$low.y      ,method = "spearman")

all_ensemble$high   <- (all_ensemble$high.x+all_ensemble$high.y)/2
all_ensemble$medium <- (all_ensemble$medium.x+all_ensemble$medium.y)/2
all_ensemble$low    <- (all_ensemble$low.x+all_ensemble$low.y)/2


cols <- c("listing_id","high","medium","low")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/prav.xgb0607091and11MeanAndxgb15Mean.csv', row.names=FALSE, quote = FALSE)


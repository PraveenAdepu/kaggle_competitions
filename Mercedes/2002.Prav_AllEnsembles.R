
#########################################################################################################################

X1 <- read_csv("./submissions/Prav.xgb001.full.csv")  # 1101.81469
X2 <- read_csv("./submissions/Prav.lgbm001.full.csv") # 1101.71536
X3 <- read_csv("./submissions/Prav.rf101.full.csv")   # 1100.56209
X4 <- read_csv("./submissions/Prav.et101.full.csv")   # 1100.82047

X1 <- arrange(X1 ,ID); names(X1)[2] <- "X1y" 
X2 <- arrange(X2 ,ID); names(X2)[2] <- "X2y" 
X3 <- arrange(X3 ,ID); names(X3)[2] <- "X3y" 
X4 <- arrange(X4 ,ID); names(X4)[2] <- "X4y" 


names(X1)

X <- merge(X1,X2, by ="ID", all.X= T)
X <- merge(X, X3, by ="ID", all.X= T)
X <- merge(X, X4, by ="ID", all.X= T)
X <- merge(X, X5, by ="id", all.X= T)
X <- merge(X, X6, by ="id", all.X= T)
X <- merge(X, X7, by ="id", all.X= T)

X <- arrange(X ,ID)
names(X)
cor(X[2:8])
cor(X$loss.x  , X$loss.y  ,method = "pearson")
cor(X$loss.x  , X$loss    ,method = "pearson")
cor(X$loss.y  , X$loss    ,method = "pearson")

ensemble.features <- setdiff(names(X),"ID")
names(X)

head(model01)
head(model02)
head(all_ensemble)

cor(X[, ensemble.features])

rows <- c("loss.x", "loss.y","loss")
X$lossAll <- rowMeans(X[,2:5])
head(X)
X[,2:5] <- NULL


names(X)[2] <- "y"

write.csv(X, paste(root_directory, "/submissions/Prav_4modelsEnsemble.csv", sep=''), row.names = F)

###############################################################################################################


X1 <- read_csv("./submissions/Prav.xgb002.full.csv")  # 1101.81469
X2 <- read_csv("./submissions/Prav_4modelsEnsemble.csv") # 1101.71536


X1 <- arrange(X1 ,ID); names(X1)[2] <- "X1y" 
X2 <- arrange(X2 ,ID); names(X2)[2] <- "X2y" 


names(X1)

X <- merge(X1,X2, by ="ID", all.X= T)


X <- arrange(X ,ID)
names(X)

ensemble.features <- setdiff(names(X),"ID")
names(X)

cor(X[, ensemble.features])

rows <- c("loss.x", "loss.y","loss")
X$lossAll <- rowMeans(X[,2:3])
head(X)
X[,2:3] <- NULL


names(X)[2] <- "y"

write.csv(X, paste(root_directory, "/submissions/Prav_2modelsEnsemble.csv", sep=''), row.names = F)


###############################################################################################################

X4 <- read_csv("./submissions/prav.xgb11_01.full.csv")


X  <- arrange(X ,id) 
X4 <- arrange(X4 ,id)


X5 <- merge(X, X4, by ="id", all.X= T)

X5 <- arrange(X5 ,id)

cor(X5$lossAll  ,X5$loss     ,method = "pearson")

# X$loss <- (X$loss.x + X$loss.y)/2
X5$lossAvg <- X5$lossAll * 0.5 + X5$loss * 0.5

head(X5)

X5$lossAll <- NULL
X5$loss <- NULL

names(X5)[2] <- "loss"



write.csv(X5, paste(root_directory, "/submissions/Prav_xgb11_01_keras3modelmeans.csv", sep=''), row.names = F)


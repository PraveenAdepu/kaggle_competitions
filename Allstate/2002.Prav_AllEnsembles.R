rm(X1, X2, X)

X1 <- read_csv("./submissions/prav.xgb13.full.csv") # prav.xgb13.full.csv
X2 <- read_csv("./submissions/Prav_keras2022anddl2021average.csv") # submission_keras_02.csv

names(X1)
names(X2)

head(X1)
head(X2)

X1 <- arrange(X1 ,id) 
X2 <- arrange(X2 ,id)

X <- merge(X1, X2, by ="id", all.X= T)

X <- arrange(X ,id)

cor(X$loss.x  ,X$loss.y     ,method = "pearson")
cor(X$loss.x  ,X$loss.y     ,method = "spearman")

# X$loss <- (X$loss.x + X$loss.y)/2
X$loss <- X$loss.x * 0.6 + X$loss.y * 0.4

head(X)

X$loss.x <- NULL
X$loss.y <- NULL

write.csv(X, paste(root_directory, "/submissions/Prav_xgb13keras2022anddl2021average.csv", sep=''), row.names = F)

#########################################################################################################################

X1 <- read_csv("./submissions/Prav_xgb13_dl023keras02.csv") # 1101.81469
X2 <- read_csv("./submissions/Prav_XGBkerasdlweigts.csv") # 1101.71536
X3 <- read_csv("./submissions/Prav_xgb13_keras02.csv") # 1100.56209
X4 <- read_csv("./submissions/Prav_top2_Average.csv") # 1100.82047
X5 <- read_csv("./submissions/Prav_xgb11_01_keras3modelmeans.csv") # 1100.95461
X6 <- read_csv("./submissions/Prav_xgb11_01_keras_gpu05_Average.csv") # 1101.12208
X7 <- read_csv("./submissions/Prav_xgb11_01_keras02_Average.csv") # 1100.81437



X1 <- arrange(X1 ,id); names(X1)[2] <- "X1loss" 
X2 <- arrange(X2 ,id); names(X2)[2] <- "X2loss" 
X3 <- arrange(X3 ,id); names(X3)[2] <- "X3loss" 
X4 <- arrange(X4 ,id); names(X4)[2] <- "X4loss" 
X5 <- arrange(X5 ,id); names(X5)[2] <- "X5loss" 
X6 <- arrange(X6 ,id); names(X6)[2] <- "X6loss" 
X7 <- arrange(X7 ,id); names(X7)[2] <- "X7loss" 

names(X1)

X <- merge(X1, X2, by ="id", all.X= T)
X <- merge(X, X3, by ="id", all.X= T)
X <- merge(X, X4, by ="id", all.X= T)
X <- merge(X, X5, by ="id", all.X= T)
X <- merge(X, X6, by ="id", all.X= T)
X <- merge(X, X7, by ="id", all.X= T)

X <- arrange(X ,id)
names(X)
cor(X[2:8])
cor(X$loss.x  , X$loss.y  ,method = "pearson")
cor(X$loss.x  , X$loss    ,method = "pearson")
cor(X$loss.y  , X$loss    ,method = "pearson")

rows <- c("loss.x", "loss.y","loss")
X$lossAll <- rowMeans(X[,2:8])
head(X)
X[,2:8] <- NULL


names(X)[2] <- "loss"

write.csv(X, paste(root_directory, "/submissions/Prav_7modelsEnsemble.csv", sep=''), row.names = F)

###############################################################################################################



X1 <- read_csv("./submissions/Prav_xgb13_keras02.csv") # 1100.56209
X2 <- read_csv("./submissions/Prav_top2_Average.csv") # 1100.82047
X3 <- read_csv("./submissions/Prav_xgb11_01_keras3modelmeans.csv") # 1100.95461
X4 <- read_csv("./submissions/Prav_xgb11_01_keras02_Average.csv") # 1100.81437



X1 <- arrange(X1 ,id); names(X1)[2] <- "X1loss" 
X2 <- arrange(X2 ,id); names(X2)[2] <- "X2loss" 
X3 <- arrange(X3 ,id); names(X3)[2] <- "X3loss" 
X4 <- arrange(X4 ,id); names(X4)[2] <- "X4loss" 


names(X1)

X <- merge(X1, X2, by ="id", all.X= T)
X <- merge(X, X3, by ="id", all.X= T)
X <- merge(X, X4, by ="id", all.X= T)


X <- arrange(X ,id)
names(X)
cor(X[2:5])

X$lossAll <- rowMeans(X[,2:5])
head(X)
X[,2:5] <- NULL


names(X)[2] <- "loss"

write.csv(X, paste(root_directory, "/submissions/Prav_4_1100scoremodelsEnsemble.csv", sep=''), row.names = F)


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


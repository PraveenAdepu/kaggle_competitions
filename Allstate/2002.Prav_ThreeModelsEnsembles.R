

X1 <- read_csv("./submissions/prav.xgb13.full.csv") # prav.dl02_02.full.csv
X2 <- read_csv("./submissions/submission_keras_02.csv") # submission_keras_02.csv
X3 <- read_csv("./submissions/prav.dl02_02.full.csv")

X1 <- arrange(X1 ,id) 
X2 <- arrange(X2 ,id)
X3 <- arrange(X3 ,id)
head(X1)
head(X2)
head(X3)
X <- merge(X1, X2, by ="id", all.X= T)

X <- merge(X, X3, by ="id", all.X= T)

X <- arrange(X ,id)
names(X)
cor(X$loss.x  , X$loss.y  ,method = "pearson")
cor(X$loss.x  , X$loss    ,method = "pearson")
cor(X$loss.y  , X$loss    ,method = "pearson")

# X$loss <- (X$loss.x + X$loss.y)/2
X$Allloss <- X$loss.x * 0.6 + X$loss.y * 0.25 + X$loss * 0.15

head(X)

X$loss.x <- NULL
X$loss.y <- NULL
X$loss <- NULL
names(X)[2] <- "loss"

write.csv(X, paste(root_directory, "/submissions/Prav_XGBkerasdlweigts.csv", sep=''), row.names = F)

#########################################################################################################################

X1 <- read_csv("./submissions/submission_keras_gpu05_02.csv")
X2 <- read_csv("./submissions/submission_keras02_gpu.csv")
X3 <- read_csv("./submissions/submission_keras_02.csv")

X1 <- arrange(X1 ,id) 
X2 <- arrange(X2 ,id)
X3 <- arrange(X3 ,id)

X <- merge(X1, X2, by ="id", all.X= T)

X <- merge(X, X3, by ="id", all.X= T)

X <- arrange(X ,id)
names(X)
cor(X$loss.x  , X$loss.y  ,method = "pearson")
cor(X$loss.x  , X$loss    ,method = "pearson")
cor(X$loss.y  , X$loss    ,method = "pearson")

rows <- c("loss.x", "loss.y","loss")
X$lossAll <- rowMeans(X[,rows])
head(X)
X$loss.x <- NULL
X$loss.y <- NULL
X$loss   <- NULL

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


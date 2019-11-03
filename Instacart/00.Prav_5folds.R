

set.seed(2016)

folds.train <- as.data.frame(unique(train$user_id))
names(folds.train)[1] <- "user_id"

folds <- createFolds(folds.train$user_id, k = 5)

trainingFold01 <- as.data.frame(folds.train[folds$Fold1, ])
trainingFold01$CVindices <- 1

trainingFold02 <- as.data.frame(folds.train[folds$Fold2, ])
trainingFold02$CVindices <- 2

trainingFold03 <- as.data.frame(folds.train[folds$Fold3, ])
trainingFold03$CVindices <- 3

trainingFold04 <- as.data.frame(folds.train[folds$Fold4, ])
trainingFold04$CVindices <- 4

trainingFold05 <- as.data.frame(folds.train[folds$Fold5, ])
trainingFold05$CVindices <- 5

names(trainingFold01)[1] <- "user_id"
names(trainingFold02)[1] <- "user_id"
names(trainingFold03)[1] <- "user_id"
names(trainingFold04)[1] <- "user_id"
names(trainingFold05)[1] <- "user_id"

names(trainingFold05)
trainingFolds <- rbind(trainingFold01, trainingFold02 , trainingFold03, trainingFold04, trainingFold05 )

write.csv(trainingFolds, './CVSchema/Prav_CVindices_5folds.csv', row.names=FALSE, quote = FALSE)



set.seed(2017)

train  <- fread("./input/train.csv") 

names(train)

head(train$ID,5)
# Random shuffle dataset row wise
train <- train[sample(nrow(train)),]

head(train$ID,5)
CVColumns <- c("ID")
train <- as.data.frame(train$ID)
names(train)[1] <- "ID"

folds <- createFolds(train$ID, k = 10)

split_up <- lapply(folds, function(ind, dat) dat[ind,], dat = train)
dim(train)
unlist(lapply(split_up, nrow))

trainingFold01 <- as.data.frame(train[folds$Fold01, ])
trainingFold01$CVindices <- 1

trainingFold02 <- as.data.frame(train[folds$Fold02, ])
trainingFold02$CVindices <- 2

trainingFold03 <- as.data.frame(train[folds$Fold03, ])
trainingFold03$CVindices <- 3

trainingFold04 <- as.data.frame(train[folds$Fold04, ])
trainingFold04$CVindices <- 4

trainingFold05 <- as.data.frame(train[folds$Fold05, ])
trainingFold05$CVindices <- 5

trainingFold06 <- as.data.frame(train[folds$Fold06, ])
trainingFold06$CVindices <- 6

trainingFold07 <- as.data.frame(train[folds$Fold07, ])
trainingFold07$CVindices <- 7

trainingFold08 <- as.data.frame(train[folds$Fold08, ])
trainingFold08$CVindices <- 8

trainingFold09 <- as.data.frame(train[folds$Fold09, ])
trainingFold09$CVindices <- 9

trainingFold10 <- as.data.frame(train[folds$Fold10, ])
trainingFold10$CVindices <- 10


names(trainingFold01)[1] <- "ID"
names(trainingFold02)[1] <- "ID"
names(trainingFold03)[1] <- "ID"
names(trainingFold04)[1] <- "ID"
names(trainingFold05)[1] <- "ID"
names(trainingFold06)[1] <- "ID"
names(trainingFold07)[1] <- "ID"
names(trainingFold08)[1] <- "ID"
names(trainingFold09)[1] <- "ID"
names(trainingFold10)[1] <- "ID"

names(trainingFold05)
trainingFolds <- rbind(trainingFold01, trainingFold02 , trainingFold03, trainingFold04, trainingFold05,
                       trainingFold06, trainingFold07 , trainingFold08, trainingFold09, trainingFold10)

head(train)
head(trainingFolds)

trainingFolds <- trainingFolds[with(trainingFolds, order(ID)), ]



write.csv(trainingFolds, paste(root_directory, "/CVSchema/Prav_CVindices_10folds.csv", sep=''), row.names=FALSE, quote = FALSE)

# Unit testing
unique(trainingFolds$CVindices)
sqldf("SELECT CVindices,  count(*) Count from trainingFolds Group by CVindices")
# CVindices Count
# 1          1   421
# 2          2   421
# 3          3   420
# 4          4   422
# 5          5   421
# 6          6   421
# 7          7   420
# 8          8   422
# 9          9   420
# 10        10   421





set.seed(2016)

train  <- fread("./input/train.csv") 

names(train)

CVColumns <- c("id")
train <- as.data.frame(train$id)

names(train)[1] <- "id"


folds <- createFolds(train$id, k = 5)

split_up <- lapply(folds, function(ind, dat) dat[ind,], dat = train)
dim(train)
unlist(lapply(split_up, nrow))

trainingFold01 <- as.data.frame(train[folds$Fold1, ])
trainingFold01$CVindices <- 1

trainingFold02 <- as.data.frame(train[folds$Fold2, ])
trainingFold02$CVindices <- 2

trainingFold03 <- as.data.frame(train[folds$Fold3, ])
trainingFold03$CVindices <- 3

trainingFold04 <- as.data.frame(train[folds$Fold4, ])
trainingFold04$CVindices <- 4

trainingFold05 <- as.data.frame(train[folds$Fold5, ])
trainingFold05$CVindices <- 5

names(trainingFold01)[1] <- "id"
names(trainingFold02)[1] <- "id"
names(trainingFold03)[1] <- "id"
names(trainingFold04)[1] <- "id"
names(trainingFold05)[1] <- "id"

names(trainingFold05)
trainingFolds <- rbind(trainingFold01, trainingFold02 , trainingFold03, trainingFold04, trainingFold05 )

head(train)
head(trainingFolds)

trainingFolds <- trainingFolds[with(trainingFolds, order(id)), ]



write.csv(trainingFolds, paste(root_directory, "/CVSchema/Prav_CVindices_5folds.csv", sep=''), row.names=FALSE, quote = FALSE)

# Unit testing
unique(trainingFolds$CVindices)
sqldf("SELECT CVindices,  count(*) Count from trainingFolds Group by CVindices")






set.seed(2017)

train  <- read_csv("./input/Prav_Patients_2016_train_target_10.csv") 

head(train)

# Random shuffle dataset row wise
train <- train[sample(nrow(train)),]

folds <- createFolds(train$DiabetesDispense, k = 5)

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


names(trainingFold05)
trainingFolds <- rbind(trainingFold01, trainingFold02 , trainingFold03, trainingFold04, trainingFold05 )

head(train)
head(trainingFolds)



write.csv(trainingFolds, paste(root_directory, "/CVSchema/Prav_CVindices_5folds_10.csv", sep=''), row.names=FALSE, quote = FALSE)

# Unit testing
unique(trainingFolds$CVindices)
sqldf("SELECT CVindices, DiabetesDispense, count(*) Count from trainingFolds Group by CVindices,DiabetesDispense")


# CVindices DiabetesDispense Count
# 1          1                0 45153
# 2          1                1 10687
# 3          2                0 45351
# 4          2                1 10489
# 5          3                0 45262
# 6          3                1 10578
# 7          4                0 45242
# 8          4                1 10598
# 9          5                0 45347
# 10         5                1 10493

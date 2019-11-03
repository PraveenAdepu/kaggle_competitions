

set.seed(2017)

train  <- read_csv("./input/sources/stage1_labels.csv") 

# Random shuffle dataset row wise
train <- train[sample(nrow(train)),]

# table(train$cancer)
# 
# names(train)
# 
# CVColumns <- c("id")
# train <- as.data.frame(train$id)
# 
# names(train)[1] <- "id"


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

# names(trainingFold01)[1] <- "id"
# names(trainingFold02)[1] <- "id"
# names(trainingFold03)[1] <- "id"
# names(trainingFold04)[1] <- "id"
# names(trainingFold05)[1] <- "id"

names(trainingFold05)
trainingFolds <- rbind(trainingFold01, trainingFold02 , trainingFold03, trainingFold04, trainingFold05 )

head(train)
head(trainingFolds)

# trainingFolds <- trainingFolds[with(trainingFolds, order(id)), ]



write.csv(trainingFolds, paste(root_directory, "/CVSchema/Prav_CVindices_5folds.csv", sep=''), row.names=FALSE, quote = FALSE)

# Unit testing
unique(trainingFolds$CVindices)
sqldf("SELECT CVindices, cancer, count(*) Count from trainingFolds Group by CVindices,cancer")


# CVindices cancer Count
# 1          1      0   189
# 2          1      1    69
# 3          2      0   225
# 4          2      1    71
# 5          3      0   211
# 6          3      1    74
# 7          4      0   191
# 8          4      1    81
# 9          5      0   219
# 10         5      1    67

setwd("C:/Users/SriPrav/Documents/R/29Carvana")
root_directory = "C:/Users/SriPrav/Documents/R/29Carvana"

# paste(root_directory, "/input/events.csv", sep='')

# rm(list=ls())
require(data.table)
require(Matrix)
require(sqldf)
require(plyr)
require(dplyr)
require(ROCR)
require(Metrics)
require(pROC)
require(caret)
require(readr)
library(tidyr)
library(stringr)

set.seed(2017)

train.source  <- read_csv("./input/train_masks.csv") 
train.source$rle_mask <- NULL

train <- train.source %>%
            separate(img, c("image01", "image02"), "_")

train.source <- cbind(train.source, train)

head(train)

train <- data.frame(unique(train$image01))

names(train) <- "image01"          
# Random shuffle dataset row wise
train <- train[sample(nrow(train)),]

CVColumns <- c("image01")
train <- data.frame(train)

names(train)[1] <- "image01"


folds <- createFolds(train$image01, k = 10)

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


names(trainingFold01)[1] <- "image01"
names(trainingFold02)[1] <- "image01"
names(trainingFold03)[1] <- "image01"
names(trainingFold04)[1] <- "image01"
names(trainingFold05)[1] <- "image01"
names(trainingFold06)[1] <- "image01"
names(trainingFold07)[1] <- "image01"
names(trainingFold08)[1] <- "image01"
names(trainingFold09)[1] <- "image01"
names(trainingFold10)[1] <- "image01"

names(trainingFold05)
trainingFolds <- rbind(trainingFold01, trainingFold02 , trainingFold03, trainingFold04, trainingFold05,
                       trainingFold06, trainingFold07 , trainingFold08, trainingFold09, trainingFold10)

rm(trainingFold01, trainingFold02 , trainingFold03, trainingFold04, trainingFold05,
      trainingFold06, trainingFold07 , trainingFold08, trainingFold09, trainingFold10)
head(train)
head(trainingFolds)

trainingFolds <- trainingFolds[with(trainingFolds, order(image01)), ]

train.source <- left_join(train.source, trainingFolds, by = "image01")

head(train.source)

train.source$image01 <- NULL
train.source$image02 <- NULL

train_path <- "C:/Users/SriPrav/Documents/R/29Carvana/input/train/"
train_masks_path <- "C:/Users/SriPrav/Documents/R/29Carvana/input/train_masks/"

train.source$train_path <- paste0(train_path,train.source$img)
train.source$train_masks_path <- paste0(train_masks_path,train.source$img)

train.source$train_masks_path <- str_replace_all(train.source$train_masks_path,".jpg","_mask.gif")

write.csv(train.source, paste(root_directory, "/CVSchema/Prav_CVindices_10folds.csv", sep=''), row.names=FALSE, quote = FALSE)

# Unit testing
unique(trainingFolds$CVindices)
sqldf("SELECT CVindices,  count(*) Count from trainingFolds Group by CVindices")
# CVindices Count
# 1          1  4081
# 2          2  4054
# 3          3  4063
# 4          4  3982
# 5          5  4004
# 6          6  4164
# 7          7  4079
# 8          8  4021
# 9          9  4022
# 10        10  4009




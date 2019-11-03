

set.seed(2017)


train.json <- fromJSON("./input/train.json")
# There has to be a better way to do this while getting repeated rows for the "feature" and "photos" columns
train <- data.table(bathrooms=unlist(train.json$bathrooms)
                    ,bedrooms=unlist(train.json$bedrooms)
                    ,building_id=unlist(train.json$building_id)
                    ,created=as.POSIXct(unlist(train.json$created))
                    ,description=unlist(train.json$description) # parse errors
                    ,display_address=unlist(train.json$display_address) # parse errors
                    ,latitude=unlist(train.json$latitude)
                    ,longitude=unlist(train.json$longitude)
                    ,listing_id=unlist(train.json$listing_id)
                    ,manager_id=as.factor(unlist(train.json$manager_id))
                    ,price=unlist(train.json$price)
                    ,interest_level=as.factor(unlist(train.json$interest_level))
                    ,street_adress=unlist(train.json$street_address) # parse errors
                    # ,features=unlist(train.json$features) # parse errors
                    # ,photos=unlist(train.json$photos) # parse errors
)

train <- as.data.frame(train)
cv.columns <- c("listing_id","interest_level")

train <- train[, cv.columns]

# Random shuffle dataset row wise
train <- train[sample(nrow(train)),]

folds <- createFolds(train$interest_level, k = 5)

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
sqldf("SELECT CVindices, interest_level, count(*) Count from trainingFolds Group by CVindices,interest_level")


# CVindices interest_level Count
# 1          1           high   768
# 2          1            low  6857
# 3          1         medium  2246
# 4          2           high   768
# 5          2            low  6857
# 6          2         medium  2246
# 7          3           high   767
# 8          3            low  6857
# 9          3         medium  2245
# 10         4           high   768
# 11         4            low  6856
# 12         4         medium  2246
# 13         5           high   768
# 14         5            low  6857
# 15         5         medium  2246

##########################################################################################################################################
### 10 folds #############################################################################################################################
##########################################################################################################################################


set.seed(2017)


train.json <- fromJSON("./input/train.json")
# There has to be a better way to do this while getting repeated rows for the "feature" and "photos" columns
train <- data.table(bathrooms=unlist(train.json$bathrooms)
                    ,bedrooms=unlist(train.json$bedrooms)
                    ,building_id=unlist(train.json$building_id)
                    ,created=as.POSIXct(unlist(train.json$created))
                    ,description=unlist(train.json$description) # parse errors
                    ,display_address=unlist(train.json$display_address) # parse errors
                    ,latitude=unlist(train.json$latitude)
                    ,longitude=unlist(train.json$longitude)
                    ,listing_id=unlist(train.json$listing_id)
                    ,manager_id=as.factor(unlist(train.json$manager_id))
                    ,price=unlist(train.json$price)
                    ,interest_level=as.factor(unlist(train.json$interest_level))
                    ,street_adress=unlist(train.json$street_address) # parse errors
                    # ,features=unlist(train.json$features) # parse errors
                    # ,photos=unlist(train.json$photos) # parse errors
)

train <- as.data.frame(train)
cv.columns <- c("listing_id","interest_level")

train <- train[, cv.columns]

# Random shuffle dataset row wise
train <- train[sample(nrow(train)),]

folds <- createFolds(train$interest_level, k = 10)

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

# names(trainingFold01)[1] <- "id"
# names(trainingFold02)[1] <- "id"
# names(trainingFold03)[1] <- "id"
# names(trainingFold04)[1] <- "id"
# names(trainingFold05)[1] <- "id"

names(trainingFold05)
trainingFolds <- rbind(trainingFold01, trainingFold02 , trainingFold03, trainingFold04, trainingFold05 , trainingFold06, trainingFold07, trainingFold08, trainingFold09, trainingFold10)

head(train)
head(trainingFolds)

# trainingFolds <- trainingFolds[with(trainingFolds, order(id)), ]

write.csv(trainingFolds, paste(root_directory, "/CVSchema/Prav_CVindices_10folds.csv", sep=''), row.names=FALSE, quote = FALSE)

# Unit testing
unique(trainingFolds$CVindices)
sqldf("SELECT CVindices, interest_level, count(*) Count from trainingFolds Group by CVindices,interest_level")

# CVindices interest_level Count
# 1          1           high   384
# 2          1            low  3429
# 3          1         medium  1123
# 4          2           high   384
# 5          2            low  3429
# 6          2         medium  1123
# 7          3           high   384
# 8          3            low  3428
# 9          3         medium  1123
# 10         4           high   384
# 11         4            low  3428
# 12         4         medium  1123
# 13         5           high   384
# 14         5            low  3429
# 15         5         medium  1123
# 16         6           high   384
# 17         6            low  3428
# 18         6         medium  1123
# 19         7           high   384
# 20         7            low  3428
# 21         7         medium  1122
# 22         8           high   383
# 23         8            low  3428
# 24         8         medium  1123
# 25         9           high   384
# 26         9            low  3428
# 27         9         medium  1123
# 28        10           high   384
# 29        10            low  3429
# 30        10         medium  1123



training <- read_csv("./input/trainingSet12.csv")
trainadCount  <- read_csv("./input/trainadCountFeatures.csv")

names(training)
#87141731
training <- left_join(training, trainadCount, by = c("display_id","ad_id"))

write_csv(training,"./input/training20.csv")

##################################################################################################################
Prav_CVindices_5folds  <- fread("./CVSchema/splits.csv") 

head(Prav_CVindices_5folds)

unique(Prav_CVindices_5folds$is_train)

table(Prav_CVindices_5folds$is_train)

#training <- read_csv("./input/trainingSet20.csv")

training <- left_join(training, Prav_CVindices_5folds, by = "display_id")

#63,502,376
X_build  <- training[training$is_train == 1,]
#23,639,355
X_valid  <- training[training$is_train == 0,]

X_build$is_train <- NULL
X_valid$is_train <- NULL

write_csv(X_build, "./input/training20_train.csv")
write_csv(X_valid, "./input/training20_valid.csv")

###################################################################################################################

testing <- read_csv("./input/testingSet12.csv")
testadCount  <- read_csv("./input/testadCountFeatures.csv")

names(testing)
testing <- left_join(testing, testadCount, by = c("display_id","ad_id"))

write_csv(testing,"./input/testing20.csv")

sapply(testing, class)

###################################################################################################################
#63,502,376
X_build <- read_csv("./input/training20_train.csv")
# 23,639,355
X_valid <- read_csv("./input/training20_valid.csv")

feat_test  <- read_csv("./input/feat_test.csv")
feat_train <- read_csv("./input/feat_train.csv")


summary(X_valid$user_next_document_id)
summary(X_valid$user_next_publisher_id)

X_build <- left_join(X_build, feat_train , by = c("display_id","ad_id") )
X_valid <- left_join(X_valid, feat_train , by = c("display_id","ad_id") )


write_csv(X_build, "./input/training40_train.csv")
write_csv(X_valid, "./input/training40_valid.csv")

rm(X_build,X_valid); gc()

train <- read_csv("./input/training20.csv")
test <- read_csv("./input/testing20.csv")


train <- left_join(train, feat_train , by = c("display_id","ad_id") )
test <- left_join(test, feat_test , by = c("display_id","ad_id") )
names(test)

write_csv(train, "./input/training40.csv")
write_csv(test, "./input/testing40.csv")






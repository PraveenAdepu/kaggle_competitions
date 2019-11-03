
train <- read_csv('./input/train.csv')
test  <- read_csv('./input/test.csv')

CVindices5folds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")
test$loss <- -100

train_test = rbind(train, test)

train_test <- as.data.table(train_test)
new.cat.raw <- c("cat80","cat87","cat57","cat12","cat79","cat10","cat7","cat89","cat2","cat72",
                 "cat81","cat11","cat1","cat13","cat9","cat3","cat16","cat90","cat23","cat36",
                 "cat73","cat103","cat40","cat28","cat111","cat6","cat76","cat50","cat5",
                 "cat4","cat14","cat38","cat24","cat82","cat25")


features_pair <- combn(new.cat.raw, 2, simplify = F)

for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  train_test[, eval(as.name(paste(f1, f2, sep = "_"))) :=
               paste0(train_test[, eval(as.name(f1))], train_test[, eval(as.name(f2))])]
}

# features_pair <- combn(new.cat.raw, 3, simplify = F)
# 
# for(pair in features_pair) {
#   f1 <- pair[1]
#   f2 <- pair[2]
#   f3 <- pair[3]
#   
#   train_test[, eval(as.name(paste(f1, f2, f3, sep = "_"))) :=
#                paste0(train_test[, eval(as.name(f1))], train_test[, eval(as.name(f2))], train_test[, eval(as.name(f3))])]
# }


# 
# LETTERS_AY <- LETTERS[-length(LETTERS)]
# LETTERS702 <- c(LETTERS_AY, sapply(LETTERS_AY, function(x) paste0(x, LETTERS_AY)), "ZZ")
# 
# train_test <- as.data.frame(train_test)
# 
# feature.names     <- names(train_test[,-which(names(train_test) %in% c("id","loss"))])
# 
# for (f in feature.names) {
#   if (class(train_test[[f]])=="character") {
#     levels <- intersect(LETTERS702, unique(train_test[[f]])) # get'em ordered!
#     labels <- match(levels, LETTERS702)
#     #train_test[[f]] <- factor(train_test[[f]], levels=levels) # uncomment this for one-hot
#     train_test[[f]] <- as.integer(as.character(factor(train_test[[f]], levels=levels, labels = labels))) # comment this one away for one-hot
#   }
# }

train_test <- as.data.frame(train_test)

onehot.variables <- grep("cat", names(train_test), value = T)


formula <-  as.formula(paste("~ ", paste(onehot.variables, collapse= "+"))) 
ohe_feats = onehot.variables

dummies <- dummyVars(formula, data = train_test)
train_test_ohe <- as.data.frame(predict(dummies, newdata = train_test))
dim(train_test)
dim(train_test_ohe)
train_test_combined <- cbind(train_test[,-c(which(colnames(train_test) %in% ohe_feats))],train_test_ohe)




cont.features <- grep("con", names(train_test_combined), value = TRUE)
gc()

for (f in cont.features) {
  if (class(train_test_combined[[f]])=="numeric" & (skewness(train_test_combined[[f]]) > 0.25 | skewness(train_test_combined[[f]]) < -0.25)) {
    lambda = BoxCox.lambda( train_test_combined[[f]] )
    skewness = skewness( train_test_combined[[f]] )
    kurtosis = kurtosis( train_test_combined[[f]] )
    cat("VARIABLE : ",f, "lambda : ",lambda, "skewness : ",skewness, "kurtosis : ",kurtosis, "\n")
    train_test_combined[[f]] = BoxCox( train_test_combined[[f]], lambda)
    
  }
}



training <- train_test_combined[train_test_combined$loss != -100,]
testing  <- train_test_combined[train_test_combined$loss == -100,]

rm(train,test); gc()
summary(training$loss)
summary(testing$loss)

testing$loss <- NULL

trainFeatures <- read_csv('./input/trainFeatures.csv')
testFeatures  <- read_csv('./input/testFeatures.csv')

training <- left_join(training, trainFeatures, by = "id")
testing  <- left_join(testing, testFeatures, by = "id")

trainingSet <- left_join(training, CVindices5folds, by = "id")
rm(training); gc()
rm(train_test); gc()
testingSet  <- testing

rm(testing,  CVindices5folds,trainFeatures,testFeatures); gc()

write.csv(trainingSet, paste(root_directory, "/input/train_nnetcombifeatures.csv", sep=''), row.names=FALSE, quote = FALSE)
write.csv(testingSet, paste(root_directory, "/input/test_nnetcombifeatures.csv", sep=''), row.names=FALSE, quote = FALSE)

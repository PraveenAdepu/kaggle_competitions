
train <- read_csv('./input/train.csv')
test  <- read_csv('./input/test.csv')

CVindices5folds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")
test$loss <- -100

trainFeatures <- read_csv('./input/trainFeatures.csv')
testFeatures  <- read_csv('./input/testFeatures.csv')

train <- left_join(train, trainFeatures, by = "id")
test <- left_join(test, testFeatures, by = "id")

train_test = rbind(train, test)

train_test <- as.data.table(train_test)
new.cat.raw <- c("cat80","cat87","cat57","cat12","cat79","cat10","cat7","cat89","cat2","cat72",
                 "cat81","cat11","cat1","cat13","cat9","cat3","cat16","cat90","cat23","cat36",
                 "cat73","cat103","cat40","cat28","cat111","cat6","cat76","cat50","cat5",
                 "cat4","cat14","cat38","cat24","cat82","cat25")


# features_pair <- combn(new.cat.raw, 2, simplify = F)
# 
# for(pair in features_pair) {
#   f1 <- pair[1]
#   f2 <- pair[2]
#   
#   train_test[, eval(as.name(paste(f1, f2, sep = "_"))) :=
#                paste0(train_test[, eval(as.name(f1))], train_test[, eval(as.name(f2))])]
# }

features_pair <- combn(new.cat.raw, 3, simplify = F)

for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  f3 <- pair[3]

  train_test[, eval(as.name(paste(f1, f2, f3, sep = "_"))) :=
               paste0(train_test[, eval(as.name(f1))], train_test[, eval(as.name(f2))], train_test[, eval(as.name(f3))])]
}

combi.3way.impfeatures <- c("id",
            "cat80_cat81_cat1",
            "cat12_cat79_cat111",
            "cont14",
            "cat12_cat79_cat16",
            "cont_allEnergy",
            "cat87_cat10_cat103",
            "cont_energy",
            "cont7",
            "cont2",
            "cat12_cat79_cat72",
            "cat80_cat81_cat82",
            "cat12_cat11_cat103",
            "cat100",
            "cat80_cat81_cat73",
            "cat53",
            "cat114",
            "cont12",
            "cat80_cat12_cat81",
            "cont3",
            "cat112",
            "cat57_cat12_cat79",
            "cont11",
            "cont1",
            "cat113",
            "cat10_cat13_cat103",
            "cont4",
            "cont8",
            "cat80_cat1_cat82",
            "cont6",
            "cat57_cat79_cat2",
            "cont5",
            "cat1_cat73_cat82",
            "cont13",
            "cat12_cat13_cat38",
            "cat57_cat10_cat103",
            "cont9",
            "cont10",
            "cont_energyCont",
            "cat108",
            "cat116",
            "cat57_cat2_cat6",
            "cat110",
            "cat75",
            "cat44",
            "cat12_cat38_cat25",
            "cat57_cat9_cat6",
            "cat80_cat1_cat73",
            "cat10_cat38_cat25",
            "cat87_cat7_cat111",
            "cat26",
            "cat87_cat57_cat103",
            "cat31",
            "cat72_cat13_cat111",
            "cat11_cat90_cat111",
            "cat94",
            "cat79_cat111_cat5",
            "cat3_cat111_cat5",
            "cat39",
            "cat106",				
            "cat57_cat79_cat10",	
            "cat13_cat23_cat111",	
            "cat115",				
            "cat105",				
            "cat91",				
            "cat83",				
            "cat107",				
            "cat99",				
            "cat109",				
            "cat87_cat57_cat111",	
            "cat2_cat111_cat25",	
            "cat72_cat9_cat38",	
            "cat87_cat89_cat11",	
            "cat11_cat103_cat4",	
            "cat89_cat13_cat4",	
            "cat7_cat13_cat111",	
            "cat9_cat103_cat111",	
            "cat49",				
            "cat7_cat11_cat111",	
            "cat80_cat73_cat82",	
            "cat98",				
            "cat1_cat9_cat6",		
            "cat1_cat50_cat82",	
            "cat84",				
            "cat80_cat81_cat103",	
            "cat12_cat16_cat111",	
            "cat57_cat13_cat111",	
            "cat52",				
            "cat9_cat36_cat111",	
            "cat81_cat1_cat73",	
            "cat57_cat72_cat103",	
            "cat95",				
            "cat57_cat111_cat6",	
            "cat9_cat103_cat4",	
            "cat73_cat40_cat50",	
            "cat97",				
            "cat93",				
            "cat104",				
            "cat57_cat79_cat25",	
            "cat66",				
            "cat9_cat38_cat25",	
            "cat27",				
            "cat57_cat76_cat38",	
            "cat1_cat73_cat6",	
            "cat87_cat89_cat103",	
            "cat92",				
            "cat2_cat103_cat4",	
            "cat57_cat79_cat76",	
            "cat81_cat1_cat103",	
            "cat101",				
            "cat12_cat13_cat103",	
            "cat103_cat111_cat6",	
            "cat1_cat73_cat40",	
            "cat81_cat16_cat82",	
            "cat80_cat79_cat81",	
            "cat2_cat103_cat6",	
            "cat87_cat72_cat111",	
            "cat87_cat7_cat76",	
            "cat11_cat13_cat103",	
            "cat57_cat111_cat5",
            "loss"
            )
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

 train_test <- train_test[, combi.3way.impfeatures]
 
onehot.variables <- grep("cat", names(train_test), value = T)


formula <-  as.formula(paste("~ ", paste(onehot.variables, collapse= "+"))) 
ohe_feats = onehot.variables

dummies <- dummyVars(formula, data = train_test)
train_test_ohe <- as.data.frame(predict(dummies, newdata = train_test))
dim(train_test)
dim(train_test_ohe)
train_test_combined <- cbind(train_test[,-c(which(colnames(train_test) %in% ohe_feats))],train_test_ohe)




gc()
cont.features <- grep("con", names(train_test_combined), value = TRUE)

train_test_combined[,cont.features] <- apply(train_test_combined[,cont.features], 2, normalit)
# for (f in cont.features) {
#   if (class(train_test_combined[[f]])=="numeric" & (skewness(train_test_combined[[f]]) > 0.25 | skewness(train_test_combined[[f]]) < -0.25)) {
#     lambda = BoxCox.lambda( train_test_combined[[f]] )
#     skewness = skewness( train_test_combined[[f]] )
#     kurtosis = kurtosis( train_test_combined[[f]] )
#     cat("VARIABLE : ",f, "lambda : ",lambda, "skewness : ",skewness, "kurtosis : ",kurtosis, "\n")
#     train_test_combined[[f]] = BoxCox( train_test_combined[[f]], lambda)
#     
#   }
# }



training <- train_test_combined[train_test_combined$loss != -100,]
testing  <- train_test_combined[train_test_combined$loss == -100,]

rm(train,test); gc()
summary(training$loss)
summary(testing$loss)

testing$loss <- NULL



trainingSet <- left_join(training, CVindices5folds, by = "id")
rm(training); gc()
rm(train_test); gc()
testingSet  <- testing

rm(testing,  CVindices5folds,trainFeatures,testFeatures); gc()

write.csv(trainingSet, paste(root_directory, "/input/train_nnetthreecombifeatures.csv", sep=''), row.names=FALSE, quote = FALSE)
write.csv(testingSet, paste(root_directory, "/input/test_nnetthreecombifeatures.csv", sep=''), row.names=FALSE, quote = FALSE)

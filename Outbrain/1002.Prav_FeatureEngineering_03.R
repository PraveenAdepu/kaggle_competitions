
set.seed(2016)


train  <- fread("./input/clicks_train.csv") 
names(train)

test  <- fread("./input/clicks_test.csv") 
names(test)

events <- fread("./input/events.csv") 
names(events)
promoted_content <- fread("./input/promoted_content.csv") 
names(promoted_content)

documents_meta <- fread("./input/documents_meta.csv") 
names(documents_meta)



Prav_CVindices_5folds  <- fread("./CVSchema/Prav_CVindices_5folds.csv") 
names(Prav_CVindices_5folds)

# train -- 87,141,731
train <- left_join(train, Prav_CVindices_5folds, by = "display_id")
names(train)
train$day <- NULL

test$clicked   <- 0
test$CVindices <- 100

train_test <- rbind(train, test)

head(train_test,2)

# 119,366,893
train_test <- left_join(train_test, events, by = "display_id")
names(train_test)[6]<- "events_document_id"
train_test <- left_join(train_test, promoted_content, by = c("ad_id"))
train_test <- left_join(train_test, documents_meta, by = "document_id")

head(train_test)

#######################################################################################################################

setDT(train_test)[, paste0("location", 1:3) := tstrsplit(geo_location, ">")]


train_test$Date =  as.POSIXct((train_test$timestamp+1465876799998)/1000, origin="1970-01-01", tz="UTC")

t.lub <- ymd_hms(train_test$Date)

h.lub <- hour(t.lub) + minute(t.lub)/60
d.lub <- day(t.lub)
m.lub <- minute(t.lub)/60

head(h.lub)
head(d.lub)
head(m.lub)

train_test$hour    <- h.lub
train_test$day     <- d.lub
train_test$minutes <- m.lub

train_test$day <- as.integer(train_test$day)



train_test$platform           <- as.numeric(train_test$platform)
train_test[is.na(train_test)] <- 0

train_test$geo_location <- NULL
train_test$publish_time <- NULL
train_test$Date         <- NULL
train_test$timestamp    <- NULL

names(train_test)
#########################################################################################################################

training <- train_test[train_test$CVindices != 100,]
testing  <- train_test[train_test$CVindices == 100,]

rm(train,test, promoted_content, documents_categories, documents_entities, documents_meta, documents_topics, events, Prav_CVindices_5folds); gc()


testing$clicked   <- NULL
testing$CVindices <- NULL

#########################################################################################################################  
trainLeak <- read_csv("./input/trainingSet_Leak.csv")
testLeak <- read_csv("./input/testingSet_Leak.csv")

trainLeak$leak <- gsub("\\[|\\'|\\]","", trainLeak$leak)
testLeak$leak <- gsub("\\[|\\'|\\]","", testLeak$leak)

unique(trainLeak$leak)
unique(testLeak$leak)

training <- left_join(training, trainLeak, by = c("display_id","ad_id"))
testing  <- left_join(testing, testLeak, by = c("display_id","ad_id"))

unique(training$leak)
unique(testing$leak)

#########################################################################################################################  

names(training)
names(testing)




i = 5

X_build <- subset(training, CVindices != i, select = -c( CVindices))
X_val   <- subset(training, CVindices == i, select = -c( CVindices)) 

names(X_build)

write.csv(X_build,  './input/trainingSet03_fold1to4.csv', row.names=FALSE, quote = FALSE)
write.csv(X_val, './input/trainingSet03_fold5.csv', row.names=FALSE, quote = FALSE)


training$CVindices <- NULL

gc()

names(training)

write.csv(training,  './input/trainingSet3_20161215.csv', row.names=FALSE, quote = FALSE)
write.csv(testing,  './input/testingSet3_20161215.csv', row.names=FALSE, quote = FALSE)


head(training)






trainingSet <- training
testingSet  <- testing

rm(training, testing); gc()


head(trainingSet)


trainingSet <- as.data.frame(trainingSet)
testingSet  <- as.data.frame(testingSet)
#names(testingSet)
feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c( "geo_location","timestamp", "publish_time" ,"CVindices","Date" ))])

testfeature.names <- names(testingSet[,-which(names(testingSet) %in% c("geo_location","timestamp", "publish_time" ,"CVindices","Date", "clicked" ))])

trainingSet$clicked1 <- trainingSet$clicked

trainingSet$clicked <- NULL

names(trainingSet)[names(trainingSet)=="clicked1"] <- "clicked"

names(trainingSet)

trainfile <- trainingSet[,feature.names]
testfile  <- testingSet[,testfeature.names]

names(trainfile)
names(testfile)

gc()

rm(trainingSet, testingSet); gc()

################################################################################################

# Sys.time()
# save.image(file = "Outbrain_Baseline01_20161216.RData" , safe = TRUE)
# Sys.time()

# Sys.time()
# load("Outbrain_Baseline01_20161216.RData"); gc()
# Sys.time()

################################################################################################
# leakfile  <- fread("./input/leak.csv") 
# names(leakfile)
# 
# names(leakfile)[2] <- "leakuuid"
# names(trainfile)
# names(testfile)
# 
# trainfile <- left_join(trainfile, leakfile, by ="document_id")
# testfile  <- left_join(testfile, leakfile, by ="document_id")
# 
# is.data.table(trainfile)
# is.data.frame(trainfile)
# 
# trainfile <- as.data.table(trainfile)
# testfile  <- as.data.table(testfile)
# 
# trainfileLeak    <-subset(trainfile, (!is.na(trainfile$leakuuid)) )
# trainfileNOnLeak <-subset(trainfile, ( is.na(trainfile$leakuuid)) )
# 
# testfileLeak    <-subset(testfile, (!is.na(testfile$leakuuid)) )
# testfileNOnLeak <-subset(testfile, ( is.na(testfile$leakuuid)) )
# 
# is.data.table(trainfileLeak)
# 
# head(trainfileLeak)
# 
# # 7406344
# Sys.time()
# trainfileLeak[, LeakuuidMatchCount := grepl(uuid, leakuuid), by = uuid]
# Sys.time()
# # 2536753
# testfileLeak[, LeakuuidMatchCount := grepl(uuid, leakuuid), by = uuid]
# Sys.time()
# 
# 
# summary(trainfile$leakuuid)
# 
# gc()
# 
# #t <- Sys.time()
# word_match <- function(words,title){
#   n_title <- 0
#  
#   #words <- unlist(strsplit(words," "))
#   nwords <- length(words)
#   #for(i in 1:length(words)){
#     #pattern <- paste("(^| )",words[i],"($| )",sep="")
#     pattern <- words #[i]
#     n_title <- n_title + grepl(pattern,title,fixed = FALSE,ignore.case=TRUE)
# 
#   #}
#   return(c(n_title,nwords))
# }
# 
# head(trainfileLeak)
# 
# cat("Get number of words and word matching title in train\n")
# train_words <- as.data.frame(t(mapply(word_match,trainfileLeak$uuid,trainfileLeak$leakuuid)))
# trainfileLeak$LeakuuidMatchCount <- train_words[,1]
# trainfileLeak$uuidCount          <- train_words[,2]
# 
# cat("Get number of words and word matching title in train\n")
# test_words <- as.data.frame(t(mapply(word_match,testfileLeak$uuid,testfileLeak$leakuuid)))
# testfileLeak$LeakuuidMatchCount <- test_words[,1]
# testfileLeak$uuidCount          <- test_words[,2]
# 
# # trainfile$LeakStatus <- mapply(grepl, pattern=trainfile$uuid, x=trainfile$leakuuid)
# # testfile$LeakStatus  <- mapply(grepl, pattern=testfile$uuid, x=testfile$leakuuid)
# 
# 
# head(trainfile)

#######################################################################################################################

write.csv(trainfile,  './input/trainingSet_20161215.csv', row.names=FALSE, quote = FALSE)
write.csv(testfile,   './input/testingSet_20161215.csv', row.names=FALSE, quote = FALSE)

write_csv(trainfile,  './input/trainingSet_20161218.csv')
write_csv(testfile,   './input/testingSet_20161218.csv')


########################################################################################################################

trainfile <- read_csv("./input/trainingSet_20161215.csv")

Prav_CVindices_5folds  <- read_csv("./CVSchema/Prav_CVindices_5folds.csv") 
names(Prav_CVindices_5folds)
Prav_CVindices_5folds$day <- NULL
# train -- 87,141,731
trainfile <- left_join(trainfile, Prav_CVindices_5folds, by = "display_id")

names(trainfile)
i = 5

X_build <- subset(trainfile, CVindices != i, select = -c( CVindices))
X_val   <- subset(trainfile, CVindices == i, select = -c( CVindices)) 

write.csv(X_build,  './input/trainingSet_20161215_train_fold1to4.csv', row.names=FALSE, quote = FALSE)
write.csv(X_val, './input/trainingSet_20161215_valid_fold5.csv', row.names=FALSE, quote = FALSE)


#########################################################################################################################
trainLeak <- read_csv("./input/trainingSet_Leak.csv")
testLeak <- read_csv("./input/testingSet_Leak.csv")

trainLeak$leak <- gsub("\\[|\\'|\\]","", trainLeak$leak)
testLeak$leak <- gsub("\\[|\\'|\\]","", testLeak$leak)
#########################################################################################################################   
trainfile <- read_csv("./input/trainingSet_20161215.csv")
trainLeak <- read_csv("./input/trainingSet_Leak.csv")
trainLeak$leak <- gsub("\\[|\\'|\\]","", trainLeak$leak)

trainfile <- left_join(trainfile, trainLeak , by = c("display_id","ad_id"))

Prav_CVindices_5folds  <- read_csv("./CVSchema/Prav_CVindices_5folds.csv") 
names(Prav_CVindices_5folds)
Prav_CVindices_5folds$day <- NULL
# train -- 87,141,731
trainfile <- left_join(trainfile, Prav_CVindices_5folds, by = "display_id")

names(trainfile)
i = 5

X_build <- subset(trainfile, CVindices != i, select = -c( CVindices))
X_val   <- subset(trainfile, CVindices == i, select = -c( CVindices)) 

write.csv(X_build,  './input/trainingSet_20161215_trainv2_fold1to4.csv', row.names=FALSE, quote = FALSE)
write.csv(X_val, './input/trainingSet_20161215_validv2_fold5.csv', row.names=FALSE, quote = FALSE)
















##########################################################################################################################
sapply(trainingSet, class)



cv          = 5
bags        = 1
nround.cv   = 10
printeveryn = 1
seed        = 2016

## for all remaining models, use same parameters 

param <- list(  "objective"        = 'reg:linear',
                #objective = logcoshobj, #fairobj #logcoshobj, #cauchyobj #,#'reg:linear',
                "booster"          = "gbtree",
                "eval_metric"      = "rmse",
                "nthread"          = 27,     
                "max_depth"        = 8,     
                "eta"              = 0.1, 
                "subsample"        = 0.7,  
                "colsample_bytree" = 0.7,
                #                 "alpha"            =  1,
                #                 "gamma"            =  1,
                "min_child_weight" = 3   
                
)


cat(cv, "-fold Cross Validation\n", sep = "")

set.seed(seed)
for (i in 5:cv)
  
{
  cat(i ,"fold Processing\n")
  
  cat("X_build fold Processing\n")
  X_build <- subset(trainingSet, CVindices != i, select = -c( CVindices))
  cat("X_val fold Processing\n")
  X_val   <- subset(trainingSet, CVindices == i, select = -c( CVindices)) 
  
  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=X_build$clicked)
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=X_val$clicked)
  watchlist <- list( val = dval,train = dtrain)
  
  
  pred_cv_bags   <- rep(0, nrow(X_val[, feature.names]))
  pred_test_bags <- rep(0, nrow(testingSet[,testfeature.names]))
  set.seed(seed)
  for (b in 1:bags) 
  {
    # seed = seed + b
    # cat(seed , " - Random Seed\n ")
    cat(b ," - bag Processing\n")
    cat("X_build training Processing\n")
    XGModel <- xgb.train(   params              = param,
                            #feval               = xgb.metric.log.mae, #xgb.metric.mae
                            data                = dtrain,
                            watchlist           = watchlist,
                            nrounds             = nround.cv ,
                            print.every.n       = printeveryn,
                            verbose             = TRUE, 
                            #maximize            = TRUE,
                            set.seed            = seed
    )
    cat("X_val prediction Processing\n")
    pred_cv    <- predict(XGModel, data.matrix(X_val[,feature.names]))
    cat("CV TestingSet prediction Processing\n")
    pred_test  <- predict(XGModel, data.matrix(testingSet[,testfeature.names]))
    
    pred_cv_bags   <- pred_cv_bags + pred_cv
    pred_test_bags <- pred_test_bags + pred_test
  }
  pred_cv_bags   <- pred_cv_bags / bags
  pred_test_bags <- pred_test_bags / bags
  
  cat("CV Fold-", i, " ", metric, ": ", score(X_val$clicked, pred_cv_bags, metric), "\n", sep = "")
  
  head(pred_cv_bags)
  head(X_val)
  
  val_predictions <- data.frame(display_id=X_val$display_id, add_id=X_val$add_id ,clicked = pred_cv_bags)
  test_predictions <- data.frame(display_id=testingSet$display_id, add_id=testingSet$add_id, clicked = pred_test_bags)
  
  # if(i == 1)
  # {
  #   write.csv(val_predictions,  'prav.xgb21.fold1.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.xgb21.fold1-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 2)
  # {
  #   write.csv(val_predictions,  'prav.xgb21.fold2.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.xgb21.fold2-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 3)
  # {
  #   write.csv(val_predictions,  'prav.xgb21.fold3.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.xgb21.fold3-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 4)
  # {
  #   write.csv(val_predictions,  'prav.xgb21.fold4.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.xgb21.fold4-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 5)
  # {
  #   write.csv(val_predictions,  'prav.xgb21.fold5.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, 'prav.xgb21.fold5-test.csv', row.names=FALSE, quote = FALSE)
  # }
  
}

# Full training

dtrain<-xgb.DMatrix(data=data.matrix(trainingSet[, feature.names]),label=trainingSet$clicked)
watchlist <- list( train = dtrain)

fulltrainnrounds = 1.2 * nround.cv

gc()
# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

fulltest_ensemble <- rep(0, nrow(testingSet[,testfeature.names]))

set.seed(seed)
for (b in 1:bags) {
  cat(b ," - bag Processing\n")
  # seed = seed + b
  # cat(seed , " - Random Seed\n ")
  cat("Bagging Full TrainingSet training\n")
  XGModelFulltrain <- xgb.train(    params              = param,
                                    feval               = xgb.metric.log.mae,
                                    data                = dtrain,
                                    watchlist           = watchlist,
                                    nrounds             = fulltrainnrounds,
                                    print.every.n       = printeveryn,
                                    verbose             = TRUE, 
                                    #maximize            = TRUE,
                                    set.seed            = seed
  )
  cat("Bagging Full Model prediction Processing\n")
  
  predfull_test         <- predict(XGModelFulltrain, data.matrix(testingSet[,testfeature.names]))
  
  fulltest_ensemble <- fulltest_ensemble + predfull_test
  
}

fulltest_ensemble <- fulltest_ensemble / bags
testfull_predictions  <- data.frame(id=testingSet$id, clicked = fulltest_ensemble)
write.csv(testfull_predictions, './submissions/prav.xgb21.full.csv', row.names=FALSE, quote = FALSE)




# #########################################################################################################
# Bagging - Full train
# #########################################################################################################



# head(testfull_predictions)

############################################################################################
model = xgb.dump(XGModel, with.stats=TRUE)

names = dimnames(trainingSet[,feature.names])[[2]]
importance_matrix = xgb.importance( names , model = XGModel)
#xgb.plot.importance(importance_matrix)
impMatrix <- as.data.frame(importance_matrix)
#write.csv(impMatrix, './Models/prav.xgb21.ImpMatrix.csv', row.names=FALSE, quote = FALSE)
# impMatrix[1:12,1]
# impMatrix[,1]
ImpFeature <- impMatrix[,1]


# #########################################################################################################



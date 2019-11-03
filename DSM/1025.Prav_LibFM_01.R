require(libFMexe)

################################################################################################

# 20160806 - Baseline02 -- 64 features -- R Image

# Sys.time()
# load("RedHat_Baseline02_20160806.RData")
# Sys.time()
################################################################################################

# cv_02 for local cv
CVindices5folds      <- read_csv("./CVSchema/Prav_CVindices_5folds_02.csv")


trainingSet <- left_join(training, CVindices5folds, by = "people_id" , all.x = TRUE)
testingSet  <- testing


feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("people_id","activity_id","date","p_date", "CVindices"
                                                                         
))])
testfeature.names <- names(testingSet[,-which(names(testingSet) %in% c("people_id","activity_id","date","p_date"))])


char.cols <- setdiff(feature.names, "outcome")

for (f in char.cols) {
  #if (class(D[[f]])=="character") {
  levels <- unique(c(trainingSet[[f]]))
  trainingSet[[f]] <- as.numeric(factor(trainingSet[[f]], levels=levels))
  #}
}


cv       = 5
seed     = 2016

cat("X_build fold Processing\n")
X_build <- trainingSet[trainingSet$CVindices != cv, colnames(trainingSet)]
cat("X_val fold Processing\n")
X_val   <- trainingSet[trainingSet$CVindices == cv, colnames(trainingSet)]




# X_build[,char.cols] <- apply(X_build[,char.cols], 2, function(y) as.character( y))
# X_val[,char.cols]   <- apply(X_val[,char.cols], 2, function(y) as.character( y))



#setwd("C:/Users/padepu/AppData/Local/Temp")
pred_cv = libFM(X_build[, feature.names],
                X_val[, feature.names], 
                outcome ~ 
                  char_10        +   activity_category + char_1           + char_2           + char_3        +  
                  char_4         +   char_5            + char_6           + char_7           + char_8        +  
                  char_9         +   char10peopleCount + p_group_1        + p_char_1         +          
                  p_char_2       +   p_char_3          + p_char_4         + p_char_5         + p_char_6      +  
                  p_char_7       +   p_char_8          + p_char_9         + p_char_10        + p_char_11     +  
                  p_char_12      +   p_char_13         + p_char_14        + p_char_15        + p_char_16     +  
                  p_char_17      +   p_char_18         + p_char_19        + p_char_20        + p_char_21     +  
                  p_char_22      +   p_char_23         + p_char_24        + p_char_25        + p_char_26     +  
                  p_char_27      +   p_char_28         + p_char_29        + p_char_30        + p_char_31     +  
                  p_char_32      +   p_char_33         + p_char_34        + p_char_35        + p_char_36     +  
                  p_char_37      +   p_char_38          ,
               task = "r", dim = 10, iter = 1000
               , exe_loc = "C:\\Users\\padepu\\Documents\\R\\11RedHat\\LibFM")

cat("CV Fold-", cv, " ", metric, ": ", score(X_val$outcome, pred_cv, metric), "\n", sep = "")

head(pred_cv)
head(X_val$outcome)



cat("X_build training Processing\n")
rf <- h2o.randomForest(         ##
  training_frame   = X_build ,  ##
  validation_frame = X_val,     ##
  x=feature.names,              ##
  y="outcome",                  ##
  mtries = mtry,
  ntrees = ntrees,              ##
  max_depth = maxdepth,         ## Increase depth, from 20
  seed=seed) 

pred_cv                  <- predict(rf, X_val[,testfeature.names])
cv_predictions           <- h2o.cbind(X_val$outcome,pred_cv)
colnames(cv_predictions) <-c("outcome","pred_outcome")
cv_predictions01         <- as.data.frame(cv_predictions)
cat("CV Fold-", cv, " ", metric, ": ", score(cv_predictions01$outcome, cv_predictions01$pred_outcome, metric), "\n", sep = "")

h2o.scoreHistory(rf)
# target should be factor to get classification and auc CV results
# baseline 01 CV 
# h2o.auc(rf, train = TRUE)  #0.999639 
# h2o.auc(rf, valid = TRUE)  #0.9634633
# 
# # baseline 02 CV 
# h2o.auc(rf, train = TRUE)  #0.9997572 
# h2o.auc(rf, valid = TRUE)  #0.9727386 # CV Fold-5 auc: 0.9796223



# Full training

rfFulltrain <- h2o.randomForest(       
  training_frame   = trainingSet ,  
  validation_frame = trainingSet,     
  x                = feature.names,              
  y                = "outcome",              
  model_id         = "prav.rfmodel",
  mtries           = mtry,
  ntrees           = ntrees,                   
  max_depth        = maxdepth,               
  seed=seed)  

cat("CV TestingSet prediction Processing\n")
predfull_test                  <- predict(rfFulltrain, testingSet[,testfeature.names])
testfull_predictions           <- h2o.cbind(testingSet$activity_id,predfull_test)
colnames(testfull_predictions) <-c("activity_id","outcome")
h2o.exportFile(testfull_predictions,path ="./submissions/Prav_h2o_rf02_02.csv")


h2o.scoreHistory(rfFulltrain)

impMatrix <- as.data.frame(h2o.varimp(rfFulltrain))

impMatrix
# # list of features for training
# feature.names <- names(train.hex)
# feature.names <- feature.names[! feature.names %in% c('people_id','outcome','train','activity_id')]
# 
# # train random forest model, use ntrees = 100 to get LB score ~0.96 (0.96004 in my case)
# drf <- h2o.randomForest(x=feature.names, y='outcome', training_frame = train.hex, ntrees = 2)
# 
# # create output for making submission
# sub <- data.frame(activity_id = as.vector(test.hex$activity_id), outcome = as.vector(predict(drf,test.hex)))
# write.table(sub, './sub_h2o_drf.csv',quote=F,sep=',',row.names=F)
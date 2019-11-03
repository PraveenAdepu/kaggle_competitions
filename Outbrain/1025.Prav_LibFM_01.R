require(libFMexe)
# install.package("devtools")
# devtools::install_github("andland/libFMexe")
################################################################################################

# Sys.time()
# load("Outbrain_Baseline01_20161216.RData"); gc()
# Sys.time()
################################################################################################
rm(train_test, trainLeak, testLeak, d.lub, h.lub, m.lub, t.lub); gc()

i = 5

X_build <- subset(training, CVindices != i, select = -c( CVindices))
X_val   <- subset(training, CVindices == i, select = -c( CVindices))



Xbuild_libFM = model_frame_libFM(clicked ~ ad_id + uuid + event_document_id+ platform+event_source_id+event_publisher_id+event_category_id+
                                   event_entity_id + event_topic_id + document_id + campaign_id + advertiser_id + source_id + 
                                   publisher_id + category_id + entity_id + topic_id + location1 + location2 + location3 + 
                                   day + hour + minutes + event_publish_dateToDate + publish_dateToDate + event_publish_dateTopublishdate + leak
                                 , X_build)


Xval_libFM = model_frame_libFM(clicked ~ ad_id + uuid + event_document_id+ platform+event_source_id+event_publisher_id+event_category_id+
                                   event_entity_id + event_topic_id + document_id + campaign_id + advertiser_id + source_id + 
                                   publisher_id + category_id + entity_id + topic_id + location1 + location2 + location3 + 
                                   day + hour + minutes + event_publish_dateToDate + publish_dateToDate + event_publish_dateTopublishdate + leak
                                 , X_val)

names(training)

#setwd("C:/Users/padepu/AppData/Local/Temp")
pred_cv = libFM(Xbuild_libFM,
                Xval_libFM, 
                task = "r", dim = 10, iter = 10
               , exe_loc = "C:\\Users\\SriPrav\\Documents\\R\\13Outbrain\\input\\LibFM")

cat("CV Fold-", i, " ", metric, ": ", score(X_val$clicked, pred_cv, metric), "\n", sep = "")

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

##########################################################################################################################
# Metric function forecast validation ####################################################################################
##########################################################################################################################
score <- function(a,b,metric)
  
{    
  switch(metric,           
         accuracy = sum(abs(a-b)<=0.5)/length(a),           
         auc = auc(a,b),           
         logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),           
         mae = sum(abs(a-b))/length(a),           
         precision = length(a[a==b])/length(a),           
         rmse = sqrt(sum((a-b)^2)/length(a)),           
         rmspe = sqrt(sum(((a-b)/a)^2)/length(a)))    
}

metric = "auc"
##########################################################################################################################
# Metric function forecast validation ####################################################################################
##########################################################################################################################


###########################################################################################################
# CV folds creation #######################################################################################
###########################################################################################################


#Input to function


Create5Folds <- function(train, CVSourceColumn, RandomSample, RandomSeed)
{
  set.seed(RandomSeed)
  if(RandomSample)
  {
    train <- as.data.frame(train[sample(1:nrow(train)), ])
    names(train)[1] <- CVSourceColumn
  }
  names(train)[1] <- CVSourceColumn
  
  folds <- createFolds(train[[CVSourceColumn]], k = 5)
  
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
  
  names(trainingFold01)[1] <- CVSourceColumn
  names(trainingFold02)[1] <- CVSourceColumn
  names(trainingFold03)[1] <- CVSourceColumn
  names(trainingFold04)[1] <- CVSourceColumn
  names(trainingFold05)[1] <- CVSourceColumn
  
  trainingFolds <- rbind(trainingFold01, trainingFold02 , trainingFold03, trainingFold04, trainingFold05 )
  rm(trainingFold01,trainingFold02,trainingFold03,trainingFold04,trainingFold05); gc()
  
  return(trainingFolds)
}


###########################################################################################################
# CV folds creation #######################################################################################
###########################################################################################################

normalit<-function(m){
  (m - min(m))/(max(m)-min(m))
}


##########################################################################################################################
# Unit testing of CV folds creation function
##########################################################################################################################
# Start - Data pipeline preparation
# train <- read.csv("./source/source_data.csv", quote="")
# names(train)
# CVColumns <- c("Service_Number")
# train <- as.data.frame(train$Service_Number)
# names(train)[1] <- "Service_Number"
# head(train,2)
# trainingFolds <- Create5Folds(train, "Service_Number", RandomSample=TRUE , 2017)
# head(trainingFolds,2)
##########################################################################################################################



###########################################################################################################
# Data Visualisations Functions 
###########################################################################################################

plot_coeffs <- function(mlr_model) {
  coeffs <- coefficients(mlr_model)
  mp <- barplot(coeffs, col="#3F97D0", xaxt='n', main="Regression Coefficients")
  lablist <- names(coeffs)
  text(mp, par("usr")[3], labels = lablist, srt = 45, adj = c(1.1,1.1), xpd = TRUE, cex=0.6)
}

rank_comparison_auc <- function(labels, scores, plot_image=TRUE, ...){
  score_order <- order(scores, decreasing=TRUE)
  labels <- as.logical(labels[score_order])
  scores <- scores[score_order]
  pos_scores <- scores[labels]
  neg_scores <- scores[!labels]
  n_pos <- sum(labels)
  n_neg <- sum(!labels)
  M <- outer(sum(labels):1, 1:sum(!labels), 
             function(i, j) (1 + sign(pos_scores[i] - neg_scores[j]))/2)
  
  AUC <- mean (M)
  if (plot_image){
    image(t(M[nrow(M):1,]), ...)
    library(pROC)
    with( roc(labels, scores),
          lines((1 + 1/n_neg)*((1 - specificities) - 0.5/n_neg), 
                (1 + 1/n_pos)*sensitivities - 0.5/n_pos, 
                col="blue", lwd=2, type='b'))
    text(0.5, 0.5, sprintf("AUC = %0.4f", AUC))
  }
  
  #return(AUC)
}

###########################################################################################################
# rank_comparison_auc usage
###########################################################################################################
# rank_comparison_auc(labels=as.logical(testingSet$LostStolen), scores=testingSet$prediction)


# roc_full_resolution <- roc(testingSet$LostStolen, testingSet$prediction)
# rounded_scores <- round(testingSet$prediction, digits=1)
# roc_rounded <- roc(testingSet$LostStolen, rounded_scores)
# plot(roc_full_resolution, print.auc=TRUE)
# ## 
# ## Call:
# ## roc.default(response = test_set$bad_widget, predictor = glm_response_scores)
# ## 
# ## Data: glm_response_scores in 59 controls (test_set$bad_widget FALSE) < 66 cases (test_set$bad_widget TRUE).
# ## Area under the curve: 0.9037
# lines(roc_rounded, col="red", type='b')
# text(0.4, 0.43, labels=sprintf("AUC: %0.3f", auc(roc_rounded)), col="red")


simple_auc <- function(TPR, FPR){
  # inputs already sorted, best scores first 
  dFPR <- c(diff(FPR), 0)
  dTPR <- c(diff(TPR), 0)
  sum(TPR * dFPR) + sum(dTPR * dFPR)/2
}

auc_probability <- function(labels, scores, N=1e7){
  pos <- sample(scores[labels], N, replace=TRUE)
  neg <- sample(scores[!labels], N, replace=TRUE)
  # sum( (1 + sign(pos - neg))/2)/N # does the same thing
  (sum(pos > neg) + sum(pos == neg)/2) / N # give partial credit for ties
}

############################################################################################################


Create5Folds_Classification <- function(train, CVSourceColumn, RandomSample, RandomSeed)
{
  set.seed(RandomSeed)
  if(RandomSample)
  {
    train <- as.data.frame(train[sample(1:nrow(train)), ])
    
  }
  for(i in 1:length(CVSourceColumn))
  {
    names(train)[i] <- CVSourceColumn[i]
  }
  
  folds <- createFolds(train[[CVSourceColumn[2]]], k = 5) # Assuming Classification flag is in 2 position of columns list
  
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
  
  for(i in 1:length(CVSourceColumn))
  {
    names(trainingFold01)[i] <- CVSourceColumn[i]
    names(trainingFold02)[i] <- CVSourceColumn[i]
    names(trainingFold03)[i] <- CVSourceColumn[i]
    names(trainingFold04)[i] <- CVSourceColumn[i]
    names(trainingFold05)[i] <- CVSourceColumn[i]
  }
  
  
  
  trainingFolds <- rbind(trainingFold01, trainingFold02 , trainingFold03, trainingFold04, trainingFold05 )
  rm(trainingFold01,trainingFold02,trainingFold03,trainingFold04,trainingFold05); gc()
  
  
  return(trainingFolds)
}




##########################################################################################################
# Testing
# trainingFolds %>%
#     group_by(CVindices,Claim) %>%
#     summarise(Count = n())
##########################################################################################################

##########################################################################################################
## Get highly correlated features list from dataset with cutoff 

GethighlyCorrelatedFeatures <- function(train, cutoff_threshold)
                              {
                                train.Corr                  <- cor(train)
                                hc                          <- findCorrelation(train.Corr, cutoff=cutoff_threshold) # putt any value as a "cutoff" 
                                hc                          <- sort(hc)
                                highlyCorrelated.features   <- names(train[,c(hc)])
                                return(highlyCorrelated.features)
                              }


# highlyCorrelatedFeaturesList <- GethighlyCorrelatedFeatures(trainingSet[,all.cols], 0.90)
# cor(trainingSet[,surv.var.names])


##########################################################################################################

trainingSet.MultiClass.cv.pipeline <- function(trainingSet.source, source.target, source.target_label)
{
  # filter rows with source.target NA columns
  trainingSet    <- trainingSet.source[!is.na(trainingSet.source[,source.target]),]
  
  # From modelfile source, we got all columns as factors so no need to convert into factors 
  # XGB MultiClass Labels starts from 0 
  trainingSet[[source.target_label]] <- as.numeric(trainingSet[[source.target]])-1
  
  trainingSet$rowID  <- seq.int(nrow(trainingSet))
  
  
  CVSourceColumn     <- c("rowID",source.target_label)
  train              <- trainingSet[, CVSourceColumn]
  trainingFolds      <- Create5Folds_Classification(train, CVSourceColumn,RandomSample=TRUE, RandomSeed=2017)
  head(trainingFolds)
  
  trainingFolds[[source.target_label]] <- NULL
  
  trainingSet       <- left_join(trainingSet, trainingFolds, by="rowID")
  
  return(trainingSet)
}


XGB_MultiClassification_train.cv <- function(trainingSet, feature.names,target,num.class,num.class.names,cv, bags,param,nround.cv,printeveryn,verbose,maximize,seed,validation.pred.columns, VarImportanceFile, train.cv.singlefold.test)
{
  if(train.cv.singlefold.test )
  {
    cv.allfolds = 1
    cat("train.cv.singlefold.test : ", train.cv.singlefold.test , "\n")
    cat("train.cv train fold : ", cv.allfolds , "\n")
    
  }
  else
  {
    cv.allfolds = cv
  }
  cat(cv.allfolds, "-fold Cross Validation\n", sep = "")
  set.seed(seed)
  
   for (i in 1:cv.allfolds)
  
     { 
  
  cat(i ,"fold Processing\n")
  
  cat("X_build fold Processing\n")
  X_build <- subset(trainingSet, CVindices != i)
  cat("X_val fold Processing\n")
  X_val   <- subset(trainingSet, CVindices == i) 
  
  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=X_build[[target]])
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=X_val[[target]])
  watchlist <- list( val = dval,train = dtrain)
  
  
  pred_cv_bags   <- rep(0, nrow(X_val[, feature.names])     * num.class )
  # pred_test_bags <- rep(0, nrow(testingSet[,feature.names]) * num.class )
  
  for (b in 1:bags) 
  {
    cat(b ," - bag Processing\n")
    seed = b + seed
    set.seed(seed)
    cat("X_build training Processing\n")
    XGModel <- xgb.train(   params              = param,
                            data                = dtrain,
                            watchlist           = watchlist,
                            nrounds             = nround.cv ,
                            print_every_n       = printeveryn,
                            verbose             = verbose, 
                            maximize            = maximize,
                            set.seed            = seed
    )
    # cat("X_val prediction Processing\n")
    pred_cv    <- predict(XGModel, data.matrix(X_val[,feature.names]))
    # cat("CV TestingSet prediction Processing\n")
    # pred_test  <- predict(XGModel, data.matrix(testingSet[,feature.names]))
    # 
    pred_cv_bags    <- pred_cv_bags + pred_cv
    # pred_test_bags <- pred_test_bags + pred_test
  }
  pred_cv_bags      <- pred_cv_bags / bags
  # pred_test_bags <- pred_test_bags / bags
  X_val_pred        <-  t(matrix(pred_cv_bags, nrow=num.class, ncol=length(pred_cv_bags)/num.class))
  X_val_pred        <- data.frame(X_val_pred)
  names(X_val_pred) <- num.class.names
  
  
  if(i == 1)
    
  {
    OOF_preds   <- cbind(X_val[, validation.pred.columns],X_val_pred)
  }
  else{
    
       cv_preds    <- cbind(X_val[, validation.pred.columns],X_val_pred)
       OOF_preds   <- rbind(OOF_preds,cv_preds)
      }
  
  
   }

  ############################################################################################################
  
  model = xgb.dump(XGModel, with_stats=TRUE)
  
  names = dimnames(trainingSet[,feature.names])[[2]]
  importance_matrix = xgb.importance( names , model = XGModel)
  xgb.plot.importance(importance_matrix)
  impMatrix <- as.data.frame(importance_matrix)
  write.csv(impMatrix, VarImportanceFile, row.names = FALSE)
  
  ############################################################################################################
  OOF_prediction <- OOF_preds %>%
  mutate(pred_class =colnames(OOF_preds[,num.class.names])[max.col(OOF_preds[,num.class.names],ties.method="last")])
   
  
  return(OOF_prediction)
  
 
}


OOF_MultiClass_predictions_validation <- function(OOF_prediction_dataset,num.class.names, source_target)
{
  OOF_prediction <- OOF_prediction_dataset %>%
    mutate(pred_class =colnames(OOF_prediction_dataset[,num.class.names])[max.col(OOF_prediction_dataset[,num.class.names],ties.method="last")])
  
  confusion <- confusionMatrix(OOF_prediction[,c(source_target)],OOF_prediction[,c("pred_class")])
  
  accuracy <- sum(diag(confusion$table)) / sum(confusion$table)
  precision <- diag(confusion$table) / rowSums(confusion$table)
  recall <- (diag(confusion$table) / colSums(confusion$table))
  
  print(confusion)
  cat('accuracy : ' ,accuracy," \n")
  cat('precision : ',precision," \n")
  cat('recall : '   ,recall,"\n")
  
  

  for(class.name in num.class.names)
    {
    
    # cat("Confusion matrix for positive : ", class.name , "\n")
    # confusion <- confusionMatrix(OOF_prediction[,c(source_target)],OOF_prediction[,c("pred_class")], positive = class.name)
    # 
    # accuracy <- sum(diag(confusion$table)) / sum(confusion$table)
    # precision <- diag(confusion$table) / rowSums(confusion$table)
    # recall <- (diag(confusion$table) / colSums(confusion$table))
    # 
    # print(confusion)
    # cat('accuracy : ' ,accuracy," \n")
    # cat('precision : ',precision," \n")
    # cat('recall : '   ,recall,"\n")

    cat("printing roc curve for class : ", class.name, "\n")
    
    roc.plot <- plot(roc(ifelse(OOF_prediction_dataset[[source_target]] == class.name , 1 , 0),OOF_prediction_dataset[[class.name]]),xlab="False Positive Rate", ylab="True Positive Rate",print.auc=TRUE,legacy.axes=TRUE, title = class.name)
    print(roc.plot)
    
    
    }
}




XGB_MultiClassification_train <- function(trainingSet, feature.names,target,num.class,num.class.names,cv, bags,param,nround.cv,printeveryn,verbose,maximize,seed,validation.pred.columns, save.train.model.name,train_if_not_exists)
{
  if(train_if_not_exists & file.exists(save.train.model.name))
    {
    cat("train_if_not_exists : ", train_if_not_exists , "\n")
    cat("Model already existed at : ", save.train.model.name, "\n")
    cat("training at this run : ",  "No\n")
    
    }
  else
    {
    
      dtrain<-xgb.DMatrix(data=data.matrix(trainingSet[, feature.names]),label=trainingSet[[target]])
      watchlist <- list( train = dtrain)
      
      fulltrainnrounds = (1+1/cv) * nround.cv
      # 
      # for (b in 1:bags) {
      #   cat(b ," - bag Processing\n")
      #   seed = b + seed
      set.seed(seed)
      cat("Bagging Full TrainingSet training\n")
      XGModelFulltrain <- xgb.train(    params              = param,
                                        data                = dtrain,
                                        watchlist           = watchlist,
                                        nrounds             = fulltrainnrounds,
                                        print_every_n       = printeveryn,
                                        verbose             = verbose,
                                        maximize            = maximize,
                                        set.seed            = seed
      )
      
      
      # }
      
      cat("saving trained model : " ,save.train.model.name,"\n")
      xgb.save(XGModelFulltrain,save.train.model.name)
      
    }
}

XGB_MultiClassification_prediction <- function(testingSet
                                               , feature.names
                                               , target = source.target_label
                                               , num.class
                                               , num.class.names
                                               , cv
                                               , bags
                                               , param
                                               , nround.cv
                                               , printeveryn
                                               , verbose
                                               , maximize
                                               , seed
                                               , validation.pred.columns
                                               , save.train.model.name
                                               , prediction_report_features
                                               , testing_prediction_file)
{
  
  
  load.model.name         <- xgb.load(save.train.model.name)
  
  fulltest_ensemble       <- rep(0, nrow(testingSet[,feature.names]) * num.class )
  fulltest_ensemble       <- predict(load.model.name, data.matrix(testingSet[,feature.names]))
  fulltest_pred           <- t(matrix(fulltest_ensemble, nrow=num.class, ncol=length(fulltest_ensemble)/num.class))
  fulltest_pred           <- data.frame(fulltest_pred)
  names(fulltest_pred)    <- num.class.names
  testingSet              <- cbind(testingSet, fulltest_pred)
  report_features         <- union(prediction_report_features,num.class.names)
  testingSet_prediction   <- testingSet[,report_features]
  write.csv(testingSet_prediction, testing_prediction_file, row.names = FALSE)
  
}

XGB_MultiClassification_CV.Validation.Train.Prediction.pipeline <- function(trainingSet.source
                                                                            , source.target
                                                                            , source.target_label
                                                                            , cv
                                                                            , bags
                                                                            , param
                                                                            , nround.cv
                                                                            , printeveryn
                                                                            , verbose
                                                                            , maximize
                                                                            , seed
                                                                            , train.cv.singlefold.test
                                                                            , train_if_not_exists
                                                                            , train.cv
                                                                            , validate
                                                                            , train
                                                                            , prediction
                                                                            , prediction_report_features
                                                                            , model_name)
{
  
  trainingSet                   <- trainingSet.MultiClass.cv.pipeline(trainingSet.source, source.target, source.target_label) 
  num.class.names               <- levels(trainingSet[[source.target]])
  num.class                     <- length(num.class.names) 
  param["num_class"]            <- num.class
  
  
  
  validation.pred.columns       <- c("rowID","CVindices",source.target,source.target_label)
  model.importance.matrix.file  <- paste('./model_outputs/',model_name,'_',source.target,'_model.csv',sep='')
  save.train.model.name         <- paste('model_outputs/',model_name,'_',source.target,'_model.model',sep='')
  testing_prediction_file       <- paste('./model_outputs/',model_name,'_',source.target,'_testingSet_predictions.csv',sep='')
  if(train.cv)
  {
    start_time <- Sys.time()
    cat("progress : ", "train cross validation " ,"\n")
    OOF_MultiClass_prediction     <- XGB_MultiClassification_train.cv(   trainingSet
                                                                         , feature.names
                                                                         , target=source.target_label
                                                                         , num.class
                                                                         , num.class.names
                                                                         , cv
                                                                         , bags
                                                                         , param 
                                                                         , nround.cv
                                                                         , printeveryn
                                                                         , verbose
                                                                         , maximize
                                                                         , seed
                                                                         , validation.pred.columns
                                                                         , model.importance.matrix.file
                                                                         , train.cv.singlefold.test
    )
    end_time <- Sys.time()
    cat("XGB_MultiClassification_train.cv time : ",round(as.numeric(end_time-start_time, units = "mins"),1), " minutes \n")
    
  }
  ####################################################################################################################
  # End train.cv Cross Validation for tuning       ###################################################################
  ####################################################################################################################
 
  
  ####################################################################################################################
  # Start - Validation and Confusion Matrix results       ############################################################
  ####################################################################################################################
  if(validate)
  {
    start_time <- Sys.time()
    cat("progress : ", "validation " ,"\n")
    OOF_MultiClass_predictions_validation(  OOF_MultiClass_prediction
                                           , num.class.names
                                           , source_target=source.target
    )
    end_time <- Sys.time()
    cat("OOF_MultiClass_predictions_validation time : ",round(as.numeric(end_time-start_time, units = "mins"),1), " minutes \n")
  }
  ####################################################################################################################
  # End - Validation and Confusion Matrix results       ##############################################################
  ####################################################################################################################
  
  
  ####################################################################################################################
  # Start - Full Model training and saving model          ############################################################
  ####################################################################################################################
  
  if(train){
    start_time <- Sys.time()
    cat("progress : ", "train " ,"\n")
    XGB_MultiClassification_train(  trainingSet
                                    , feature.names
                                    , target=source.target_label
                                    , num.class 
                                    , num.class.names 
                                    , cv 
                                    , bags 
                                    , param 
                                    , nround.cv 
                                    , printeveryn 
                                    , verbose 
                                    , maximize 
                                    , seed 
                                    , validation.pred.columns 
                                    , save.train.model.name 
                                    , train_if_not_exists
    )
    end_time <- Sys.time()
    cat("XGB_MultiClassification_train time : ",round(as.numeric(end_time-start_time, units = "mins"),1), " minutes \n")
  }
  
  ####################################################################################################################
  # End   - Full Model training and saving model          ############################################################
  ####################################################################################################################
  if(prediction)
  {
    start_time <- Sys.time()
    cat("progress : ", "prediction " ,"\n")
    XGB_MultiClassification_prediction(testingSet
                                                 , feature.names
                                                 , target = source.target_label
                                                 , num.class
                                                 , num.class.names
                                                 , cv
                                                 , bags
                                                 , param
                                                 , nround.cv
                                                 , printeveryn
                                                 , verbose
                                                 , maximize
                                                 , seed
                                                 , validation.pred.columns
                                                 , save.train.model.name
                                                 , prediction_report_features
                                                 , testing_prediction_file)
  
    cat("testingSet predictions results wriiten to file : ", testing_prediction_file ,"\n")
    end_time <- Sys.time()
    cat("XGB_MultiClassification_prediction time : ",round(as.numeric(end_time-start_time, units = "mins"),1), " minutes \n")
  }
}


# ####################################################################################################################
# # Features to target and target level AUC calculation - work in progress          ##################################
# ####################################################################################################################
# 
# feature       = "MOU_ANNUAL_MEAN_CAP"
# target        = "yvar_IsRecontract"
# target.levels = unique(trainingSet.source$yvar_IsRecontract)
# 
# feature.AUC.scores<- function(trainingSet.source,feature.names,target, target.has.levels)
# {
# feature.AUC.scores <- data.frame(feature=character(),
#                                  target=character(), 
#                                  level=character(),
#                                  AUC=double,
#                                  stringsAsFactors=FALSE) 
# if(target.has.levels)
# {
#   for(target.level in unique(trainingSet.source[[target]]))
#   {
#     cat("AUC metric feature : ", feature , "\n")
#     cat("AUC metric target level : ", target.level , "\n")
#     auc.score = score(trainingSet.source[[feature]], ifelse(trainingSet.source[[target]] == target.level, 1, 0), metric)
#     
#     feature.AUC.scores<- rbind(feature.AUC.scores,data.frame(feature = feature, target = target, level = target.level, AUC = auc.score))
#   }
# }
# else
# {
#   cat("AUC metric feature : ", feature , "\n")
#   cat("AUC metric target level : ", target.level , "\n")
#   auc.score = score(trainingSet.source[[feature]], trainingSet.source[[target]] , metric)
#   
#   feature.AUC.scores<- rbind(feature.AUC.scores,data.frame(feature = feature, target = target, level = target.level, AUC = auc.score))
# }
# return(feature.AUC.scores)
# }
# 
# feature.AUC.scores <- feature.AUC.scores(trainingSet.source,feature.names = "MOU_ANNUAL_MEAN_CAP",target = "yvar_IsRepayment", target.has.levels = TRUE)
# 
# ####################################################################################################################
# # Features to target and target level AUC calculation - work in progress          ##################################
# ####################################################################################################################




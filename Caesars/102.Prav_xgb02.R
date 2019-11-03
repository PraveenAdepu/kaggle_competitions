
# library(devtools)
# install_version("xgboost", "0.6-4")
library(xgboost)
require(data.table)

gc()
#names(testingSet)
# feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("customer_id","market", "date" ,"f_16","f_17","f_19","f_21","f_22"
#                                                                          ,"f_23","f_24","f_25","f_26","f_40","f_41","target"))])

# X_build <- read_csv("./input/X_build.csv")
# X_val   <- read_csv("./input/X_val.csv")

X_build <- as.data.frame(X_build)
X_val  <- as.data.frame(X_val)

cust.test <- X_build %>% filter(customer_id == "144112433441")

X_build$f_13_f15_div <- X_build$f_13 / X_build$f_15
X_val$f_13_f15_div   <- X_val$f_13 / X_val$f_15
# 
# X_build$customer_f3_lag1_diffdiv  <-ifelse(is.na(X_build$customer_f3_lag1) , NA, X_build$customer_f3_lag1_diff  / X_build$customer_f3_lag1   )
# X_build$customer_f4_lag1_diffdiv  <-ifelse(is.na(X_build$customer_f4_lag1) , NA, X_build$customer_f4_lag1_diff  / X_build$customer_f4_lag1)
# X_build$customer_f6_lag1_diffdiv  <-ifelse(is.na(X_build$customer_f6_lag1) , NA, X_build$customer_f6_lag1_diff  / X_build$customer_f6_lag1)
# X_build$customer_f8_lag1_diffdiv  <-ifelse(is.na(X_build$customer_f8_lag1) , NA, X_build$customer_f8_lag1_diff  / X_build$customer_f8_lag1)
# X_build$customer_f10_lag1_diffdiv <-ifelse(is.na(X_build$customer_f10_lag1), NA, X_build$customer_f10_lag1_diff / X_build$customer_f10_lag1)
# X_build$customer_f11_lag1_diffdiv <-ifelse(is.na(X_build$customer_f11_lag1), NA, X_build$customer_f11_lag1_diff / X_build$customer_f11_lag1)
# X_build$customer_f12_lag1_diffdiv <-ifelse(is.na(X_build$customer_f12_lag1), NA, X_build$customer_f12_lag1_diff / X_build$customer_f12_lag1)
# X_build$customer_f13_lag1_diffdiv <-ifelse(is.na(X_build$customer_f13_lag1), NA, X_build$customer_f13_lag1_diff / X_build$customer_f13_lag1)
# X_build$customer_f14_lag1_diffdiv <-ifelse(is.na(X_build$customer_f14_lag1), NA, X_build$customer_f14_lag1_diff / X_build$customer_f14_lag1)
# X_build$customer_f15_lag1_diffdiv <-ifelse(is.na(X_build$customer_f15_lag1), NA, X_build$customer_f15_lag1_diff / X_build$customer_f15_lag1)
# X_build$customer_f17_lag1_diffdiv <-ifelse(is.na(X_build$customer_f17_lag1), NA, X_build$customer_f17_lag1_diff / X_build$customer_f17_lag1)
# X_build$customer_f18_lag1_diffdiv <-ifelse(is.na(X_build$customer_f18_lag1), NA, X_build$customer_f18_lag1_diff / X_build$customer_f18_lag1)
# X_build$customer_f21_lag1_diffdiv <-ifelse(is.na(X_build$customer_f21_lag1), NA, X_build$customer_f21_lag1_diff / X_build$customer_f21_lag1)
# X_build$customer_f22_lag1_diffdiv <-ifelse(is.na(X_build$customer_f22_lag1), NA, X_build$customer_f22_lag1_diff / X_build$customer_f22_lag1)
# X_build$customer_f25_lag1_diffdiv <-ifelse(is.na(X_build$customer_f25_lag1), NA, X_build$customer_f25_lag1_diff / X_build$customer_f25_lag1)
# X_build$customer_f26_lag1_diffdiv <-ifelse(is.na(X_build$customer_f26_lag1), NA, X_build$customer_f26_lag1_diff / X_build$customer_f26_lag1)
# X_build$customer_f27_lag1_diffdiv <-ifelse(is.na(X_build$customer_f27_lag1), NA, X_build$customer_f27_lag1_diff / X_build$customer_f27_lag1)
# X_build$customer_f28_lag1_diffdiv <-ifelse(is.na(X_build$customer_f28_lag1), NA, X_build$customer_f28_lag1_diff / X_build$customer_f28_lag1)
# X_build$customer_f30_lag1_diffdiv <-ifelse(is.na(X_build$customer_f30_lag1), NA, X_build$customer_f30_lag1_diff / X_build$customer_f30_lag1)
# X_build$customer_f32_lag1_diffdiv <-ifelse(is.na(X_build$customer_f32_lag1), NA, X_build$customer_f32_lag1_diff / X_build$customer_f32_lag1)
# X_build$customer_f34_lag1_diffdiv <-ifelse(is.na(X_build$customer_f34_lag1), NA, X_build$customer_f34_lag1_diff / X_build$customer_f34_lag1)
# X_build$customer_f35_lag1_diffdiv <-ifelse(is.na(X_build$customer_f35_lag1), NA, X_build$customer_f35_lag1_diff / X_build$customer_f35_lag1)
# X_build$customer_f36_lag1_diffdiv <-ifelse(is.na(X_build$customer_f36_lag1), NA, X_build$customer_f36_lag1_diff / X_build$customer_f36_lag1)
# X_build$customer_f37_lag1_diffdiv <-ifelse(is.na(X_build$customer_f37_lag1), NA, X_build$customer_f37_lag1_diff / X_build$customer_f37_lag1)
# X_build$customer_f38_lag1_diffdiv <-ifelse(is.na(X_build$customer_f38_lag1), NA, X_build$customer_f38_lag1_diff / X_build$customer_f38_lag1)
# X_build$customer_f39_lag1_diffdiv <-ifelse(is.na(X_build$customer_f39_lag1), NA, X_build$customer_f39_lag1_diff / X_build$customer_f39_lag1)
# X_build$customer_f40_lag1_diffdiv <-ifelse(is.na(X_build$customer_f40_lag1), NA, X_build$customer_f40_lag1_diff / X_build$customer_f40_lag1)
# X_build$customer_f41_lag1_diffdiv <-ifelse(is.na(X_build$customer_f41_lag1), NA, X_build$customer_f41_lag1_diff / X_build$customer_f41_lag1)
# 
# X_val$customer_f3_lag1_diffdiv  <-ifelse(is.na(X_val$customer_f3_lag1) , NA, X_val$customer_f3_lag1_diff  / X_val$customer_f3_lag1   )
# X_val$customer_f4_lag1_diffdiv  <-ifelse(is.na(X_val$customer_f4_lag1) , NA, X_val$customer_f4_lag1_diff  / X_val$customer_f4_lag1)
# X_val$customer_f6_lag1_diffdiv  <-ifelse(is.na(X_val$customer_f6_lag1) , NA, X_val$customer_f6_lag1_diff  / X_val$customer_f6_lag1)
# X_val$customer_f8_lag1_diffdiv  <-ifelse(is.na(X_val$customer_f8_lag1) , NA, X_val$customer_f8_lag1_diff  / X_val$customer_f8_lag1)
# X_val$customer_f10_lag1_diffdiv <-ifelse(is.na(X_val$customer_f10_lag1), NA, X_val$customer_f10_lag1_diff / X_val$customer_f10_lag1)
# X_val$customer_f11_lag1_diffdiv <-ifelse(is.na(X_val$customer_f11_lag1), NA, X_val$customer_f11_lag1_diff / X_val$customer_f11_lag1)
# X_val$customer_f12_lag1_diffdiv <-ifelse(is.na(X_val$customer_f12_lag1), NA, X_val$customer_f12_lag1_diff / X_val$customer_f12_lag1)
# X_val$customer_f13_lag1_diffdiv <-ifelse(is.na(X_val$customer_f13_lag1), NA, X_val$customer_f13_lag1_diff / X_val$customer_f13_lag1)
# X_val$customer_f14_lag1_diffdiv <-ifelse(is.na(X_val$customer_f14_lag1), NA, X_val$customer_f14_lag1_diff / X_val$customer_f14_lag1)
# X_val$customer_f15_lag1_diffdiv <-ifelse(is.na(X_val$customer_f15_lag1), NA, X_val$customer_f15_lag1_diff / X_val$customer_f15_lag1)
# X_val$customer_f17_lag1_diffdiv <-ifelse(is.na(X_val$customer_f17_lag1), NA, X_val$customer_f17_lag1_diff / X_val$customer_f17_lag1)
# X_val$customer_f18_lag1_diffdiv <-ifelse(is.na(X_val$customer_f18_lag1), NA, X_val$customer_f18_lag1_diff / X_val$customer_f18_lag1)
# X_val$customer_f21_lag1_diffdiv <-ifelse(is.na(X_val$customer_f21_lag1), NA, X_val$customer_f21_lag1_diff / X_val$customer_f21_lag1)
# X_val$customer_f22_lag1_diffdiv <-ifelse(is.na(X_val$customer_f22_lag1), NA, X_val$customer_f22_lag1_diff / X_val$customer_f22_lag1)
# X_val$customer_f25_lag1_diffdiv <-ifelse(is.na(X_val$customer_f25_lag1), NA, X_val$customer_f25_lag1_diff / X_val$customer_f25_lag1)
# X_val$customer_f26_lag1_diffdiv <-ifelse(is.na(X_val$customer_f26_lag1), NA, X_val$customer_f26_lag1_diff / X_val$customer_f26_lag1)
# X_val$customer_f27_lag1_diffdiv <-ifelse(is.na(X_val$customer_f27_lag1), NA, X_val$customer_f27_lag1_diff / X_val$customer_f27_lag1)
# X_val$customer_f28_lag1_diffdiv <-ifelse(is.na(X_val$customer_f28_lag1), NA, X_val$customer_f28_lag1_diff / X_val$customer_f28_lag1)
# X_val$customer_f30_lag1_diffdiv <-ifelse(is.na(X_val$customer_f30_lag1), NA, X_val$customer_f30_lag1_diff / X_val$customer_f30_lag1)
# X_val$customer_f32_lag1_diffdiv <-ifelse(is.na(X_val$customer_f32_lag1), NA, X_val$customer_f32_lag1_diff / X_val$customer_f32_lag1)
# X_val$customer_f34_lag1_diffdiv <-ifelse(is.na(X_val$customer_f34_lag1), NA, X_val$customer_f34_lag1_diff / X_val$customer_f34_lag1)
# X_val$customer_f35_lag1_diffdiv <-ifelse(is.na(X_val$customer_f35_lag1), NA, X_val$customer_f35_lag1_diff / X_val$customer_f35_lag1)
# X_val$customer_f36_lag1_diffdiv <-ifelse(is.na(X_val$customer_f36_lag1), NA, X_val$customer_f36_lag1_diff / X_val$customer_f36_lag1)
# X_val$customer_f37_lag1_diffdiv <-ifelse(is.na(X_val$customer_f37_lag1), NA, X_val$customer_f37_lag1_diff / X_val$customer_f37_lag1)
# X_val$customer_f38_lag1_diffdiv <-ifelse(is.na(X_val$customer_f38_lag1), NA, X_val$customer_f38_lag1_diff / X_val$customer_f38_lag1)
# X_val$customer_f39_lag1_diffdiv <-ifelse(is.na(X_val$customer_f39_lag1), NA, X_val$customer_f39_lag1_diff / X_val$customer_f39_lag1)
# X_val$customer_f40_lag1_diffdiv <-ifelse(is.na(X_val$customer_f40_lag1), NA, X_val$customer_f40_lag1_diff / X_val$customer_f40_lag1)
# X_val$customer_f41_lag1_diffdiv <-ifelse(is.na(X_val$customer_f41_lag1), NA, X_val$customer_f41_lag1_diff / X_val$customer_f41_lag1)

sapply(X_build[, feature.names], class)
feature.names     <- names(X_build[,-which(names(X_build) %in% c("customer_id", "date" ,"target","id","f_19","f_29"
                                                                 ,"customer_target_median1"
                                                                 ,"customer_target_lag_target_diff_flag"
                                                                 ,"market_target_median1","roll"
                                                                 ,"f0_target_lag1"               
                                                                 ,"f0_target_lead1"
                                                                 ,"f2_target_lag1"
                                                                 ,"f2_target_lead1"
                                                                 ,"marketf33_target_lag1"
                                                                 ,"marketf23_target_lag1"
                                                                 ,"f33_f23_target_lag1" 
                                                                    ,"customer_f3_lag1"
                                                                    ,"customer_f4_lag1" 
                                                                    ,"customer_f6_lag1"
                                                                    ,"customer_f8_lag1" 
                                                                    ,"customer_f10_lag1"
                                                                    ,"customer_f11_lag1"
                                                                    ,"customer_f12_lag1"
                                                                    ,"customer_f13_lag1"
                                                                    ,"customer_f14_lag1"
                                                                    ,"customer_f15_lag1"
                                                                    ,"customer_f17_lag1"
                                                                    ,"customer_f18_lag1"
                                                                    ,"customer_f21_lag1"
                                                                    ,"customer_f22_lag1"
                                                                    ,"customer_f25_lag1"
                                                                    ,"customer_f26_lag1"
                                                                    ,"customer_f27_lag1"
                                                                    ,"customer_f28_lag1"
                                                                    ,"customer_f30_lag1"
                                                                    ,"customer_f32_lag1"
                                                                    ,"customer_f34_lag1"
                                                                    ,"customer_f35_lag1"
                                                                    ,"customer_f36_lag1"
                                                                    ,"customer_f37_lag1"
                                                                    ,"customer_f38_lag1"
                                                                    ,"customer_f39_lag1"
                                                                    ,"customer_f40_lag1"
                                                                    ,"customer_f41_lag1"
                                                                 ,"customer_f13_lag1_diff" 
                                                                 , "customer_f15_lag1_diff" 
                                                                 ,"customer_f30_lag1_diff"  
                                                                 ,"customer_f13_lag1_div"        
                                                                 , "customer_f15_lag1_div" 
                                                                 , "customer_f30_lag1_div"))])
# char.feature.names <- c("market","f_16","f_17","f_19","f_21","f_22","f_23","f_24","f_25","f_26","f_40","f_41","f41_count","f2_4","f_411","cust_market_lag_date","row_max","row_min")
# feature.names     <- setdiff(feature.names,char.feature.names)

# [1] "market"                "f_0"                   "f_1"                   "f_2"                   "f_3"                   "f_4"                  
# [7] "f_5"                   "f_6"                   "f_7"                   "f_8"                   "f_9"                   "f_10"                 
# [13] "f_11"                  "f_12"                  "f_13"                  "f_14"                  "f_15"                  "f_16"                 
# [19] "f_17"                  "f_18"                  "f_20"                  "f_21"                  "f_22"                  "f_23"                 
# [25] "f_24"                  "f_25"                  "f_26"                  "f_27"                  "f_28"                  "f_30"                 
# [31] "f_31"                  "f_32"                  "f_33"                  "f_34"                  "f_35"                  "f_36"                 
# [37] "f_37"                  "f_38"                  "f_39"                  "f_40"                  "f_41"                  "market_target_lag1"   
# [43] "market_target_lead1"   "customer_target_lag1"  "customer_target_lead1" "f33_target_lag1"       "f33_target_lead1"      "f23_target_lag1"      
# [49] "f23_target_lead1" 

X_build[,feature.names][is.na(X_build[,feature.names])] <- -1
X_val[,feature.names][is.na(X_val[,feature.names])]   <- -1



cv = 1
nround.cv =  2000
printeveryn = 200
seed = 2017
bags = 1

## for all remaining models, use same parameters 

param <- list(  "objective"        = "reg:linear",
                "booster"          = "gbtree",
                "tree_method"      = "exact",
                "eval_metric"      = "rmse",
                "nthread"          = 32,     
                "max_depth"        = 7,     
                "eta"              = 0.05,
                "subsample"        = 0.8,  
                "colsample_bytree" = 0.7,  
                "min_child_weight" = 3     
                
)


cat(cv, "-fold Cross Validation\n", sep = "")


pred_cv <- rep(0, nrow(X_val[,feature.names]))

set.seed(seed)
for (i in 1:cv)
  
{
  cat(i ,"fold Processing\n")
  
  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=X_build$target)
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=X_val$target)
  watchlist <- list( val = dval,train = dtrain)
  for (b in 1:bags) {
    cat(b ," - bag Processing\n")
    seed = b + seed
    set.seed(seed)
    cat("X_build training Processing\n")
    XGModel <- xgb.train(   params              = param,
                            #feval               = xgb.metric.qwk,
                            data                = dtrain,
                            watchlist           = watchlist,
                            nrounds             = nround.cv ,
                            print_every_n       = printeveryn,
                            verbose             = TRUE, 
                            #maximize            = TRUE,
                            set.seed            = seed
    )
    
    cat("X_val prediction Processing\n")
    pred_bag  <- predict(XGModel, data.matrix(X_val[,feature.names]))
    cat("bag - ",b, "score ", qwKappa(X_val$target,pred_bag ,0,20),"\n")
    pred_cv <- pred_cv + pred_bag
  }
  
  pred_cv <- pred_cv / bags
  cat("fold - ",i, " score ", qwKappa(X_val$target,pred_cv ,0,20),"\n")
  
  # pred_cv_predictions  <- data.frame(customer_id=X_val$customer_id, date=X_val$date, target = pred_cv)
  # write.csv(pred_cv_predictions, './submissions/prav.xgb04.bags5Prob.fold1.csv', row.names=FALSE, quote = FALSE)
}

# fold -  1  score  0.7468631 

# [1]	  val-kappa:0.009183	train-kappa:0.009096 
# [51]	val-kappa:0.735171	train-kappa:0.764920 
# [101]	val-kappa:0.736964	train-kappa:0.769482 
# [151]	val-kappa:0.738029	train-kappa:0.771456 
# [201]	val-kappa:0.738238	train-kappa:0.772825 
# [251]	val-kappa:0.738529	train-kappa:0.773930 
# [301]	val-kappa:0.738823	train-kappa:0.774836 
# [351]	val-kappa:0.738873	train-kappa:0.775812 
# [401]	val-kappa:0.739003	train-kappa:0.776572 
# [451]	val-kappa:0.739377	train-kappa:0.777396 
# [500]	val-kappa:0.739423	train-kappa:0.778147 
# X_val prediction Processing
# bag -  1 score  0.7394225 

# bag -  1 score  0.7437818

# #########################################################################################################
# Bagging - Full train
# #########################################################################################################
head(trainingSet[, feature.names])
trainingSet <- as.data.frame(trainingSet)
testingSet <- as.data.frame(testingSet)

trainingSet[,feature.names][is.na(trainingSet[,feature.names])] <- -1
testingSet[,feature.names][is.na(testingSet[,feature.names])]   <- -1



dtrain<-xgb.DMatrix(data=data.matrix(trainingSet[, feature.names]),label=trainingSet$target)
watchlist <- list( train = dtrain)

fulltrainnrounds = 1.2 * nround.cv

bags = 5
predfull_test <- rep(0, nrow(testingSet[,feature.names]))


for (b in 1:bags) {
  cat(b ," - bag Processing\n")
  seed = b + seed
  set.seed(seed)
  cat("Bagging Full TrainingSet training\n")
  XGModelFulltrain <- xgb.train(    params              = param,
                                    feval               = xgb.metric.qwk,
                                    data                = dtrain,
                                    watchlist           = watchlist,
                                    nrounds             = fulltrainnrounds,
                                    print_every_n       = printeveryn,
                                    verbose             = TRUE, 
                                    maximize            = TRUE,
                                    set.seed            = seed
  )
  cat("Bagging Full Model prediction Processing\n")
  
  pred_test         <- predict(XGModelFulltrain, data.matrix(testingSet[,feature.names]))
  
  predfull_test <- predfull_test + pred_test
  
}

predfull_test <- predfull_test / bags
testfull_predictions  <- data.frame(id=testingSet$id, target = predfull_test)
write.csv(testfull_predictions, './submissions/prav.xgb04.bags5Prob.full.csv', row.names=FALSE, quote = FALSE)


max(testfull_predictions$target)
min(testfull_predictions$target)

# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

testfull_predictions_submission  <- data.frame(id=testingSet$id, target = round(predfull_test))

sapply(testfull_predictions_submission, class)

testfull_predictions_submission$id <- as.integer(testfull_predictions_submission$id)
testfull_predictions_submission$target <- ifelse(testfull_predictions_submission$target < 0,0, testfull_predictions_submission$target)
write.csv(testfull_predictions_submission, './submissions/prav.xgb03.bags5Prob.full.submission.csv', row.names=FALSE, quote = FALSE)

#submission <- read_csv("./input/sample_submission_v2.csv")

max(testfull_predictions_submission$target)
min(testfull_predictions_submission$target)
summary(testfull_predictions_submission$target)
# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

write.csv(X_build,"./input/X_build.csv",  row.names=FALSE, quote = FALSE)
write.csv(X_val,"./input/X_val.csv",  row.names=FALSE, quote = FALSE)

write.csv(trainingSet,"./input/trainingSet.csv",  row.names=FALSE, quote = FALSE)
write.csv(testingSet,"./input/testingSet.csv",  row.names=FALSE, quote = FALSE)


# [1]	val-kappa:0.007384	train-kappa:0.007354 
# [51]	val-kappa:0.713305	train-kappa:0.717260 
# [101]	val-kappa:0.715791	train-kappa:0.723420 
# [151]	val-kappa:0.716279	train-kappa:0.726398 
# [201]	val-kappa:0.716696	train-kappa:0.728794 
# [251]	val-kappa:0.716900	train-kappa:0.730811 
# [300]	val-kappa:0.717151	train-kappa:0.732532 

# > impMatrix
# Feature         Gain        Cover   Frequency
# 1     f_13 5.002763e-01 0.0813791110 0.056634861
# 2     f_15 2.897913e-01 0.0506341467 0.048378628
# 3     f_30 7.575351e-02 0.0275626535 0.034310442
# 4      f_3 2.452477e-02 0.0462173450 0.040303453
# 5      f_8 2.452161e-02 0.0588678646 0.050388369
# 6      f_5 9.647086e-03 0.0153320655 0.020151726
# 7     f_26 8.405660e-03 0.0302612894 0.044250511
# 8     f_38 7.965916e-03 0.0270394453 0.027231084
# 9     f_21 5.982836e-03 0.0279953053 0.049030436
# 10    f_17 5.896746e-03 0.0353747399 0.033839691
# 11    f_36 5.041963e-03 0.0606478080 0.042095924
# 12    f_14 4.364162e-03 0.0232743998 0.033332730
# 13    f_12 3.541342e-03 0.0484176081 0.039669751
# 14    f_22 3.424487e-03 0.0318741931 0.030236642
# 15     f_4 3.201084e-03 0.0310018713 0.036845250
# 16    f_18 2.687615e-03 0.0374790077 0.040665568
# 17    f_11 2.548746e-03 0.0295671336 0.040919049
# 18    f_28 2.508027e-03 0.0181389586 0.021907986
# 19  market 2.109369e-03 0.0458247032 0.027575094
# 20    f_40 2.027060e-03 0.0281279827 0.026778440
# 21    f_25 1.943099e-03 0.0315850937 0.027394036
# 22    f_41 1.753365e-03 0.0275438726 0.026162843
# 23    f_35 1.615707e-03 0.0118146681 0.026470641
# 24    f_32 1.565997e-03 0.0100651788 0.025257555
# 25    f_27 1.501921e-03 0.0390894811 0.027339719
# 26     f_0 1.111469e-03 0.0104432260 0.017417755
# 27     f_6 9.860081e-04 0.0070539143 0.009125310
# 28    f_23 8.333347e-04 0.0199855052 0.013869023
# 29    f_33 6.540700e-04 0.0101868670 0.014665677
# 30    f_31 6.471507e-04 0.0113641396 0.010193551
# 31    f_37 5.352416e-04 0.0069230597 0.006300809
# 32    f_20 4.834356e-04 0.0083714203 0.009903858
# 33    f_24 4.108123e-04 0.0068574832 0.007224204
# 34    f_39 3.330422e-04 0.0119017536 0.005486050
# 35     f_2 3.081609e-04 0.0063182645 0.005522261
# 36    f_10 2.857567e-04 0.0097992094 0.006101646
# 37    f_16 2.194839e-04 0.0028451657 0.004291068
# 38     f_9 1.994340e-04 0.0032227041 0.003132299
# 39    f_29 1.649569e-04 0.0047451156 0.003639261
# 40     f_1 1.302645e-04 0.0007407246 0.003820318
# 41    f_34 9.775559e-05 0.0041255210 0.002136481

# Full training

dtrain<-xgb.DMatrix(data=data.matrix(trainingSet[, feature.names]),label=trainingSet$target)
watchlist <- list( train = dtrain)

fulltrainnrounds = 1.2 * nround.cv

# #########################################################################################################
# Full train
# #########################################################################################################

cat("Full TrainingSet training\n")
XGModelFulltrain <- xgb.train(    params              = param,
                                  feval               = xgb.metric.qwk,#evalerror, #xgb.metric.mae
                                  data                = dtrain,
                                  watchlist           = watchlist,
                                  nrounds             = fulltrainnrounds,
                                  print_every_n       = printeveryn,
                                  verbose             = TRUE,
                                  maximize            = TRUE,
                                  set.seed            = seed
)
cat("Full Model prediction Processing\n")

predfull_test         <- predict(XGModelFulltrain, data.matrix(testingSet[,feature.names]))
testfull_predictions  <- data.frame(id=testingSet$id, target = round(predfull_test))
write.csv(testfull_predictions, './submissions/prav.xgb01.full.csv', row.names=FALSE, quote = FALSE)

# #########################################################################################################
# Bagging - Full train
# #########################################################################################################
bags = 5
ensemble <- rep(0, nrow(testingSet[,testfeature.names]))


for (b in 1:bags) {
  cat(b ," - bag Processing\n")
  seed = b + seed
  set.seed(seed)
  cat("Bagging Full TrainingSet training\n")
  XGModelFulltrain <- xgb.train(    params              = param,
                                    feval               = xgb.metric.log.mae,
                                    data                = dtrain,
                                    watchlist           = watchlist,
                                    nrounds             = fulltrainnrounds,
                                    print_every_n       = printeveryn,
                                    verbose             = TRUE, 
                                    #maximize            = TRUE,
                                    set.seed            = seed
  )
  cat("Bagging Full Model prediction Processing\n")
  
  predfull_test         <- predict(XGModelFulltrain, data.matrix(testingSet[,testfeature.names]))
  
  ensemble <- ensemble + predfull_test
  
}

ensemble <- ensemble / bags
testfull_predictions  <- data.frame(id=testingSet$id, loss = exp(ensemble))
write.csv(testfull_predictions, './submissions/prav.xgb03.bags5.full.csv', row.names=FALSE, quote = FALSE)


max(testfull_predictions$target)
min(testfull_predictions$target)

# #########################################################################################################
# Bagging - Full train
# #########################################################################################################



# head(testfull_predictions)

############################################################################################
model = xgb.dump(XGModel, with_stats=TRUE)

names = dimnames(X_build[,feature.names])[[2]]
importance_matrix = xgb.importance( names , model = XGModel) 
#xgb.plot.importance(importance_matrix)
impMatrix <- as.data.frame(importance_matrix)
# impMatrix[1:12,1]
# impMatrix[,1]
ImpFeature <- impMatrix[,1]


# #########################################################################################################

for(i in seq(0.1,0.9,by=0.1))
{
  
  cat("score adjustment factor - ",i, " score ", qwKappa(X_val$target, pred_cv+i ,0,20),"\n")
}




train_df = read_csv("./input/TrainingSet01.csv")
test_df = read_csv("./input/TestingSet01.csv")


# train_df = train # .groupby('id').first().reset_index()

train_features_06 = read_csv("./input/train_features_06.csv")
test_features_06  = read_csv("./input/test_features_06.csv")


train_df = left_join(train_df, train_features_06, by = 'id')
test_df = left_join(test_df, test_features_06, by = 'test_id')



train_features_07 = read_csv("./input/train_features_07.csv")
test_features_07  = read_csv("./input/test_features_07.csv")


train_df = left_join(train_df, train_features_07, by = 'id')
test_df  = left_join(test_df, test_features_07, by = 'test_id')

train_features_11 = read_csv("./input/train_question_freq_features.csv")
test_features_11  = read_csv("./input/test_question_freq_features.csv")

train_features_11$is_duplicate <- NULL
names(test_features_11)[1] <- "test_id"

train_df = left_join(train_df, train_features_11, by = 'id')
test_df  = left_join(test_df, test_features_11, by = 'test_id')



CV_Schema = read_csv("./CVSchema/Prav_CVindices_5folds.csv")

train_df = left_join(train_df, CV_Schema, how = 'left', by = c('id','qid1','qid2'))

train_features_52 = read_csv("./input/train_features_52.csv")
test_features_52  = read_csv("./input/test_features_52.csv")

train_df = left_join(train_df, train_features_52, by = 'id')
test_df  = left_join(test_df, test_features_52, by = 'test_id')

train_features_53 = read_csv("./input/train_features_53.csv")
test_features_53  = read_csv("./input/test_features_53.csv")

train_df = left_join(train_df, train_features_53, by = 'id')
test_df  = left_join(test_df, test_features_53, by = 'test_id')


feature.names     <- c('q_dist_soundex'
                       ,'q_dist_jarowinkler'
                       ,'q_dist_lcs'
                       ,'q1_nchar'
                       ,'q2_nchar'
                       ,'q1_EndsWith_q2'
                       ,'q1_EndsWith_Sound_q2'
                       ,'q_nchar_ratios_pmax'
                       ,'q_nchar_pmin'
                       ,'q_nchar_pmax'
                       ,'q1_StartsWith_Sound_q2'
                       ,'q1_StartsWith_q2'
                       ,'q1_nwords'
                       ,'q1_nwords_matched_q2'
                       ,'q1_MatchedWords_ratio_to_q2'
                       ,'q2_nwords'
                       ,'q2_nwords_matched_q1'
                       ,'q2_MatchedWords_ratio_to_q1'
                       ,'q_MatchedWords_ratio_to_q_ratios_pmax'
                       ,'q_MatchedWords_ratio_to_q_pmin'
                       ,'q_MatchedWords_ratio_to_q_pmax'
                       ,'q_2gram_jaccard'
                       ,'q_3gram_jaccard'
                       ,'q_4gram_jaccard'
                       ,'q_5gram_jaccard'
                       ,'q_dist_lv'
                       ,'q_dist_cosine'
                       ,'q1_PunctCount'
                       ,'q2_PunctCount'
                       ,'q_PunctCount_ratios_pmax'
                       ,'q_PunctCount_pmin'
                       ,'q_PunctCount_pmax'
                       ,'len_q1'
                       ,'len_q2'
                       ,'diff_len'
                       ,'len_char_q1'
                       ,'len_char_q2'
                       ,'len_word_q1'
                       ,'len_word_q2'
                       ,'common_words'
                       ,'fuzz_qratio'
                       ,'fuzz_WRatio'
                       ,'fuzz_partial_ratio'
                       ,'fuzz_partial_token_set_ratio'
                       ,'fuzz_partial_token_sort_ratio'
                       ,'fuzz_token_set_ratio'
                       ,'fuzz_token_sort_ratio'
                       ,'word_match'
                       ,'tfidf_word_match'
                       ,'cosine_distance'
                       ,'cityblock_distance'
                       ,'jaccard_distance'
                       ,'canberra_distance'
                       ,'euclidean_distance'
                       ,'minkowski_distance'
                       ,'braycurtis_distance'
                       ,'skew_q1vec'
                       ,'skew_q2vec'
                       ,'kur_q1vec'
                       ,'kur_q2vec',
                       'zbigrams_common_count',
                       'zbigrams_common_ratio',
                       'z_noun_match',
                       'z_match_ratio',
                       'z_word_match',                    
                       'q1_freq',
                       'q2_freq'
                       , "q1qm"          , "q2qm"         ,   "qmatchqm"        ,"qratioqm"       , "q1math"        
                       , "q2math"        , "qmatchmath"   ,   "qratiomath"      ,"q1stop"         , "q2stop"         , "qmatchstop"     
                       , "qratiostop"    , "q1num"        ,   "q2num"           ,"qmatchnum"      , "qrationum"      , "q1what"         
                       , "q2what"        , "qmatchwhat"   ,   "qratiowhat"      ,"q1when"         , "q2when"         , "qmatchwhen"     
                       , "qratiowhen"    , "q1why"        ,   "q2why"           ,"qmatchwhy"      , "qratiowhy"      , "q1how"          
                       , "q2how"         , "qmatchhow"    ,   "qratiohow"       ,"q1_anger"       , "q1_anticipation", "q1_disgust"     
                       , "q1_fear"       , "q1_joy"       ,   "q1_sadness"      ,"q1_surprise"    , "q1_trust"       , "q1_negative"    
                       , "q1_positive"   , "q2_anger"     ,   "q2_anticipation" ,"q2_disgust"     , "q2_fear"        , "q2_joy"         
                       , "q2_sadness"    , "q2_surprise"  ,   "q2_trust"        ,"q2_negative"    , "q2_positive" )



train_df <- as.data.frame(train_df)
test_df <- as.data.frame(test_df)


train_df[is.na(train_df)] <- 0
test_df[is.na(test_df)]   <- 0



##################################################################################################################################
##################################################################################################################################
#for column in features_to_use:
#    print(column)
#    print(' AUC:', roc_auc_score(train_df['is_duplicate'], train_df[column]))
##################################################################################################################################

gc()

cv = 5
nround.cv =  3510 
printeveryn = 500
seed = 2016

## for all remaining models, use same parameters 

param <- list(  "objective"        = "binary:logistic", 
                "booster"          = "gbtree",
                "eval_metric"      = "logloss",
                "nthread"          = 25,     
                "max_depth"        = 4,     
                "eta"              = 0.02, 
                "subsample"        = 0.7,  
                "colsample_bytree" = 0.7,  
                "min_child_weight" = 1     
                
)


cat(cv, "-fold Cross Validation\n", sep = "")

set.seed(seed)
for (i in 1:cv)
  
{
  cat(i ,"fold Processing\n")
  
  cat("X_build fold Processing\n")
  X_build <- subset(train_df, CVindices != i, select = -c( CVindices))
  cat("X_val fold Processing\n")
  X_val   <- subset(train_df, CVindices == i, select = -c( CVindices)) 
  
  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=X_build$is_duplicate , missing = 0)
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=X_val$is_duplicate , missing = 0 )
  watchlist <- list( val = dval,train = dtrain)
  
  cat("X_build training Processing\n")
  XGModel <- xgb.train(   params              = param,
                          data                = dtrain,
                          watchlist           = watchlist,
                          nrounds             = nround.cv ,
                          print.every.n       = printeveryn,
                          verbose             = TRUE, 
                          #maximize            = TRUE,
                          set.seed            = seed
  )
  
  cat("X_val prediction Processing\n")
  pred_cv  <- predict(XGModel, data.matrix(X_val[,feature.names]))
  val_predictions <- data.frame(id=X_val$id, is_duplicate = pred_cv)
  
  cat("CV TestingSet prediction Processing\n")
  pred_test  <- predict(XGModel, data.matrix(test_df[,feature.names]))
  test_predictions <- data.frame(test_id=test_df$test_id, is_duplicate = pred_test)
  
  if(i == 1)
  {
    write.csv(val_predictions,  './submissions/prav.xgb54.fold1.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb54.fold1-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 2)
  {
    write.csv(val_predictions,  './submissions/prav.xgb54.fold2.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb54.fold2-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 3)
  {
    write.csv(val_predictions,  './submissions/prav.xgb54.fold3.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb54.fold3-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 4)
  {
    write.csv(val_predictions,  './submissions/prav.xgb54.fold4.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb54.fold4-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 5)
  {
    write.csv(val_predictions,  './submissions/prav.xgb54.fold5.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb54.fold5-test.csv', row.names=FALSE, quote = FALSE)
  }
  
}

# Full training

dtrain<-xgb.DMatrix(data=data.matrix(train_df[, feature.names]),label=train_df$is_duplicate , missing = 0)
watchlist <- list( train = dtrain)

fulltrainnrounds = 1.2 * nround.cv

# #########################################################################################################
# Full train
# #########################################################################################################

cat("Full TrainingSet training\n")
XGModelFulltrain <- xgb.train(   params              = param,
                                 data                = dtrain,
                                 watchlist           = watchlist,
                                 nrounds             = fulltrainnrounds ,
                                 print.every.n       = printeveryn,
                                 verbose             = TRUE, 
                                 #maximize            = TRUE,
                                 set.seed            = seed
)
cat("Full Model prediction Processing\n")

predfull_test         <- predict(XGModelFulltrain, data.matrix(test_df[,feature.names]))
testfull_predictions  <- data.frame(test_id=test_df$test_id, is_duplicate = predfull_test)
write.csv(testfull_predictions, './submissions/prav.xgb54.full.csv', row.names=FALSE, quote = FALSE)

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
                                    print.every.n       = printeveryn,
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
write.csv(testfull_predictions, './submissions/prav.xgb01.bags5.full.csv', row.names=FALSE, quote = FALSE)




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
# impMatrix[1:12,1]
# impMatrix[,1]
ImpFeature <- impMatrix[,1]


# #########################################################################################################


x_test = test_df[features_to_use].apply(pd.to_numeric)

import xgboost as xgb


param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.02
param['max_depth'] = 4
param['silent'] = 1
param['eval_metric'] = "logloss"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 25
param['seed'] = 2017
param['print_every_n'] = 50
num_rounds = 10010
plst = list(param.items())


def train_xgboost(i):
  print('Fold ', i , ' Processing')
X_build = train_df[train_df['CVindices'] != i] # 636112
#X_build = X_build.groupby('id').first().reset_index() # 331085
X_val   = train_df[train_df['CVindices'] == i]

print(X_build.shape) # (404290, 6)
print(X_val.shape)  # (2345796, 3)

X_train = X_build[features_to_use]
X_valid = X_val[features_to_use]

X_train = X_train.fillna(0) 
X_valid = X_valid.fillna(0)

X_train = X_train.apply(pd.to_numeric)
X_valid = X_valid.apply(pd.to_numeric)

X_trainy = X_build['is_duplicate']
X_validy = X_val['is_duplicate']

X_trainy = X_trainy.apply(pd.to_numeric).values
X_validy = X_validy.apply(pd.to_numeric).values

xgbbuild = xgb.DMatrix(X_train, label=X_trainy, missing = 0)
xgbval = xgb.DMatrix(X_valid, label=X_validy, missing = 0)
watchlist = [ (xgbbuild,'build'), (xgbval, 'valid') ]

model = xgb.train(plst, 
                  xgbbuild, 
                  num_rounds, 
                  watchlist, 
                  verbose_eval = 1000 #,
                  #early_stopping_rounds=20
)
pred_cv = model.predict(xgb.DMatrix(X_valid))
pred_cv = pd.DataFrame(pred_cv)
pred_cv.columns = ["is_duplicate"]
pred_cv["id"] = X_val.id.values
pred_cv = pred_cv[['id','is_duplicate']]
sub_valfile = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.xgb54.fold' + str(i) + '.csv'
pred_cv.to_csv(sub_valfile, index=False)

xgtest = xgb.DMatrix(x_test, missing = 0)
pred_test = model.predict(xgtest)
pred_test = pd.DataFrame(pred_test)
pred_test.columns = ["is_duplicate"]
pred_test["test_id"] = test_df.test_id.values
pred_test = pred_test[['test_id','is_duplicate']]
sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.xgb54.fold' + str(i) + '-test' + '.csv'
pred_test.to_csv(sub_file, index=False)
del pred_cv
del pred_test

##########################################################################################
# Full model training
########################################################################################## 

fullnum_rounds = int(num_rounds * 1.2)

def fulltrain_xgboost(nbags):
  predfull_test = np.zeros(x_test.shape[0]) 
xgbtrain = xgb.DMatrix(train_df[features_to_use].apply(pd.to_numeric), label=train_df['is_duplicate'].apply(pd.to_numeric).values, missing = 0)
watchlist = [ (xgbtrain,'train') ]
for j in range(1,nbags+1):
  print('bag ', j , ' Processing')
fullmodel = xgb.train(plst, 
                      xgbtrain, 
                      fullnum_rounds, 
                      watchlist,
                      verbose_eval = 1000,
)
xgtest = xgb.DMatrix(x_test, missing = 0)
predfull_test += fullmodel.predict(xgtest)        
predfull_test/= nbags
predfull_test = pd.DataFrame(predfull_test)
predfull_test.columns = ["is_duplicate"]
predfull_test["test_id"] = test_df.test_id.values
predfull_test = predfull_test[['test_id','is_duplicate']]
sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.xgb54.full' + '.csv'
predfull_test.to_csv(sub_file, index=False)

folds = 5
nbags = 2
i = 1
if __name__ == '__main__':
  #for i in range(1, folds+1):
  train_xgboost(i)
fulltrain_xgboost(nbags)
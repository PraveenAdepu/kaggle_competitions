library(tidyverse)
library(lubridate)
library(magrittr)
library(text2vec)
library(tokenizers)
library(stopwords)
library(xgboost)
library(Matrix)
set.seed(2018)

#---------------------------
cat("Loading data...\n")
tr <- read_csv("./input/train.csv") 
te <- read_csv("./input/test.csv")


Prav_5fold_CVindices <- read_csv("./input/Prav_5folds_CVindices_weekdayStratified.csv") 

tr <- inner_join(tr, Prav_5fold_CVindices, by="item_id")


train_FE01 <- read_csv("./input/Prav_train_FE_01.csv")
test_FE01  <- read_csv("./input/Prav_test_FE_01.csv")

tr <- inner_join(tr, train_FE01, by="item_id")
te <- inner_join(te, test_FE01, by="item_id")

traintest_FE02 <- read_csv("./input/traintest_FE_022.csv")

tr <- left_join(tr, traintest_FE02, by="user_id")
te <- left_join(te, traintest_FE02, by="user_id")

train_FE03 <- read_csv("./input/Prav_train_FE_03.csv")
test_FE03  <- read_csv("./input/Prav_test_FE_03.csv")

tr <- inner_join(tr, train_FE03, by="item_id")
te <- inner_join(te, test_FE03, by="item_id")


head(tr)
#---------------------------
cat("Preprocessing...\n")
tri <- 1:nrow(tr)
y <- tr$deal_probability

tr_te <- tr %>%
  select(-deal_probability,-CVindices) %>%
  bind_rows(te) %>%
  mutate(
         user_type = as.factor(user_type),
         category_name = category_name %>% factor() %>% as.integer(),
         parent_category_name = parent_category_name %>% factor() %>% as.integer(),
         region = region %>% factor() %>% as.integer(),
         param_1 = param_1 %>% factor() %>% as.integer(),
         param_2 = param_2 %>% factor() %>% as.integer(),
         param_3 = param_3 %>% factor() %>% fct_lump(prop = 0.00005) %>% as.integer(),
         city = city %>% factor() %>% fct_lump(prop = 0.0003) %>% as.integer(),
         user_id = user_id %>% factor() %>% fct_lump(prop = 0.000025) %>% as.integer(),
         #price = log1p(price),
         txt = paste(title, description, sep = " ")
        ) %>%
  select(-item_id, -image, -title, -description, -activation_date) %>%
  replace_na(list(image_top_1 = -99,
                  param_1 = -99, param_2 = -99, param_3 = -99
                  )) %T>%
  glimpse()

#rm(tr, te); gc()


tr_te[is.na(tr_te)] <- -99
#---------------------------
cat("Parsing text...\n")
it <- tr_te %$%
  str_to_lower(txt) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  tokenize_word_stems(language = "russian") %>% 
  itoken()

vect <- create_vocabulary(it, ngram = c(1, 1), stopwords = stopwords("ru")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.4, vocab_term_max = 6500) %>% 
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <-  create_dtm(it, vect) %>% 
  fit_transform(m_tfidf)

rm(it, vect, m_tfidf); gc()

dim(tfidf)
dim(tr_te)

#---------------------------
cat("Preparing data...\n")
X <- tr_te %>% 
  select(-txt) %>% 
  sparse.model.matrix(~ . - 1, .) %>% 
  cbind(tfidf)

rm(tr_te, tfidf); gc()

dim(X)

dtest <- xgb.DMatrix(data = X[-tri, ])
X <- X[tri, ]; gc()

dim(X)
dim(dtest)
tr$rowIndex <-  1:nrow(tr)



##################################################################################

# order of columns are matching 
##################################################################################
Model_validation = FALSE

if(Model_validation){
  cv=1} else {
    cv=5}

# cv          = 5
bags        = 2
nround.cv   = 2000
printeveryn = 250
seed        = 201805

## for all remaining models, use same parameters 

param <- list(  "objective"        = "reg:logistic",
                "booster"          = "gbtree",
                "eval_metric"      = "rmse",
                "nthread"          = 30,     
                "max_depth"        = 8,     
                "eta"              = 0.05, 
                "subsample"        = 0.75,  
                "colsample_bytree" = 0.7,
                "alpha"            = 2.0,
                "gamma"            = 0,
                "lambda"           = 0,
                "min_child_weight" = 9     
                
)


cols <- colnames(X)

cat(cv, "-fold Cross Validation\n", sep = "")

set.seed(seed)

for (i in 1:cv)

  
{
  
  cat(i ,"fold Processing\n")
  
  cat("X_build fold Processing\n")
  X_build <- subset(tr, CVindices != i, select = c( item_id, rowIndex))
  cat("X_val fold Processing\n")
  X_val   <- subset(tr, CVindices == i, select = c( item_id,rowIndex)) 
  
  tri <- X_build$rowIndex %>% c()
  
  dtrain <- xgb.DMatrix(data = X[tri, ], label = y[tri])
  dval   <- xgb.DMatrix(data = X[-tri, ], label = y[-tri])
  watchlist <- list( val = dval,train = dtrain)
  
  
  
  pred_cv_bags   <- rep(0, nrow(X_val))
  pred_test_bags <- rep(0, nrow(te))
  
  for (b in 1:bags) 
  {
    seed = seed + b
    cat(seed , " - Random Seed\n ")
    cat(b ," - bag Processing\n")
    cat("X_build training Processing\n")
    XGModel <- xgb.train(   params              = param,
                            data                = dtrain,
                            watchlist           = watchlist,
                            nrounds             = nround.cv ,
                            print_every_n       = printeveryn,
                            verbose             = TRUE,
                            set.seed            = seed
    )
    cat("X_val prediction Processing\n")
    pred_cv    <- predict(XGModel, X[-tri, ]) #, ntreelimit = 900
    cat("CV TestingSet prediction Processing\n")
    pred_test  <- predict(XGModel, dtest)
    
    pred_cv_bags   <- pred_cv_bags + pred_cv 
    pred_test_bags <- pred_test_bags + pred_test 
  }
  pred_cv_bags   <- pred_cv_bags / bags
  pred_test_bags <- pred_test_bags / bags
  
  cat("CV Fold-", i, " ", metric, ": ", score(y[-tri], pred_cv_bags, metric), "\n", sep = "")
  
  val_predictions <- data.frame(item_id=X_val$item_id, deal_probability = pred_cv_bags)
  test_predictions <- data.frame(item_id=te$item_id, deal_probability = pred_test_bags)
  
  if(i == 1)
  {
    write.csv(val_predictions,  './submissions/Prav.xgb3.fold1.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/Prav.xgb3.fold1-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 2)
  {
    write.csv(val_predictions,  './submissions/Prav.xgb3.fold2.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/Prav.xgb3.fold2-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 3)
  {
    write.csv(val_predictions,  './submissions/Prav.xgb3.fold3.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/Prav.xgb3.fold3-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 4)
  {
    write.csv(val_predictions,  './submissions/Prav.xgb3.fold4.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/Prav.xgb3.fold4-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 5)
  {
    write.csv(val_predictions,  './submissions/Prav.xgb3.fold5.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/Prav.xgb3.fold5-test.csv', row.names=FALSE, quote = FALSE)
  }
  
}

# Full training

dtrain<-xgb.DMatrix(data = X, label = y)
watchlist <- list( train = dtrain)

fulltrainnrounds = 1.2 * nround.cv


# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

fulltest_ensemble <- rep(0, nrow(te))


for (b in 1:bags) {
  cat(b ," - bag Processing\n")
  seed = seed + b
  cat(seed , " - Random Seed\n ")
  cat("Bagging Full TrainingSet training\n")
  XGModelFulltrain <- xgb.train(    params              = param,
                                    data                = dtrain,
                                    watchlist           = watchlist,
                                    nrounds             = fulltrainnrounds,
                                    print_every_n       = printeveryn,
                                    verbose             = TRUE, 
                                    set.seed            = seed
  )
  
  
  cat("Bagging Full Model prediction Processing\n")
  
  predfull_test         <- predict(XGModelFulltrain, dtest)
  
  fulltest_ensemble <- fulltest_ensemble + predfull_test
  
}

fulltest_ensemble <- fulltest_ensemble / bags
testfull_predictions  <- data.frame(item_id=te$item_id, deal_probability = fulltest_ensemble)
write.csv(testfull_predictions, './submissions/Prav.xgb3.full.csv', row.names=FALSE, quote = FALSE)




# #########################################################################################################
# Bagging - Full train
# #########################################################################################################



# head(testfull_predictions)

############################################################################################
model = xgb.dump(XGModel, with.stats=TRUE)

names = cols
importance_matrix = xgb.importance( names , model = XGModel)
#xgb.plot.importance(importance_matrix)
impMatrix <- as.data.frame(importance_matrix)
write.csv(impMatrix, './Modellogs/Prav.xgb3.impMatrix.csv', row.names=FALSE, quote = FALSE)
# impMatrix[1:12,1]
# impMatrix[,1]
ImpFeature <- impMatrix[,1]

###############################################################################################################################
# 
# i = 1
# cat("X_build fold Processing\n")
# X_build <- subset(tr, CVindices != i, select = c( rowIndex))
# cat("X_val fold Processing\n")
# X_val   <- subset(tr, CVindices == i, select = c( rowIndex)) 
# #tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c()
# 
# tri <- X_build$rowIndex %>% c()
# 
# dtrain <- xgb.DMatrix(data = X[tri, ], label = y[tri])
# dval <- xgb.DMatrix(data = X[-tri, ], label = y[-tri])
# watchlist <- list( val = dval,train = dtrain)
# 
# cols <- colnames(X)
# 
# #rm(X, y, tri); gc()
# 
# dim(dtrain)
# dim(dval)
# dim(dtest)
# #---------------------------
# cat("Training model...\n")
# p <- list(objective = "reg:logistic",
#           booster = "gbtree",
#           eval_metric = "rmse",
#           nthread = 30,
#           eta = 0.05,
#           max_depth = 18,
#           min_child_weight = 9,
#           gamma = 0,
#           subsample = 0.75,
#           colsample_bytree = 0.7,
#           alpha = 1.95,
#           lambda = 0,
#           nrounds = 5000)
# 
# m_xgb <- xgb.train(p, dtrain, p$nrounds, watchlist  = watchlist, print_every_n = 50, early_stopping_rounds = 50)
# 
# xgb.importance(cols, model = m_xgb) %>%   
#   xgb.plot.importance(top_n = 25)
# 
# #---------------------------
# cat("Creating submission file...\n")
# read_csv("../input/sample_submission.csv") %>%  
#   mutate(deal_probability = predict(m_xgb, dtest)) %>%
#   write_csv(paste0("xgb_tfidf", round(m_xgb$best_score, 5), ".csv"))
###########################################################################################################
#
# Kaggle Instacart competition
# Fabien Vavrand, June 2017
# Simple xgboost starter, score 0.3791 on LB
# Products selection is based on product by product binary classification, with a global threshold (0.21)
#
###########################################################################################################

library(data.table)
library(dplyr)
library(tidyr)


# Load Data ---------------------------------------------------------------
path <- "../input"

aisles      <- fread("./input/aisles.csv")
departments <- fread("./input/departments.csv")
orderp      <- fread("./input/order_products__prior.csv")
ordert      <- fread("./input/order_products__train.csv")
orders      <- fread("./input/orders.csv")
products    <- read_csv("./input/products.csv")

trainingFolds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")

# Reshape data ------------------------------------------------------------

head(aisles)
aisles$aisle <- as.factor(aisles$aisle)

head(departments)
departments$department <- as.factor(departments$department)

head(orders)
orders$eval_set <- as.factor(orders$eval_set)

head(products)
products$product_name <- as.factor(products$product_name)

products <- products %>% 
  inner_join(aisles) %>% inner_join(departments) %>% 
  select(-aisle_id, -department_id)
rm(aisles, departments)

head(ordert)
ordert$user_id <- orders$user_id[match(ordert$order_id, orders$order_id)]

head(orderp)
orders_products <- orders %>% inner_join(orderp, by = "order_id")

rm(orderp)
gc()


# Products ----------------------------------------------------------------
prd <- orders_products %>%
  arrange(user_id, order_number, product_id) %>%
  group_by(user_id, product_id) %>%
  mutate(product_time = row_number()) %>%
  ungroup() %>%
  group_by(product_id) %>%
  summarise(
    prod_orders = n(),
    prod_reorders = sum(reordered),
    prod_first_orders = sum(product_time == 1),
    prod_second_orders = sum(product_time == 2)
  )

prd$prod_reorder_probability <- prd$prod_second_orders / prd$prod_first_orders
prd$prod_reorder_times <- 1 + prd$prod_reorders / prd$prod_first_orders
prd$prod_reorder_ratio <- prd$prod_reorders / prd$prod_orders

prd <- prd %>% select(-prod_reorders, -prod_first_orders, -prod_second_orders)

rm(products)
gc()

# Users -------------------------------------------------------------------
users <- orders %>%
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  summarise(
    user_orders = max(order_number),
    user_period = sum(days_since_prior_order, na.rm = T),
    user_mean_days_since_prior = mean(days_since_prior_order, na.rm = T)
  )

us <- orders_products %>%
  group_by(user_id) %>%
  summarise(
    user_total_products = n(),
    user_reorder_ratio = sum(reordered == 1) / sum(order_number > 1),
    user_distinct_products = n_distinct(product_id)
  )

users <- users %>% inner_join(us)
users$user_average_basket <- users$user_total_products / users$user_orders

us <- orders %>%
  filter(eval_set != "prior") %>%
  select(user_id, order_id, eval_set,
         time_since_last_order = days_since_prior_order)

users <- users %>% inner_join(us)

rm(us)
gc()


# Database ----------------------------------------------------------------
data <- orders_products %>%
  group_by(user_id, product_id) %>% 
  summarise(
    up_orders = n(),
    up_first_order = min(order_number),
    up_last_order = max(order_number),
    up_average_cart_position = mean(add_to_cart_order))

rm(orders_products, orders)

data <- data %>% 
  inner_join(prd, by = "product_id") %>%
  inner_join(users, by = "user_id")

data$up_order_rate <- data$up_orders / data$user_orders
data$up_orders_since_last_order <- data$user_orders - data$up_last_order
data$up_order_rate_since_first_order <- data$up_orders / (data$user_orders - data$up_first_order + 1)

data <- data %>% 
  left_join(ordert %>% select(user_id, product_id, reordered), 
            by = c("user_id", "product_id"))

rm(ordert, prd, users)
gc()


# Train / Test datasets ---------------------------------------------------
train <- as.data.frame(data[data$eval_set == "train",])
# train$eval_set <- NULL
# train$user_id <- NULL
# train$product_id <- NULL
# train$order_id <- NULL
train$reordered[is.na(train$reordered)] <- 0

test <- as.data.frame(data[data$eval_set == "test",])
# test$eval_set <- NULL
# test$user_id <- NULL
# test$reordered <- NULL

rm(data)
gc()

head(train)
head(test)

trainingSet <- train
testingSet  <- test 

# Join CV folds script here

trainingSet <- left_join(trainingSet, trainingFolds, by="user_id")


feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c("user_id","order_id","product_id","eval_set", "reordered", "CVindices" ))])

# Model -------------------------------------------------------------------


cv          = 5
bags        = 1
nround.cv   = 500 
printeveryn = 100
seed        = 2016

param <- list(
              "objective"           = "reg:logistic",
              "booster"             = "gbtree",
              "eval_metric"         = "logloss",
              "tree_method"         = "exact",
              "nthread"             = 28,  
              "max_depth"           = 6,
              "eta"                 = 0.02,
              "min_child_weight"    = 10,
              "gamma"               = 0.7,
              "subsample"           = 0.7,
              "colsample_bytree"    = 0.95,
              "alpha"               = 2e-05,
              "lambda"              = 10
            )

cat(cv, "-fold Cross Validation\n", sep = "")

set.seed(seed)
for (i in 1:cv)
  
{
  cat(i ,"fold Processing\n")
  
  cat("X_build fold Processing\n")
  X_build <- subset(trainingSet, CVindices != i, select = -c( CVindices))
  cat("X_val fold Processing\n")
  X_val   <- subset(trainingSet, CVindices == i, select = -c( CVindices)) 
  
  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=X_build$reordered)
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=X_val$reordered)
  watchlist <- list( val = dval,train = dtrain)
  
  cat("X_build training Processing\n")
  XGModel <- xgb.train(   params              = param,
                          data                = dtrain,
                          watchlist           = watchlist,
                          nrounds             = nround.cv ,
                          print_every_n       = printeveryn,
                          verbose             = TRUE, 
                          #maximize           = TRUE,
                          set.seed            = seed
  )
  
    cat("X_val prediction Processing\n")
    pred_cv  <- predict(XGModel, data.matrix(X_val[,feature.names]))
    val_predictions <- data.frame(user_id=X_val$user_id,order_id=X_val$order_id,product_id=X_val$product_id, pred = pred_cv)
    
    dt <- data.frame(user_id=X_val$user_id, purch=X_val$reordered, pred=pred_cv)
     f1score <- dt %>%
                  group_by(user_id) %>%
                  summarise(f1score=f1Score(purch, pred, cutoff=0.22))
     
     cat("fold " , i  , " F1 score - including NA : " , mean(f1score$f1score, na.rm = TRUE), "\n", sep = "")
     f1score[is.na(f1score)] <- 0
     cat("fold " , i  , " F1 score - NA replace with 0 : " , mean(f1score$f1score), "\n", sep = "")
    

    cat("CV TestingSet prediction Processing\n")
    pred_test  <- predict(XGModel, data.matrix(testingSet[,feature.names]))
    test_predictions <- data.frame(user_id=testingSet$user_id,order_id=testingSet$order_id,product_id=testingSet$product_id, pred = pred_test)

    
    if(i == 1)
    {
      write.csv(val_predictions,  './submissions/prav.xgb01.fold1.csv', row.names=FALSE, quote = FALSE)
      write.csv(test_predictions, './submissions/prav.xgb01.fold1-test.csv', row.names=FALSE, quote = FALSE)
    }
    if(i == 2)
    {
      write.csv(val_predictions,  './submissions/prav.xgb01.fold2.csv', row.names=FALSE, quote = FALSE)
      write.csv(test_predictions, './submissions/prav.xgb01.fold2-test.csv', row.names=FALSE, quote = FALSE)
    }
    if(i == 3)
    {
      write.csv(val_predictions,  './submissions/prav.xgb01.fold3.csv', row.names=FALSE, quote = FALSE)
      write.csv(test_predictions, './submissions/prav.xgb01.fold3-test.csv', row.names=FALSE, quote = FALSE)
    }
    if(i == 4)
    {
      write.csv(val_predictions,  './submissions/prav.xgb01.fold4.csv', row.names=FALSE, quote = FALSE)
      write.csv(test_predictions, './submissions/prav.xgb01.fold4-test.csv', row.names=FALSE, quote = FALSE)
    }
    if(i == 5)
    {
      write.csv(val_predictions,  './submissions/prav.xgb01.fold5.csv', row.names=FALSE, quote = FALSE)
      write.csv(test_predictions, './submissions/prav.xgb01.fold5-test.csv', row.names=FALSE, quote = FALSE)
    }
  
}



dtrain<-xgb.DMatrix(data=data.matrix(trainingSet[, feature.names]),label=trainingSet$reordered)
watchlist <- list( train = dtrain)

fulltrainnrounds = as.integer(1.2 * nround.cv)


# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

fulltest_ensemble <- rep(0, nrow(testingSet[,feature.names]))

for (b in 1:bags) {
  # seed = seed + b
  # set.seed(seed)
  cat(b ," - bag Processing\n")
  cat("Bagging Full TrainingSet training\n")
  XGModelFulltrain <- xgb.train(    params              = param,
                                    data                = dtrain,
                                    watchlist           = watchlist,
                                    nrounds             = fulltrainnrounds,
                                    print_every_n       = printeveryn,
                                    verbose             = TRUE, 
                                    #maximize            = TRUE,
                                    set.seed            = seed
  )
  cat("Bagging Full Model prediction Processing\n")
  
  predfull_test         <- predict(XGModelFulltrain, data.matrix(testingSet[,feature.names]))
  fulltest_ensemble     <- fulltest_ensemble + predfull_test
}

fulltest_ensemble <- fulltest_ensemble / bags
testfull_predictions <- data.frame(user_id=testingSet$user_id,order_id=testingSet$order_id,product_id=testingSet$product_id, pred = fulltest_ensemble)

write.csv(testfull_predictions, './submissions/prav.xgb01.full.csv', row.names=FALSE, quote = FALSE)

# #########################################################################################################
# Bagging - Full train
# #########################################################################################################






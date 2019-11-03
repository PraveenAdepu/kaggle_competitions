

train <- fread('./input/train.csv')
test <-fread('./input/test.csv')

head(train)
head(test)


train$date <- as.Date(parse_date_time(train$date,'%y-%m-%d'))
test$date  <- as.Date(parse_date_time(test$date,'%y-%m-%d'))

train_sub <- train[date >= as.Date("2017-04-01"), ]
train_sub <- as.data.frame(train_sub)

rm(train);gc()

max(train_sub$date)
train_dates <- seq(as.Date('2017-04-01'),as.Date('2017-08-15'),by = 1)
train_dates <- as.data.frame(train_dates)
names(train_dates) <- "date"

train_store_items <- train_sub %>%
                        distinct(store_nbr ,item_nbr)

train_store_items_dates <- merge(train_store_items,train_dates,all=TRUE)

head(train_store_items_dates)
head(train_sub)
train_sub$id <- NULL

trainingSet <- left_join(train_store_items_dates, train_sub , by =c("store_nbr","item_nbr","date"))

unique(trainingSet$onpromotion)

trainingSet$onpromotion[is.na(trainingSet$onpromotion)] <- FALSE
trainingSet$unit_sales[is.na(trainingSet$unit_sales)] <- 0
trainingSet$unit_sales <- as.numeric(trainingSet$unit_sales)
trainingSet$unit_sales[trainingSet$unit_sales < 0] <- 0



trainingSet$onpromotion <- as.integer(as.factor(trainingSet$onpromotion))
test$onpromotion <- as.integer(as.factor(test$onpromotion))

stores <- read_csv("./input/stores.csv")

head(stores)

stores$city  <- as.integer(as.factor(stores$city))
stores$state <- as.integer(as.factor(stores$state))
stores$type  <- as.integer(as.factor(stores$type))


oil <- read_csv("./input/oil.csv")

head(oil)


items <- read_csv("./input/items.csv")

head(items)

items$family   <- as.integer(as.factor(items$family))
holiday_events <- read_csv("./input/holidays_events.csv")

head(holiday_events)

holiday_events$type        <- as.integer(as.factor(holiday_events$type))
holiday_events$locale      <- as.integer(as.factor(holiday_events$locale))
holiday_events$locale_name <- as.integer(as.factor(holiday_events$locale_name))
holiday_events$description <- NULL
holiday_events$transferred <- as.integer(as.factor(holiday_events$transferred))

trainingSet <- as.data.frame(trainingSet)


trainingSet <- left_join(trainingSet, stores, by="store_nbr")
trainingSet <- left_join(trainingSet, oil, by = "date" )
trainingSet <- left_join(trainingSet, items, by="item_nbr")
trainingSet <- left_join(trainingSet, holiday_events, by = "date")

testingSet <- left_join(test, stores, by="store_nbr")
testingSet <- left_join(testingSet, oil, by = "date" )
testingSet <- left_join(testingSet, items, by="item_nbr")
testingSet <- left_join(testingSet, holiday_events, by = "date")

testingSet$month <- month(testingSet$date)
testingSet$day   <- day(testingSet$date)
testingSet$wday <- as.integer(as.factor(weekdays(testingSet$date)))

trainingSet$month <- month(trainingSet$date)
trainingSet$day   <- day(trainingSet$date)
trainingSet$wday <- as.integer(as.factor(weekdays(trainingSet$date)))

unique(trainingSet$wday)

rm(test,train_sub, stores, oil, items, holiday_events, train_dates, train_store_items, train_store_items_dates); gc()




trainingSet <- as.data.table(trainingSet)


X_build <- trainingSet[date < "2017-08-01"]
X_val   <- trainingSet[date >= "2017-08-01"]

X_build <- as.data.frame(X_build)
X_val   <- as.data.frame(X_val)

rm(trainingSet); gc()



######################################################################################################################
# X_build <- trainingSet %>% filter(date < "2017-08-01")
# X_val   <- trainingSet %>% filter(date >= "2017-08-01")


names(X_build)

feature.names     <- names(X_build[,-which(names(X_build) %in% c("id","date", "store_item_nbr","unit_sales","locale_name","wday" ))])

###########################################################################################################################
# Metric function forecast validation #####################################################################################
###########################################################################################################################
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

metric = "rmse"

cv          = 5
bags        = 1
nround.cv   = 600 
printeveryn = 100
seed        = 2017

## for all remaining models, use same parameters 

param <- list(  "objective"        = "reg:linear", 
                "booster"          = "gbtree",
                "tree_method"      = "exact",
                "eval_metric"      = "rmse",
                "nthread"          = 20,     
                "max_depth"        = 7,     
                "eta"              = 0.02, 
                "subsample"        = 0.7,  
                "colsample_bytree" = 0.7,  
                "min_child_weight" = 3     
                
)

X_build[is.na(X_build)] <- 0
X_val[is.na(X_val)] <- 0

testingSet[is.na(testingSet)] <- 0

dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=log(1+X_build$unit_sales))
dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=log(1+X_val$unit_sales))
watchlist <- list( val = dval,train = dtrain)


pred_cv_bags   <- rep(0, nrow(X_val[, feature.names]))
pred_test_bags <- rep(0, nrow(testingSet[,feature.names]))

for (b in 1:bags) 
{
  cat(b ," - bag Processing\n")
  seed = b + seed
  set.seed(seed)
  cat("X_build training Processing\n")
  XGModel <- xgb.train(   params              = param,
                          #feval               = xgb.metric.log.mae, #xgb.metric.mae
                          data                = dtrain,
                          watchlist           = watchlist,
                          nrounds             = nround.cv ,
                          print_every_n       = printeveryn,
                          verbose             = TRUE,
                          #maximize           = TRUE,
                          set.seed            = seed
  )
  cat("X_val prediction Processing\n")
  pred_cv    <- predict(XGModel, data.matrix(X_val[,feature.names]))
  #cat("CV TestingSet prediction Processing\n")
  pred_test  <- predict(XGModel, data.matrix(testingSet[,feature.names]))
  
  pred_cv_bags   <- pred_cv_bags + (exp(pred_cv)-1)
  pred_test_bags <- pred_test_bags + (exp(pred_test)-1)
}
pred_cv_bags   <- pred_cv_bags / bags
pred_test_bags <- pred_test_bags / bags


testfull_predictions  <- data.frame(id=testingSet$id, unit_sales = pred_test_bags)
write.csv(testfull_predictions, './submissions/prav.xgb02.full.csv', row.names=FALSE, quote = FALSE)

# i = 1
# cat("CV Fold-", i, " ", metric, ": ", score(log(X_val$unit_sales+1), log(pred_cv_bags+1), metric), "\n", sep = "")


sum(X_val$unit_sales)
# [1] 12433958
sum(pred_cv_bags)
# [1] 7311170
# > 

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



mean_sub <- read_csv("./submissions/Prav_arima01.csv")
#test_predictions <- read_csv('./submissions/submission_v0.csv')
test_predictions1 <- read_csv('./submissions/Prav_nnetar01.csv')#ets01 0.558


test_predictions1$unit_sales[test_predictions1$unit_sales < 0] <- 0


all_ensemble <- left_join(test_predictions1, mean_sub, by = "id")


head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"id")
cor(all_ensemble[, ensemble.features])

sum(all_ensemble$unit_sales.x)
sum(all_ensemble$unit_sales.y)

all_ensemble$unit_sales <- all_ensemble$unit_sales.x * 0.5 + all_ensemble$unit_sales.y * 0.5

cols <- c("id","unit_sales")

write.csv(all_ensemble[,cols],"./submissions/Prav_arima01_ets01_mean.csv", row.names = FALSE, quote = FALSE)


# 
# train <- fread('./input/train.csv')
# test <-fread('./input/test.csv')
# 
# head(train)
# head(test)
# 
# 
# train$date <- as.Date(parse_date_time(train$date,'%y-%m-%d'))
# test$date  <- as.Date(parse_date_time(test$date,'%y-%m-%d'))
# 
# 
# train$store_item_nbr <- paste(train$store_nbr, train$item_nbr, sep="_")
# test$store_item_nbr <- paste(test$store_nbr, test$item_nbr, sep="_")
# 
# tail(train)
# # train_sub -  use 2017 data to start with the forecast
# train_sub <- train[date >= as.Date("2017-04-01"), ]
# train_sub <- as.data.frame(train_sub)
# #train_sub <- train_sub[, c('date','store_nbr','item_nbr','store_item_nbr', 'unit_sales')]
# 
# head(train_sub)
# 
# #write.csv(train_sub,"./input/train_sub.csv", row.names = FALSE, quote = FALSE)
# rm(train); gc()
train_sub <- fread('./input/train_sub.csv')
train_sub$date <- as.Date(parse_date_time(train_sub$date,'%y-%m-%d'))
train_sub <- as.data.table(train_sub)
# transform to log1p
train_sub$unit_sales <- as.numeric(train_sub$unit_sales)
train_sub$unit_sales[train_sub$unit_sales < 0] <- 0
#train_sub$unit_sales <- log1p(train_sub$unit_sales)

head(train_sub)

train_sub$onpromotion <- as.integer(as.factor(train_sub$onpromotion))

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

train_sub <- as.data.frame(train_sub)
names(train_sub); names(stores)

trainingSet <- left_join(train_sub, stores, by="store_nbr")

trainingSet <- left_join(trainingSet, oil, by = "date" )

trainingSet <- left_join(trainingSet, items, by="item_nbr")

trainingSet <- left_join(trainingSet, holiday_events, by = "date")

trainingSet$month <- month(trainingSet$date)
trainingSet$day   <- day(trainingSet$date)

head(trainingSet)

rm(test, train_sub, stores, oil, items, holiday_events); gc()

min(trainingSet$date)
max(trainingSet$date)


trainingSet <- as.data.table(trainingSet)

X_build <- trainingSet[date < "2017-08-01"]
X_val   <- trainingSet[date >= "2017-08-01"]

X_build <- as.data.frame(X_build)
X_val   <- as.data.frame(X_val)

rm(trainingSet); gc()
#####################################################################################################################
names(X_build)

colClasses = c("integer","integer", "Date","double","double","double")
col.names = c("store_nbr","item_nbr", "date","unit_sales","unit_sales_lower","unit_sales_upper")

store_item.forecast.periods  <- read.table(text = "",
                                          colClasses = colClasses,
                                          col.names = col.names)


#########################################################################
forecast.periods = 13
forecast.result.columns <- c("store_nbr","item_nbr","ds","yhat","yhat_lower","yhat_upper")

for (StoreItem in unique(X_build$store_nbr)){
  cat("Processing store_nbr : ", StoreItem,"\n")
  #StoreItem = 1
  Store.planogram.source <- X_build %>%
                              filter(store_nbr == StoreItem)
  for(UPCItem in unique(X_build$item_nbr)){
    cat("Processing item_nbr : ", UPCItem, "\n")
    #UPCItem = 103520
    #UPCItem = "APPLE IPHONE 7 32GB"
    Store.UPC.planogram.source <- Store.planogram.source %>%
                                    filter(item_nbr == UPCItem) %>%
                                    dplyr::select(date,unit_sales) %>%
                                    arrange(date) %>%
                                    dplyr::rename(ds = date, y = unit_sales)
    
  
    prophet.model <- prophet(Store.UPC.planogram.source, daily.seasonality = TRUE)
    future <- make_future_dataframe(prophet.model, periods = forecast.periods+10)
    forecast <- predict(prophet.model, future)

    forecast$ds     <- as.Date(forecast$ds)
    forecast$store_nbr  <- StoreItem
    forecast$item_nbr   <- UPCItem
    
    predictions     <- tail(forecast[,forecast.result.columns],forecast.periods)
    predictions     <- predictions %>%
                          dplyr::rename(date = ds, unit_sales = yhat, unit_sales_lower = yhat_lower, unit_sales_upper = yhat_upper)
    
    planogram.forecast.periods <- rbind(store_item.forecast.periods,predictions)
    
    
    
  }
  
}



planogram.forecast.periods

planogram.forecast.periods.rank <- planogram.forecast.periods %>%
  group_by(Store,YearMonthDate) %>%
  mutate(  StoreUPCPeriodRank = rank(desc(SalesForecast), ties.method = "first")
           , StoreUPCPeriodRank_lower = rank(desc(SalesForecast_lower), ties.method = "first")
           , StoreUPCPeriodRank_upper = rank(desc(SalesForecast_upper), ties.method = "first")
  ) %>%
  ungroup() %>%
  group_by(Store, UPC) %>%
  mutate(YearMonthDateAdjustmentMonth = dense_rank(YearMonthDate)
  )
# Prav - Don't require this, prophet freq option handles this automatically
# month(planogram.forecast.periods.rank$YearMonthDate) <- month(planogram.forecast.periods.rank$YearMonthDate) + planogram.forecast.periods.rank$YearMonthDateAdjustmentMonth
# day(planogram.forecast.periods.rank$YearMonthDate)   <- days_in_month(planogram.forecast.periods.rank$YearMonthDate)

planogram.forecast.periods.rank$YearMonthDateAdjustmentMonth <- NULL


planogram.forecast.periods.rank <- planogram.forecast.periods.rank %>%
  arrange(Store, YearMonthDate,StoreUPCPeriodRank,StoreUPCPeriodRank_lower,StoreUPCPeriodRank_upper)

validation.set$SalesForecast_lower <- 0
validation.set$SalesForecast_upper <- 0

source.details <- validation.set %>% 
  distinct(Store,UPC,YearMonthDate,SalesCount,SalesForecast_lower,SalesForecast_upper) 

source.details$StoreUPCPeriodRank <- 1
source.details$StoreUPCPeriodRank_lower <- 1
source.details$StoreUPCPeriodRank_upper <- 1
source.details$Source <- "Validation"

planogram.forecast.periods.rank$Source <- "Prediction"

planogram.forecast.periods.rank <- planogram.forecast.periods.rank %>% 
  mutate(SalesCount = ifelse(SalesForecast <= 0 ,0, SalesForecast)
         ,SalesForecast_lower = ifelse(SalesForecast_lower <= 0 ,0, SalesForecast_lower)
         ,SalesForecast_upper = ifelse(SalesForecast_upper <= 0 ,0, SalesForecast_upper)
  ) %>% ungroup()

planogram.forecast.periods.rank$SalesForecast <- NULL                                             

planogram.forecast.periods.rank$SalesCount <- ifelse(planogram.forecast.periods.rank$SalesCount<=0,0,planogram.forecast.periods.rank$SalesCount)

planogram.validation <- rbind(planogram.forecast.periods.rank,source.details)

head(planogram.validation)

# planogram.validation$UPC <-  gsub("4GX|32GB|64GB|128GB|256GB|HD", "", planogram.validation$UPC)
# planogram.validation$UPC <- trimws(planogram.validation$UPC)

write.csv(planogram.validation,"C:/Users/PA23309/Documents/Prav-Development/Julian/planogram_validation.csv",row.names = FALSE, quote = FALSE)

##############################################################################################################################
# forecast.periods = 4

# source.data <- list(ds = c('2016-09-01','2016-10-01','2016-11-01','2016-12-01','2017-01-01','2017-02-01','2017-03-01','2017-04-01','2017-05-01','2017-06-01','2017-07-01'),
#                     y=c(29,
#                         48,
#                         64,
#                         60,
#                         62,
#                         56,
#                         58,
#                         42,
#                         58,
#                         37,
#                         51
#                     ))



# 
# dates <- list(seq(as.Date("2016/09/12"), as.Date("2017/08/28"), "week"))
# 
# y <- list(c( 13
#              ,11
#              ,2
#              ,13
#              ,6
#              ,6
#              ,16
#              ,16
#              ,12
#              ,12
#              ,15
#              ,9
#              ,20
#              ,16
#              ,18
#              ,4
#              ,10
#              ,19
#              ,22
#              ,15
#              ,15
#              ,10
#              ,15
#              ,15
#              ,19
#              ,8
#              ,17
#              ,14
#              ,13
#              ,5
#              ,5
#              ,16
#              ,16
#              ,16
#              ,13
#              ,13
#              ,16
#              ,11
#              ,18
#              ,10
#              ,13
#              ,6
#              ,18
#              ,14
#              ,13
#              ,11
#              ,8
#              ,8
#              ,7
#              ,5
#              ,3
#              
#              
#             
# ))
# 
# source.data <- data.frame(dates,y)
# names(source.data) <-c("ds","y")
# sapply(source.data, class)
# 
# #source.data$ds <- as.Date(source.data$ds)
# 
# train.source <- source.data %>%
#                   filter(ds <= '2017-08-01')
# 
# prophet.model <- prophet(train.source, weekly.seasonality = TRUE, yearly.seasonality = FALSE, daily.seasonality = FALSE)
# 
# 
# future <- make_future_dataframe(prophet.model, periods = 4, freq = "week")
# 
# # head(future)
# # tail(future)
# 
# forecast <- predict(prophet.model, future)
# 
# 
# # tail(forecast[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])
# # plot(prophet.model, forecast)
# # prophet_plot_components(prophet.model, forecast)
# forecast.result.columns <- c('ds', 'yhat')
# predictions     <- tail(forecast[,forecast.result.columns],forecast.periods)
# predictions     <- predictions %>%
#                       dplyr::rename( y = yhat)
# 
# planogram.forecast.periods <- rbind(source.data,predictions)
# 
# 
# source.data
# planogram.forecast.periods


planogram.source <- read_csv("./planogram_validation.csv")

planogram.source <- planogram.source %>% filter(YearMonthDate == "2017-08-01")

planogram.source <- dplyr::filter(planogram.source, !grepl('PRE', UPC))
head(planogram.source)

unique(planogram.source$Store)
unique(planogram.source$UPC)

planogram.source$UPC <-  gsub("4GX|32GB|64GB|128GB|256GB|HD|4G|16GB|-PRE|", "", planogram.source$UPC)
planogram.source$UPC <- trimws(planogram.source$UPC)

planogram.source.info <- planogram.source %>% group_by(Store, UPC, YearMonthDate) %>% summarise(SalesForecast = sum(SalesCount))

planogram.source.info <- planogram.source.info %>%
  group_by(Store,YearMonthDate) %>%
  mutate(  StoreUPCPeriodRank = rank(desc(SalesForecast), ties.method = "first")
  )

planogram.source.info$Store <-  gsub("TELSTRA STORE", "", planogram.source.info$Store)
planogram.source.info$Store <- trimws(planogram.source.info$Store)


planogram.info <- read_csv("./planogram_info.csv")

planogram.info <- planogram.info %>% filter(Subcategory == "Postpaid")


head(planogram.info)

planogram.UPC.count <- planogram.info %>% group_by(`Store Name`) %>% summarise(PlanogramCount = n())
names(planogram.UPC.count)[1] <- "Store"

head(planogram.source.info); head(planogram.UPC.count)

planogram.source.with.similarstores.mean <- left_join(planogram.source.info, planogram.UPC.count, by=c("Store"))

planogram.source.with.similarstores.mean$MaxSlot <- ifelse(planogram.source.with.similarstores.mean$StoreUPCPeriodRank == planogram.source.with.similarstores.mean$PlanogramCount, 1, 0)

head(planogram.source.with.similarstores.mean)


write.csv(planogram.source.with.similarstores.mean, "./planogram_validation_with_similarstores_means.csv", row.names = FALSE, quote = FALSE)











































######################################################################################################################
# X_build <- trainingSet %>% filter(date < "2017-08-01")
# X_val   <- trainingSet %>% filter(date >= "2017-08-01")

rm(trainingSet); gc()

feature.names     <- names(X_build[,-which(names(X_build) %in% c("id","date", "store_item_nbr","unit_sales" ))])

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
nround.cv   = 20 
printeveryn = 2
seed        = 2017

## for all remaining models, use same parameters 

param <- list(  "objective"        = "reg:linear", 
                "booster"          = "gbtree",
                "tree_method"      = "exact",
                "eval_metric"      = "rmse",
                "nthread"          = 7,     
                "max_depth"        = 7,     
                "eta"              = 0.02, 
                "subsample"        = 0.95,  
                "colsample_bytree" = 0.3,  
                "min_child_weight" = 1     
                
)

X_build[is.na(X_build)] <- 0
X_val[is.na(X_val)] <- 0

dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=log(1+X_build$unit_sales))
dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=log(1+X_val$unit_sales))
watchlist <- list( val = dval,train = dtrain)


pred_cv_bags   <- rep(0, nrow(X_val[, feature.names]))
#pred_test_bags <- rep(0, nrow(testingSet[,testfeature.names]))

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
                          #maximize            = TRUE,
                          set.seed            = seed
  )
  cat("X_val prediction Processing\n")
  pred_cv    <- predict(XGModel, data.matrix(X_val[,feature.names]))
  #cat("CV TestingSet prediction Processing\n")
  #pred_test  <- predict(XGModel, data.matrix(testingSet[,testfeature.names]))
  
  pred_cv_bags   <- pred_cv_bags + (exp(pred_cv)-1)
  #pred_test_bags <- pred_test_bags + exp(pred_test)
}
pred_cv_bags   <- pred_cv_bags / bags
#pred_test_bags <- pred_test_bags / bags
X_val$predictions <- pred_cv_bags
i = 1
cat("CV Fold-", i, " ", metric, ": ", score(X_val$loss, pred_cv_bags, metric), "\n", sep = "")


# dcast the data from long to wide format for time series forecasting
train_sub_wide <- dcast(train_sub, store_item_nbr ~ date, value.var = "unit_sales", fill = 0)
head(train_sub_wide)
train_ts <- ts(train_sub_wide, frequency = 7) # considering one week as a short shopping cycle


fcst_intv = 16  # 16 days of forecast interval (Aug 16 ~ 31) per the submission requirement
fcst_matrix <- matrix(NA,nrow=nrow(train_ts),ncol=fcst_intv)

# register 15 cores for parallel processing in ETS forecasting
i = 1
registerDoMC(detectCores()-1)
fcst_matrix <- foreach(i=1:nrow(train_ts),.combine=rbind, .packages=c("forecast")) %dopar% { 
  fcst_matrix <- forecast(ets(train_ts[i,]),h=fcst_intv)$mean
}




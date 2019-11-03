


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
# train_sub <- train[date >= as.Date("2017-04-01"), ]
# train_sub <- as.data.frame(train_sub)
# 
# rm(train);gc()
# 
# max(train_sub$date)
# train_dates <- seq(as.Date('2017-04-01'),as.Date('2017-08-15'),by = 1)
# train_dates <- as.data.frame(train_dates)
# names(train_dates) <- "date"
# 
# train_store_items <- train_sub %>%
#   distinct(store_nbr ,item_nbr)
# 
# train_store_items_dates <- merge(train_store_items,train_dates,all=TRUE)
# 
# head(train_store_items_dates)
# head(train_sub)
# train_sub$id <- NULL
# 
# trainingSet <- left_join(train_store_items_dates, train_sub , by =c("store_nbr","item_nbr","date"))
# 
# is.data.table(trainingSet)
# 
# trainingSet <- as.data.table(trainingSet)
# 
# X_build <- trainingSet[date < "2017-08-01"]
# X_val   <- trainingSet[date >= "2017-08-01"]
# 
# X_build <- as.data.frame(X_build)
# X_val   <- as.data.frame(X_val)
# 
# trainingSet <- as.data.frame(trainingSet)
# 
# X_build$unit_sales[is.na(X_build$unit_sales)] <- 0
# X_val$unit_sales[is.na(X_val$unit_sales)] <- 0
# 
# trainingSet$unit_sales[is.na(trainingSet$unit_sales)] <- 0
# 
# rm(trainingSet); gc()
# 
# saveRDS(trainingSet, "./input/trainingSet.rds")
# saveRDS(X_build, "./input/X_build.rds")
# saveRDS(X_val, "./input/X_val.rds")

trainingSet <- readRDS("./input/trainingSet.rds")
# X_build <- readRDS("./input/X_build.rds")
# X_val   <- readRDS("./input/X_val.rds")
#sort(unique(trainingSet$store_nbr))
#########################################################################
colClasses = c("integer","integer", "Date","double","double","double")
col.names = c("store_nbr","item_nbr", "date","unit_sales","unit_sales_lower","unit_sales_upper")

store_item.forecast.periods  <- read.table(text = "",
                                           colClasses = colClasses,
                                           col.names = col.names)


#########################################################################
forecast.periods = 16
forecast.result.columns <- c("store_nbr","item_nbr","ds","yhat","yhat_lower","yhat_upper")

for (StoreItem in seq(36,40, by=1)){
  cat("Processing store_nbr : ", StoreItem,"\n")
  #StoreItem = 1
  Store.planogram.source <- trainingSet %>%
    filter(store_nbr == StoreItem)
  for(UPCItem in unique(Store.planogram.source$item_nbr)){
    cat("Processing item_nbr : ", UPCItem, "\n")
    #UPCItem = 265266 103520
    #UPCItem = "APPLE IPHONE 7 32GB"
    Store.UPC.planogram.source <- Store.planogram.source %>%
      filter(item_nbr == UPCItem) %>%
      dplyr::select(date,unit_sales) %>%
      arrange(date) %>%
      dplyr::rename(ds = date, y = unit_sales)
    
    
    prophet.model <- prophet(Store.UPC.planogram.source, daily.seasonality = TRUE, weekly.seasonality = TRUE, yearly.seasonality = FALSE)
    future <- make_future_dataframe(prophet.model, periods = forecast.periods)
    forecast <- predict(prophet.model, future)
    
    forecast$ds     <- as.Date(forecast$ds)
    forecast$store_nbr  <- StoreItem
    forecast$item_nbr   <- UPCItem
    
    predictions     <- tail(forecast[,forecast.result.columns],forecast.periods)
    predictions     <- predictions %>%
      dplyr::rename(date = ds, unit_sales = yhat, unit_sales_lower = yhat_lower, unit_sales_upper = yhat_upper)
    
    store_item.forecast.periods <- rbind(store_item.forecast.periods,predictions)
    
    
    
  }
  
}


write.csv(store_item.forecast.periods,"./submissions/stores36-40.csv", row.names = FALSE, quote = FALSE)

planogram.forecast.periods
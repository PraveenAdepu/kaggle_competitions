
train <- fread('./input/train.csv')
test <-fread('./input/test.csv')

head(train)
head(test)

test %>% filter(store_item_nbr == "1_96995")


train$date <- as.Date(parse_date_time(train$date,'%y-%m-%d'))
test$date  <- as.Date(parse_date_time(test$date,'%y-%m-%d'))


train$store_item_nbr <- paste(train$store_nbr, train$item_nbr, sep="_")
test$store_item_nbr <- paste(test$store_nbr, test$item_nbr, sep="_")

tail(train)
# train_sub -  use 2017 data to start with the forecast
train_sub <- train[date >= as.Date("2017-04-01"), ]
train_sub <- as.data.frame(train_sub)
train_sub <- train_sub[, c('date','store_item_nbr', 'unit_sales')]

head(train_sub)

#write.csv(train_sub,"./input/train_sub.csv", row.names = FALSE, quote = FALSE)

train_sub <- as.data.table(train_sub)
# transform to log1p
train_sub$unit_sales <- as.numeric(train_sub$unit_sales)
train_sub$unit_sales[train_sub$unit_sales < 0] <- 0
train_sub$unit_sales <- log1p(train_sub$unit_sales)

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




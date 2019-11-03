


train <- fread('./input/train.csv')
test <-fread('./input/test.csv')


train$date <- as.Date(parse_date_time(train$date,'%y-%m-%d'))
test$date <- as.Date(parse_date_time(test$date,'%y-%m-%d'))
train$store_item_nbr <- paste(train$store_nbr, train$item_nbr, sep="_")
test$store_item_nbr <- paste(test$store_nbr, test$item_nbr, sep="_")

head(train)
# train_sub -  use 2017 data to start with the forecast
train_sub <- train[date >= as.Date("2016-01-01"), ]

#train_sub <- as.data.frame(train_sub)
cols <- c('date','store_item_nbr', 'unit_sales')
train_sub <- train_sub[, cols,with=FALSE]

# clean up
rm(train)

head(train_sub)

train_valid <- train_sub[date >= as.Date("2017-08-01"), ]
train_sub   <- train_sub[date < as.Date("2017-08-01"), ]

summary(train_valid$unit_sales)
train_valid$unit_sales[train_valid$unit_sales < 0] <- 0
min(train_valid$date)
max(train_valid$date)
# transform to log1p
train_sub$unit_sales <- as.numeric(train_sub$unit_sales)
train_sub$unit_sales[train_sub$unit_sales < 0] <- 0
train_sub$unit_sales <- log1p(train_sub$unit_sales)

train_sub <- as.data.frame(train_sub)
train_valid <- as.data.frame(train_valid)

# dcast the data from long to wide format for time series forecasting
train_sub_wide <- dcast(train_sub, store_item_nbr ~ date, value.var = "unit_sales", fill = 0)
train_ts <- ts(train_sub_wide, frequency = 7) # considering one week as a short shopping cycle

fcst_intv = 15  # 16 days of forecast interval (Aug 16 ~ 31) per the submission requirement
fcst_matrix <- matrix(NA,nrow=nrow(train_ts),ncol=fcst_intv)

# register 15 cores for parallel processing in ETS forecasting

# cl <- makeCluster(28)
# registerDoParallel(cl)
# 
# fcst_matrix <- foreach(i=1:nrow(train_ts),.combine=rbind, .packages=c("forecast")) %dopar% { 
#   fcst_matrix <- expm1(forecast(ets(train_ts[i,]),h=fcst_intv)$mean)
# }

cl <- makeCluster(28)
registerDoParallel(cl)
fcst_matrix <- foreach(i=1:nrow(train_ts),.combine=rbind, .packages=c("forecastHybrid")) %dopar% {
  fcst_matrix <- expm1(forecast(hybridModel(train_ts[i,-1],models = "ae", weights="insample"
                                      , errorMethod = c("RMSE"),num.cores = 20, verbose = TRUE),h=fcst_intv)$mean)
}
stopCluster(cl)


# post-processing the forecast table
fcst_matrix[fcst_matrix < 0] <- 0
colnames(fcst_matrix) <- as.character(seq(from = as.Date("2017-08-01"), 
                                          to = as.Date("2017-08-15"), 
                                          by = 'day'))
fcst_df <- as.data.frame(cbind(train_sub_wide[, 1], fcst_matrix)) 
colnames(fcst_df)[1] <- "store_item_nbr"


# melt the forecast data frame from wide to long format for final submission
fcst_df_long <- melt(fcst_df, id = 'store_item_nbr', 
                     variable.name = "fcst_date", 
                     value.name = 'unit_sales')
fcst_df_long$store_item_nbr <- as.character(fcst_df_long$store_item_nbr)
fcst_df_long$fcst_date <- as.Date(parse_date_time(fcst_df_long$fcst_date,'%y-%m-%d'))
fcst_df_long$unit_sales <- as.numeric(fcst_df_long$unit_sales)

head(fcst_df_long)
# generate the final submission file
submission <- left_join(train_valid, fcst_df_long, 
                        c("store_item_nbr" = "store_item_nbr", 'date' = 'fcst_date'))
head(submission)

submission$unit_sales.x[submission$unit_sales.x < 0] <- 0
submission$unit_sales.y[is.na(submission$unit_sales.y)] <- 0

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
i  = 1
cat("CV Fold-", i, " ", metric, ": ", score(log1p(submission$unit_sales.x), log1p(submission$unit_sales.y), metric), "\n", sep = "")

























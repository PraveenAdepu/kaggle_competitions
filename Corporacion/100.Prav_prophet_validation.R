
library(dplyr)
library(forecast)
library(reshape2)
library(data.table)
library(foreach)
library(date)
library(lubridate)
library(doParallel)
library(prophet)


train <- fread('./input/train.csv')
test <-fread('./input/test.csv')


train$date <- as.Date(parse_date_time(train$date,'%y-%m-%d'))
test$date <- as.Date(parse_date_time(test$date,'%y-%m-%d'))
train$store_item_nbr <- paste(train$store_nbr, train$item_nbr, sep="_")
test$store_item_nbr <- paste(test$store_nbr, test$item_nbr, sep="_")

head(train)
# train_sub -  use 2017 data to start with the forecast
train_sub <- train[date >= as.Date("2017-01-01"), ]
train_sub <- train_sub[date <= as.Date("2017-08-10"), ]

#train_sub <- as.data.frame(train_sub)
cols <- c('date','store_item_nbr', 'unit_sales')
train_sub <- train_sub[, cols,with=FALSE]

# clean up
rm(train)

head(train_sub)

# train_valid <- train_sub[date >= as.Date("2017-08-01"), ] #2017-7-26 to 2017-08-10
# train_sub   <- train_sub[date < as.Date("2017-08-01"), ]

train_valid <- train_sub[date >= as.Date("2017-07-26"), ] #2017-7-26 to 2017-08-10
train_sub   <- train_sub[date < as.Date("2017-07-26"), ]


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

length(unique(train_sub$date))
# dcast the data from long to wide format for time series forecasting
train_sub_wide <- dcast(train_sub, store_item_nbr ~ date, value.var = "unit_sales", fill = 0)
train_ts <- ts(train_sub_wide, frequency = 7) # considering one week as a short shopping cycle
#head(train_sub_wide)


#as.Date("2017-08-10") - as.Date("2017-07-26")

fcst_intv = 16  # 16 days of forecast interval (Aug 16 ~ 31) per the submission requirement
fcst_matrix <- matrix(NA,nrow=nrow(train_ts),ncol=fcst_intv)



cl <- makeCluster(32)
registerDoParallel(cl)
#registerDoParallel(cores=detectCores()-1)
fcst_matrix <- foreach(i=1:nrow(train_sub_wide), .combine=rbind, .packages=c("prophet")) %dopar% {
  y <- unlist(train_sub_wide[i, -1])
  if (all(y==0)) 0 else {
    m.pro <- prophet(df=data.frame(ds=names(train_sub_wide[i, -1]), y=y), weekly.seasonality = TRUE) 
    fc <- make_future_dataframe(m.pro, periods=fcst_intv, include_history=F)
    fcst_matrix <- expm1(predict(m.pro, fc)$yhat) 
  }
}
stopCluster(cl)

fcst_matrix[fcst_matrix < 0] <- 0
colnames(fcst_matrix) <- as.character(seq(from = as.Date("2017-07-26"), #"2017-08-01"
                                          to = as.Date("2017-08-10"), #"2017-08-15"
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

min(fcst_df_long$fcst_date)
max(fcst_df_long$fcst_date)

# generate the final submission file
submission <- left_join(train_valid, fcst_df_long, 
                        c("store_item_nbr" = "store_item_nbr", 'date' = 'fcst_date'))
head(submission)

names(submission) <- c("date", "store_item_nbr", "unit_sales", "pred_cv_unit_sales")

submission$unit_sales[submission$unit_sales < 0] <- 0
submission$pred_cv_unit_sales[is.na(submission$pred_cv_unit_sales)] <- 0

write.csv(submission, "./submissions/prophet_validation.csv", row.names =  FALSE, quote = FALSE)


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
cat("CV Fold-", i, " ", metric, ": ", score(log1p(submission$unit_sales), log1p(submission$pred_cv_unit_sales), metric), "\n", sep = "")

























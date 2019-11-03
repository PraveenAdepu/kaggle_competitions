
library(dplyr)
library(forecast)
library(reshape2)
library(data.table)
library(foreach)
library(date)
library(lubridate)
#library(doMC)
library(doParallel) 

print(expm1(0))
print(expm1(1))
print(log1p(0))
print(log1p(1))


trainingSet <- readRDS("./input/trainingSet.rds")
head(trainingSet)
trainingSet <- as.data.table(trainingSet)
sort(unique(trainingSet$date))
train_sub <- trainingSet[date<"2017-08-01"]
sort(unique(train_sub$date))
train_sub$store_item_nbr <- paste(train_sub$store_nbr, train_sub$item_nbr, sep="_")

###############################################################################################################
train <- fread('./input/train.csv')
train$date <- as.Date(parse_date_time(train$date,'%y-%m-%d'))
train_validation <- trainingSet[date>="2017-08-01"]
train_validation$store_item_nbr <- paste(train_validation$store_nbr, train_validation$item_nbr, sep="_")
cols <- c('date','store_item_nbr', 'store_nbr','item_nbr', 'unit_sales')
train_validation <- train_validation[, cols,with=FALSE]
train_validation <- as.data.frame(train_validation)


items <- read_csv("./input/items.csv")
length(unique(items$item_nbr))

train_validation <- left_join(train_validation, items , by="item_nbr")
head(train_validation)

saveRDS(train_validation,"./input/train_validation.RDS")


test <- fread("./input/test.csv")
test$date <- as.Date(parse_date_time(test$date,'%y-%m-%d'))
test$store_item_nbr <- paste(test$store_nbr, test$item_nbr, sep="_")

test <- as.data.frame(test)

test_distinct <- test %>% distinct(store_nbr, item_nbr, store_item_nbr)
train$store_item_nbr <- paste(train$store_nbr, train$item_nbr, sep="_")
train$train_record <- 1
train_distinct <- train %>% distinct(store_item_nbr,train_record)

test_new_store_items <- left_join(test_distinct, train_distinct, by="store_item_nbr")

test_new_store_items$train_record[is.na(test_new_store_items$train_record)]<- 0

test_only_recods <- test_new_store_items %>% filter(train_record == 0)
saveRDS(test_only_recods, "./input/test_only_records.rds")

head(test_new_store_items)

################################################################################################################

#train_sub <- as.data.frame(train_sub)
cols <- c('date','store_item_nbr', 'unit_sales')
train_sub <- train_sub[, cols,with=FALSE]

# clean up
rm(train)

head(train_sub)

# transform to log1p
train_sub$unit_sales <- as.numeric(train_sub$unit_sales)
train_sub$unit_sales[train_sub$unit_sales < 0] <- 0
train_sub$unit_sales <- log1p(train_sub$unit_sales)

# dcast the data from long to wide format for time series forecasting
train_sub_wide <- dcast(train_sub, store_item_nbr ~ date, value.var = "unit_sales", fill = 0)
train_ts <- ts(train_sub_wide, frequency = 7) # considering one week as a short shopping cycle

fcst_intv = 15  # 16 days of forecast interval (Aug 16 ~ 31) per the submission requirement
fcst_matrix <- matrix(NA,nrow=nrow(train_ts),ncol=fcst_intv)

# register 15 cores for parallel processing in ETS forecasting
#registerDoMC(detectCores()-1)
registerDoParallel(cores=25)

fcst_matrix <- foreach(i=1:nrow(train_ts),.combine=rbind, .packages=c("forecast")) %dopar% { 
  fcst_matrix <- forecast(ets(train_ts[i,]),h=fcst_intv)$mean
}

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

#fcst_df_long %>% filter(unit_sales > 5)

# transform back to exp1p
fcst_df_long$unit_sales <- expm1(fcst_df_long$unit_sales)

###########################################################################################

submission <- inner_join(train_validation, fcst_df_long,c("store_item_nbr" = "store_item_nbr", 'date' = 'fcst_date'))
submission$unit_sales.x[is.na(submission$unit_sales.x)] <- 0
submission$unit_sales.y[is.na(submission$unit_sales.y)] <- 0
submission$unit_sales.x[submission$unit_sales.x < 0] <- 0

summary(submission$unit_sales.x)
summary(submission$unit_sales.y)
head(submission)

submission$weights <- ifelse(submission$perishable == 0, 1, 1.25)


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

NWrmse = sqrt(sum( submission$weights * (log1p(submission$unit_sales.x)-log1p(submission$unit_sales.y))^2)/sum(submission$weights))

NWrmse






mean01 <- read_csv("./submissions/Prav_mean01.csv")
lgbm01 <- read_csv("./submissions/Prav.lgbm001.full.csv")
prophet <- read_csv("./submissions/prav_prophet01.csv")
head(mean01); head(lgbm01);head(prophet)
submission <- left_join(mean01, lgbm01,by="id")
submission <- left_join(submission, prophet,by="id")
head(submission)
submission$unit_sales.y <- ifelse(submission$unit_sales.x == 0, 0, submission$unit_sales.y)
submission$unit_sales   <- ifelse(submission$unit_sales.x == 0, 0, submission$unit_sales)
submission$ensem        <-  submission$unit_sales.y * 0.5 + submission$unit_sales * 0.5
ensemble.features <- setdiff(names(submission),"id")
cor(submission[, ensemble.features])
summary(submission$unit_sales.x)
summary(submission$unit_sales.y)
summary(submission$unit_sales)

sum(submission$unit_sales.x)
sum(submission$unit_sales.y)
sum(submission$unit_sales)

submission$unit_sales <- NULL
names(submission)[2] <- "unit_sales"

write.csv(submission,"./submissions/prav_lgbm01_noNewprods.csv", row.names = FALSE, quote = FALSE)

# generate the final submission file
# submission <- left_join(test, fcst_df_long, 
#                         c("store_item_nbr" = "store_item_nbr", 'date' = 'fcst_date'))
# submission$unit_sales[is.na(submission$unit_sales)] <- 0
# head(submission)
# summary(submission$unit_sales)
# submission <- submission %>% select(id, unit_sales)
# write.csv(submission, "./submissions/Prav_ets01.csv", row.names = FALSE, quote = FALSE)



require(dplyr)
require(data.table)
require(stringr)
require(prophet)
require(doParallel)

train <- fread('./input/air_reserve.csv')

train$visit_date <- as.Date(train$visit_datetime)
# train$reserve_datetime <- NULL
# names(train) <- c("air_store_id","visit_date","visitors")


head(train)
train_train <- train %>% filter(visit_date < "2017-03-12")

train <- train %>% filter(visit_date >= "2017-03-12")


train <- train %>% 
  arrange(air_store_id,visit_date,reserve_datetime) %>%
  group_by(air_store_id,visit_date) %>%
  mutate(rank=row_number())


train_testset <- train %>% filter(visit_date >= "2017-04-23")
train_cvset <- train %>% filter(visit_date < "2017-04-23")

head(train_cvset,10); head(train_testset,10)



train_testset$visit_date <- as.Date(train_testset$visit_date) - 42

train_cvset_filter <- inner_join(train_cvset,train_testset, by=c("air_store_id","visit_date","rank"), all.x = TRUE)

head(train_train)

head(train_cvset_filter)

train_cvset_filter$visit_datetime.y <- NULL
train_cvset_filter$reserve_datetime.y <- NULL
train_cvset_filter$reserve_visitors.y <- NULL
train_cvset_filter$visit_date <- NULL
train_cvset_filter$rank <- NULL

head(train_testset,10)
train_testset$visit_date <- NULL
train_testset$rank <- NULL

names(train_cvset_filter) <- names(train_testset)

train_train$visit_date <- NULL
head(train_train)
head(train_cvset_filter)
head(train_testset,10)

train_train <- as.data.frame(train_train)
train_cvset_filter <- as.data.frame(train_cvset_filter)
train_testset <- as.data.frame(train_testset)
air_reserve_cvAdjusted <- rbind(train_train, train_cvset_filter,train_testset)

write.csv(air_reserve_cvAdjusted, "./input/air_reserve_cvAdjusted.csv", row.names = FALSE)

###############################################################################################################################

train <- fread('./input/hpg_reserve.csv')

train$visit_date <- as.Date(train$visit_datetime)
# train$reserve_datetime <- NULL
# names(train) <- c("air_store_id","visit_date","visitors")


head(train)
train_train <- train %>% filter(visit_date < "2017-03-12")

train <- train %>% filter(visit_date >= "2017-03-12")


train <- train %>% 
  arrange(hpg_store_id,visit_date,reserve_datetime) %>%
  group_by(hpg_store_id,visit_date) %>%
  mutate(rank=row_number())


train_testset <- train %>% filter(visit_date >= "2017-04-23")
train_cvset <- train %>% filter(visit_date < "2017-04-23")

head(train_cvset,10); head(train_testset,10)



train_testset$visit_date <- as.Date(train_testset$visit_date) - 42

train_cvset_filter <- inner_join(train_cvset,train_testset, by=c("hpg_store_id","visit_date","rank"), all.x = TRUE)

head(train_train)

head(train_cvset_filter)

train_cvset_filter$visit_datetime.y <- NULL
train_cvset_filter$reserve_datetime.y <- NULL
train_cvset_filter$reserve_visitors.y <- NULL
train_cvset_filter$visit_date <- NULL
train_cvset_filter$rank <- NULL

head(train_testset,10)
train_testset$visit_date <- NULL
train_testset$rank <- NULL

names(train_cvset_filter) <- names(train_testset)

train_train$visit_date <- NULL
head(train_train)
head(train_cvset_filter)
head(train_testset,10)

train_train <- as.data.frame(train_train)
train_cvset_filter <- as.data.frame(train_cvset_filter)
train_testset <- as.data.frame(train_testset)
air_reserve_cvAdjusted <- rbind(train_train, train_cvset_filter,train_testset)

write.csv(air_reserve_cvAdjusted, "./input/hpg_reserve_cvAdjusted.csv", row.names = FALSE)


# "2017-03-12" , "2017-04-22"

# "2017-04-23" , "2017-05-31"


train <- train %>% group_by(air_store_id, visit_date) %>% summarise(visitors = sum(visitors))
head(train)



date <- fread('./input/date_info.csv')
date$calendar_date <- as.Date(date$calendar_date)

train_wide <- dcast(train, air_store_id ~ visit_date, value.var = "visitors", fill = 0)
id_train   <- train_wide$air_store_id
train_wide$air_store_id <- NULL


test_wide <- dcast(test, air_store_id ~ visit_date, value.var = "visitors", fill = 0)
id_test <- test_wide$air_store_id; 
test_wide$air_store_id <- NULL

test_dates <- sort(unique(test$visit_date))
train_dates <- sort(unique(train$visit_date))

# we want to mirror the composition (length + days of week) of the test set in our choice of validation
valid_dates <- train_dates[437:475]

print(summary(valid_dates))

names(train_wide)

x0 <- data.frame(train_wide)[, colnames(train_wide) < as.Date('2017-03-12')]
x1 <- data.frame(train_wide)[, (colnames(train_wide) >= as.Date('2017-03-12')) & (colnames(train_wide) <= as.Date('2017-04-19') )]

cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)

xfor <- foreach(i=1:nrow(x0),.combine=rbind, .packages=c( 'prophet')) %dopar% { 
  
  xseries <- log1p(unname(unlist(x0[i,])))
  # format prophet-style
  xmat <- data.frame(ds = seq(from = as.Date('2016-01-01'), 
                              to = as.Date('2017-03-11'), length.out =  length(xseries))  , 
                     y = xseries)
  # prediction
  m <- prophet(xmat, 
               #yearly.seasonality = T,
               weekly.seasonality = T,
               n.changepoints = 28, uncertainty.samples = 2)
  future <- make_future_dataframe(m, periods = 39, include_history = F)
  xforecast <- predict(m, future)
  xforecast$yhat  
}

stopCluster(cl)

xfor <- expm1(xfor)
pred_val <- as.data.frame(xfor)
names(pred_val) <- names(x1)
pred_val$air_store_id <- id_train
names(pred_val)
head(pred_val)
# melt the forecast data frame from wide to long format for final submission
pred_val_long <- melt(pred_val, id = 'air_store_id', 
                      variable.name = "visit_date", 
                      value.name = 'visitors')

pred_val_long$visit_date <- sub("X","",pred_val_long$visit_date)
pred_val_long$visit_date <- gsub("\\.","-",pred_val_long$visit_date)
pred_val_long$visit_date <- as.Date(as.character(pred_val_long$visit_date))
pred_val_long$visitors <- as.numeric(pred_val_long$visitors)

head(pred_val_long)
names(pred_val_long)[3] <- "visitors_pred"

x_val <- train %>% filter(visit_date >= as.Date("2017-03-12") & visit_date <= as.Date('2017-04-19') )
  #train[visit_date >= as.Date("2017-03-12") & visit_date <= as.Date('2017-04-19')]

pred_validation <- left_join(pred_val_long, x_val, by =c("air_store_id","visit_date"))

head(pred_validation)
#train[visit_date == as.Date("2017-03-12") & air_store_id <= "air_00a91d42b08b08d9"] # 4,603 NA
pred_validation[is.na(pred_validation)] <- 0

head(pred_validation)

#pred_validation_adjust <- pred_validation %>% filter(visitors != -100) # 27,728

score(log1p(pred_validation$visitors), log1p(pred_validation$visitors_pred),"rmse")

write.csv(pred_validation, "./submissions/Prav_prophet_validation_Reservepreds.csv",row.names = FALSE, quote = FALSE)



#####################################################################################################################################################################
#####################################################################################################################################################################
#  Test
#####################################################################################################################################################################
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)

xfor <- foreach(i=1:nrow(train_wide),.combine=rbind, .packages=c( 'prophet')) %dopar% { 
  
  xseries <- log1p(unname(unlist(train_wide[i,])))
  # format prophet-style
  xmat <- data.frame(ds = seq(from = as.Date('2016-01-01'), 
                              to = as.Date('2017-04-22'), length.out =  length(xseries))  , 
                     y = xseries)
  # prediction
  m <- prophet(xmat, 
               yearly.seasonality = T,
               weekly.seasonality = T,
               n.changepoints = 25, uncertainty.samples = 1)
  future <- make_future_dataframe(m, periods = 39, include_history = F)
  xforecast <- predict(m, future)
  xforecast$yhat  
}

stopCluster(cl)

xfor <- expm1(xfor)
pred_test <- as.data.frame(xfor)
names(pred_test)  <- as.character(seq(from = as.Date("2017-04-23"), 
                                      to = as.Date("2017-05-31"), 
                                      by = 'day'))
pred_test$air_store_id <- id_train
names(pred_test)
head(pred_test)
# melt the forecast data frame from wide to long format for final submission
pred_test_long <- melt(pred_test, id = 'air_store_id', 
                       variable.name = "visit_date", 
                       value.name = 'visitors')

# pred_test_long$visit_date <- sub("X","",pred_test_long$visit_date)
# pred_test_long$visit_date <- gsub("\\.","-",pred_test_long$visit_date)
pred_test_long$visit_date <- as.Date(as.character(pred_test_long$visit_date))
pred_test_long$visitors <- as.numeric(pred_test_long$visitors)

head(pred_test_long)
names(pred_test_long)[3] <- "visitors_pred"


pred_test_long[is.na(pred_test_long)] <- 0

head(pred_test_long)


write.csv(pred_test_long, "./submissions/Prav.Prophet01full-test.csv",row.names = FALSE, quote = FALSE)


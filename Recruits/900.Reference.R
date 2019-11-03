
require(dplyr)
require(data.table)
require(stringr)
require(prophet)
#require(doMC)
require(doParallel)


xtrain <- fread('./input/air_visit_data.csv')
xtest <- fread('./input/sample_submission.csv')

print(paste('nof rows: ', dim(xtrain)[1]))
head(xtrain)
print(paste('unique ids: ', length(unique(xtrain$air_store_id))))

# dates in the training set
xtrain$visit_date <- as.Date(xtrain$visit_date)

# dates in the test set
xtest$visit_date <- str_sub(xtest$id, -10)
xtest$air_store_id <- str_sub(xtest$id, 1,-12)
xtest$visit_date <- as.Date(xtest$visit_date)

print(summary(xtrain$visit_date))
print(summary(xtest$visit_date))

xdate <- fread('./input/date_info.csv')
xdate$calendar_date <- as.Date(xdate$calendar_date)

xtrain_wide <- dcast(xtrain, air_store_id ~ visit_date, value.var = "visitors", fill = 0)
id_train <- xtrain_wide$air_store_id; xtrain_wide$air_store_id <- NULL


xtest_wide <- dcast(xtest, air_store_id ~ visit_date, value.var = "visitors", fill = 0)
id_test <- xtest_wide$air_store_id; xtrain_wide$air_store_id <- NULL

test_dates <- sort(unique(xtest$visit_date))
train_dates <- sort(unique(xtrain$visit_date))

# we want to mirror the composition (length + days of week) of the test set in our choice of validation
valid_dates <- train_dates[437:475]

print(summary(valid_dates))


x0 <- data.frame(xtrain_wide)[, colnames(xtrain_wide) < as.Date('2017-03-12')]
x1 <- data.frame(xtrain_wide)[, (colnames(xtrain_wide) >= as.Date('2017-03-12')) & (colnames(xtrain_wide) <= as.Date('2017-04-19') )]

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
               yearly.seasonality = T,
               weekly.seasonality = T,
               n.changepoints = 25, uncertainty.samples = 1)
  future <- make_future_dataframe(m, periods = 39, include_history = F)
  xforecast <- predict(m, future)
  xforecast$yhat  
}

stopCluster(cl)

sqrt( mean( (xfor - log1p(x1))^2  ) )

#####################################################################################################################################################################
#####################################################################################################################################################################


xtrain$dow <- wday(xtrain$visit_date)
# x0 <- xtrain[visit_date <= as.Date('2017-03-11') ]
x0 <- xtrain[visit_date <= as.Date('2017-03-11') & visit_date >= as.Date('2017-01-28')]
x1 <- xtrain[visit_date >= as.Date('2017-03-12') & visit_date <= as.Date('2017-04-19')]

# dictionary
xdict <- x0[, j=list(med_for = median(visitors)), 
            by = list(air_store_id,dow)]
# create the forecast 
xfor2 <- merge(x1, xdict, all.x = T)
xfor2$med_for[is.na(xfor2$med_for)] <- 0

# and check the performance
sqrt( mean( (log1p(xfor2$med_for) - log1p(x1$visitors))^2  ) )

# full data: 1.06864192708676

#####################################################################################################################################################################
#####################################################################################################################################################################

cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)

xfor <- foreach(i=1:nrow(xtrain_wide),.combine=rbind, .packages=c( 'prophet')) %dopar% { 
  
  xseries <- log1p(unname(unlist(xtrain_wide[i,])))
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

prval  <- data.frame(xfor)
colnames(prval) <- test_dates
prval$id <- id_train
prval <- melt(prval, id = 'id',  variable.name = "visit_date", value.name = 'forecast')
prval$forecast <- pmax(expm1(prval$forecast),0)
summary(prval$forecast)
prval$visit_date <- as.Date(prval$visit_date)
setnames(prval, 'id', 'air_store_id')
# merge with test
forecast_full <- merge(xtest, prval, all.x = T)
# correct id
forecast_full$id <- paste(forecast_full$air_store_id, as.character(forecast_full$visit_date),sep = '_')
# subset to the relevant columns and store
forecast_full <- forecast_full[ , c('id', 'forecast'), with = TRUE]
setnames(forecast_full, 'forecast', 'visitors')

write.csv(forecast_full, 'prophecies.csv', row.names = F, quote = F)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simple port of existing python script of hklee
#
# https://www.kaggle.com/zeemeen/weighted-mean-comparisons-lb-0-497-1st
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



cat('load packages and data')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
require(data.table)
require(stringr)

date_info <- fread('./input/date_info.csv')
air_visit_data <- fread('./input/air_visit_data.csv')
sample_submission <- fread('./input/sample_submission.csv')

air_visit_data <- as.data.frame(air_visit_data)

head(air_visit_data)
X_build <- air_visit_data %>% filter(visit_date < '2017-03-12' )

min(X_build$visit_date); max(X_build$visit_date)

X_valid <- air_visit_data %>% filter(visit_date >= '2017-03-12' & visit_date <= '2017-04-19')

min(X_valid$visit_date); max(X_valid$visit_date)


cat('holidays at weekends are not special, right?')
wkend_holidays <- which(date_info$day_of_week %in% c("Saturday", "Sunday") & date_info$holiday_flg ==1)
date_info[wkend_holidays, 'holiday_flg' := 0]

cat('add decreasing weights from now')
date_info[, 'weight' := (.I/.N) ^ 7]

cat('weighted mean visitors for each (air_store_id, day_of_week, holiday_flag) or (air_store_id, day_of_week)')

visit_data = merge(X_build, date_info, by.x = 'visit_date', by.y = 'calendar_date', all.x= TRUE)
head(visit_data)
visit_data = as.data.table(visit_data)
#visit_data[, 'calendar_date' := NULL]
visit_data[, 'visitors':= log1p(visitors)]

visitors = visit_data[,.(visitors = weighted.mean(visitors, weight)), by = c('air_store_id', 'day_of_week', 'holiday_flg')]


store_day_visitors = visitors[holiday_flg == 0,]
store_visitors = visitors[,.(visitors = mean(visitors)), by = c('air_store_id')]

head(store_day_visitors)
head(store_visitors)
head(visitors)

cat('prepare to merge with date_info and visitors')
# 
# sample_submission[, 'air_store_id' := str_sub(id, 1,-12)]
# sample_submission[, 'calendar_date' := str_sub(id, -10)]                    
# sample_submission[, 'visitors' := NULL]     

head(X_valid)
names(X_valid) <- c("air_store_id","calendar_date","visitors_actual")

X_valid <- merge(X_valid, date_info, by = 'calendar_date', all.x = TRUE)
X_valid <- merge(X_valid, visitors, by = c('air_store_id', 'day_of_week', 'holiday_flg'), all.x = TRUE)

X_valid <- merge(X_valid, store_day_visitors, by = c('air_store_id', 'day_of_week'), all.x = TRUE)

X_valid <- merge(X_valid, store_visitors, by = c('air_store_id'), all.x = TRUE)

X_valid$visitor_pred <- ifelse(is.na(X_valid$visitors.x), X_valid$visitors.y, X_valid$visitors.x)
X_valid$visitor_pred <- ifelse(is.na(X_valid$visitor_pred), X_valid$visitors, X_valid$visitor_pred)

X_valid <- as.data.table(X_valid)

# fill missings with (air_store_id, day_of_week)
# 
# missings = which(is.na(sample_submission$visitors))
# sample_submission[missings][['visitors']] <- merge(sample_submission[missings, -'visitors'], visitors[holiday_flg==0], by = c('air_store_id', 'day_of_week'), all.x = TRUE)[['visitors']]
# 
# 
# # fill missings with (air_store_id)
# missings = which(is.na(sample_submission$visitors))
# 
# sample_submission[missings][['visitors']] <- merge(sample_submission[missings, -'visitors'], visitors[, .(visitors = mean(visitors)), by = 'air_store_id'], by = 'air_store_id', all.x = TRUE)[['visitors']]


X_valid[, 'visitor' := expm1(visitor_pred)]
X_valid$day_of_week <- NULL
X_valid$holiday_flg.x <- NULL
X_valid$weight <- NULL
X_valid$visitors.x <- NULL
X_valid$holiday_flg.y <- NULL
X_valid$visitors.y <- NULL
X_valid$visitors <- NULL
X_valid$visitor_pred <- NULL


X_valid <- as.data.frame(X_valid)



names(X_valid) <- c("air_store_id","visit_date","visitors","visitors_pred")

score(log1p(X_valid$visitors), log1p(X_valid$visitors_pred),"rmse")

summary(X_valid$visitors); summary(X_valid$visitors_pred)


write.csv(X_valid, file = './submissions/Prav_MA_validation_preds.csv', row.names = FALSE)

cat("done")


############################################################################################################################################################

cat('load packages and data')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
require(data.table)
require(stringr)

date_info <- fread('./input/date_info.csv')
air_visit_data <- fread('./input/air_visit_data.csv')
sample_submission <- fread('./input/sample_submission.csv')


head(date_info)
tail(date_info)

cat('holidays at weekends are not special, right?')
wkend_holidays <- which(date_info$day_of_week %in% c("Saturday", "Sunday") & date_info$holiday_flg ==1)
date_info[wkend_holidays, 'holiday_flg' := 0]

cat('add decreasing weights from now')
date_info[, 'weight' := (.I/.N) ^ 7]

cat('weighted mean visitors for each (air_store_id, day_of_week, holiday_flag) or (air_store_id, day_of_week)')

visit_data = merge(air_visit_data, date_info, by.x = 'visit_date', by.y = 'calendar_date', all.x= TRUE)
head(visit_data)
#visit_data[, 'calendar_date' := NULL]
visit_data[, 'visitors':= log1p(visitors)]

visitors = visit_data[,.(visitors = weighted.mean(visitors, weight)), by = c('air_store_id', 'day_of_week', 'holiday_flg')]


store_day_visitors = visitors[holiday_flg == 0,]
store_visitors = visitors[,.(visitors = mean(visitors)), by = c('air_store_id')]

head(store_day_visitors)
head(store_visitors)
head(visitors)

cat('prepare to merge with date_info and visitors')

sample_submission[, 'air_store_id' := str_sub(id, 1,-12)]
sample_submission[, 'calendar_date' := str_sub(id, -10)]                    
sample_submission[, 'visitors' := NULL]     

head(sample_submission)

sample_submission <- merge(sample_submission, date_info, by = 'calendar_date', all.x = TRUE)
sample_submission <- merge(sample_submission, visitors, by = c('air_store_id', 'day_of_week', 'holiday_flg'), all.x = TRUE)

sample_submission <- merge(sample_submission, store_day_visitors, by = c('air_store_id', 'day_of_week'), all.x = TRUE)

sample_submission <- merge(sample_submission, store_visitors, by = c('air_store_id'), all.x = TRUE)

sample_submission$visitor_pred <- ifelse(is.na(sample_submission$visitors.x), sample_submission$visitors.y, sample_submission$visitors.x)
sample_submission$visitor_pred <- ifelse(is.na(sample_submission$visitor_pred), sample_submission$visitors, sample_submission$visitor_pred)

# fill missings with (air_store_id, day_of_week)
# 
# missings = which(is.na(sample_submission$visitors))
# sample_submission[missings][['visitors']] <- merge(sample_submission[missings, -'visitors'], visitors[holiday_flg==0], by = c('air_store_id', 'day_of_week'), all.x = TRUE)[['visitors']]
# 
# 
# # fill missings with (air_store_id)
# missings = which(is.na(sample_submission$visitors))
# 
# sample_submission[missings][['visitors']] <- merge(sample_submission[missings, -'visitors'], visitors[, .(visitors = mean(visitors)), by = 'air_store_id'], by = 'air_store_id', all.x = TRUE)[['visitors']]


sample_submission[, 'visitors' := expm1(visitor_pred)]
sample_submission$day_of_week <- NULL
sample_submission$holiday_flg.x <- NULL
sample_submission$weight <- NULL
sample_submission$visitors.x <- NULL
sample_submission$holiday_flg.y <- NULL
sample_submission$visitors.y <- NULL
sample_submission$visitor_pred <- NULL

write.csv(sample_submission, file = './submissions/Prav_MAfull-test.csv', row.names = FALSE)

cat("done")
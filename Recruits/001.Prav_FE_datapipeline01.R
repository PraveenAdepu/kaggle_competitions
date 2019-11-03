
require(stringr)


train <- read_csv('./input/air_visit_data.csv')
test <- read_csv('./input/sample_submission.csv')

head(train)
head(test)

print(paste('nof rows: ', dim(train)[1]))
print(paste('unique ids: ', length(unique(train$air_store_id))))

# dates in the training set
train$visit_date <- as.Date(train$visit_date)

# dates in the test set
test$visit_date   <- str_sub(test$id, -10)
test$air_store_id <- str_sub(test$id, 1,-12)
test$visit_date   <- as.Date(test$visit_date)

print(summary(train$visit_date))
print(summary(test$visit_date))

date <- read_csv('./input/date_info.csv')
date$calendar_date <- as.Date(date$calendar_date)
date$dow   <- as.numeric(format(date$calendar_date, format = "%u"))
date$day_of_week <- NULL

ar <- read_csv('./input/air_reserve.csv')
hr <- read_csv('./input/hpg_reserve.csv')
id <- read_csv('./input/store_id_relation.csv')
head(ar)
head(hr)



hr <- inner_join(hr, id, by="hpg_store_id")

ar$visit_datetime <- as.Date(ar$visit_datetime)
names(ar)[2] <- "visit_date"

hr$visit_datetime <- as.Date(hr$visit_datetime)
names(hr)[2] <- "visit_date"

ar$reserve_datetime <- as.Date(ar$reserve_datetime)
hr$reserve_datetime <- as.Date(hr$reserve_datetime)

ar$reserve_datetime_diff <- as.numeric(difftime(ar$visit_date ,ar$reserve_datetime , units = c("days")))
hr$reserve_datetime_diff <- as.numeric(difftime(hr$visit_date ,hr$reserve_datetime , units = c("days")))



ar <- ar %>%
        group_by(air_store_id, visit_date) %>%
        summarise( ar_reserve_datetime_diff_min = min(reserve_datetime_diff)
                  ,ar_reserve_datetime_diff_max = max(reserve_datetime_diff)
                  ,ar_reserve_datetime_diff_mean = mean(reserve_datetime_diff)
                  ,ar_reserve_datetime_diff_std = sd(reserve_datetime_diff)
                  ,ar_reserve_datetime_diff_sum = sum(reserve_datetime_diff)
                  ,ar_reserve_visitors = sum(reserve_visitors)
                  ,ar_reserve_visitors_count = n())



hr <- hr %>%
        group_by(air_store_id, visit_date) %>%
        summarise( hr_reserve_datetime_diff_min = min(reserve_datetime_diff)
                  ,hr_reserve_datetime_diff_max = max(reserve_datetime_diff)
                  ,hr_reserve_datetime_diff_mean = mean(reserve_datetime_diff)
                  ,hr_reserve_datetime_diff_std = sd(reserve_datetime_diff)
                  ,hr_reserve_datetime_diff_sum = sum(reserve_datetime_diff)
                  ,hr_reserve_visitors = sum(reserve_visitors)
                  ,hr_reserve_visitors_count = n())



train$year  <- as.integer(format(as.Date(train$visit_date, format="%d/%m/%Y"),"%Y"))
train$month <- as.integer(format(as.Date(train$visit_date, format="%d/%m/%Y"),"%m"))
train$dow   <- as.numeric(format(train$visit_date, format = "%u"))

test$year  <- as.integer(format(as.Date(test$visit_date, format="%d/%m/%Y"),"%Y"))
test$month <- as.integer(format(as.Date(test$visit_date, format="%d/%m/%Y"),"%m"))
test$dow   <- as.numeric(format(test$visit_date, format = "%u"))



as  <- read_csv('./input/air_store_info.csv')
head(as)

as$air_genre_name <- as.integer(as.factor(as$air_genre_name))
as$air_area_name  <- as.integer(as.factor(as$air_area_name))

as <- as %>% 
        group_by(air_genre_name) %>%
        mutate(air_genre_count = n()) %>%
        ungroup() %>%
        group_by(air_area_name) %>%
        mutate(air_area_count = n()) %>%
        ungroup() %>%
        group_by(air_genre_name,air_area_name) %>%
        mutate(air_genre_area_count = n())


build <-  train %>% filter(visit_date < as.Date("2017-03-12"))
valid <-  train %>% filter(visit_date >= as.Date("2017-03-12") & visit_date <= as.Date("2017-04-19"))


build_features <- build %>% group_by(air_store_id, dow) %>%
                    summarise(min_visitors = min(visitors)
                              , mean_visitors = mean(visitors)
                              , median_visitors = median(visitors)
                              , max_visitors = max(visitors)
                              , count_observations = n()
                              , std_visitors = sd(visitors)
                    )


valid_stores <- unique(valid$air_store_id)

valid_stores <- as.data.frame(valid_stores)
names(valid_stores) <- "air_store_id"

valid_stores$dow <- 0

val_stores <- valid_stores

for(i in seq(1:6))
{
  
  valid_stores$dow <- i
  val_stores <- rbind(val_stores, valid_stores)
}

rm(valid_stores)

head(date,20)
names(date)[1] <- "visit_date"
head(build)
head(valid)
build <- left_join(build, date, by=c("visit_date","dow"))
valid <- left_join(valid, date, by=c("visit_date","dow"))

head(as)

build <- left_join(build, as, by=c("air_store_id"))
valid <- left_join(valid, as, by=c("air_store_id"))

head(ar)
build <- left_join(build, ar, by=c("air_store_id","visit_date"))
valid <- left_join(valid, ar, by=c("air_store_id","visit_date"))


head(hr)
build <- left_join(build, hr, by=c("air_store_id","visit_date"))
valid <- left_join(valid, hr, by=c("air_store_id","visit_date"))


build <- left_join(build, build_features, by=c("air_store_id","dow"))
valid <- left_join(valid, build_features, by=c("air_store_id","dow"))


write.csv(build,"./input/X_build.csv", row.names = FALSE, quote = FALSE)
write.csv(valid,"./input/X_valid.csv", row.names = FALSE, quote = FALSE)





head(train)



head(build)
head(valid)

head(build_store)

build_features <- build %>% group_by(air_store_id, dow) %>%
                  summarise(min_visitors = min(visitors)
                            , mean_visitors = mean(visitors)
                            , median_visitors = median(visitors)
                            , max_visitors = max(visitors)
                            , count_observations = n()
                            , std_visitors = sd(visitors)
                             )

head(build_features,10)
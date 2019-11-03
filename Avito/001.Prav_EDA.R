###############################################################################################################################
# 01. Read all source files
###############################################################################################################################

train <- read_csv("./input/train.csv")
test  <- read_csv("./input/test.csv")

head(train)
head(test)

train_activationdate_count <- train %>% mutate(WeekDay = weekdays(activation_date))  %>% group_by(activation_date, WeekDay) %>% summarise(rowCount = n())

test_activationdate_count <- test %>% mutate(WeekDay = weekdays(activation_date))  %>% group_by(activation_date, WeekDay)  %>% summarise(rowCount = n())

###############################################################################################################################
# 02. Prav - decision to make cv schema
#          - trainingSet from 2017-03-15 to 2017-03-30 and exclude later dates

# two possible cv schema ideas
#          - 5 folds by weekday stratified, time based training and to time based test
#          - 5 folds by random sampling

# third possible cv schema - last one to test, follow forum for experiences
#          - treat the problem as time series and use >= 2017-03-26 for validation
###############################################################################################################################

train_subset <- train %>% filter(activation_date <= "2017-03-30")

train_subset %>% mutate(WeekDay = weekdays(activation_date))  %>% group_by(activation_date, WeekDay) %>% summarise(rowCount = n())

train_subset <- train_subset %>% mutate(WeekDay = weekdays(activation_date))

length(unique(train_subset$item_id))
dim(train_subset)
###############################################################################################################################
# CV 5 folds - Weekday stratified
# Use python generated cv schema file

Prav_5folds_CVindices <- read_csv("./input/Prav_5folds_CVindices_weekdayStratified.csv")

train_subset <- left_join(train_subset, Prav_5folds_CVindices, by="item_id")

CVIndices <- train_subset %>% group_by(activation_date, WeekDay, CVindices) %>% summarise(rowCount = n())

# folds 3,4,5 has more similar to test presentation on last day distribution
###############################################################################################################################


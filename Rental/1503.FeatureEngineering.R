suppressMessages(library("jsonlite"))
suppressMessages(library("dplyr"))
suppressMessages(library("purrr"))
suppressMessages(library("RecordLinkage"))

lst.trainData <- fromJSON("./input/train.json")

vec.variables <- setdiff(names(lst.trainData), c("photos", "features"))

df.train <-map_at(lst.trainData, vec.variables, unlist) %>% tibble::as_tibble(.)

df.train$distance_lv <- levenshteinSim(tolower(df.train$street_address),tolower(df.train$display_address))
df.train$distance_lv_partition <- ifelse(df.train$distance_lv >=0.5,1,0)


lst.testData <- fromJSON("./input/test.json")

df.test <-map_at(lst.testData, vec.variables, unlist) %>% tibble::as_tibble(.)

df.test$distance_lv <- levenshteinSim(tolower(df.test$street_address),tolower(df.test$display_address))
df.test$distance_lv_partition <- ifelse(df.test$distance_lv >=0.5,1,0)

head(df.train$manager_id)

names(df.train)
cols <- c("listing_id","manager_id","created","latitude","longitude","interest_level","distance_lv","distance_lv_partition")

###############################################################################################################
temp <- subset(df.train[, cols], manager_id == "a10db4590843d78c784171a107bdacb4")

temp$created <- as.POSIXct( temp$created)
temp <- temp[ order(temp$created , decreasing = FALSE ),]

temp$date = as.Date(temp$created, format="%Y-%m-%d")

temp <- as.data.table(temp)
head(temp, 25)

temp[, OrderRank := 1:.N, by = c("manager_id", "date")]

normalit<-function(m){
  (m - min(m))/(max(m)-min(m))
}

temp[,list(OrderRankNorm=normalit(OrderRank)),by=list(manager_id,date)]

temp <- temp %>%
          group_by(manager_id,date) %>%
          mutate(OrderRankNorm = normalit(OrderRank))

temp <- as.data.table(temp)
head(temp, 25)

temp$OrderRankNorm[is.nan(temp$OrderRankNorm)] <- 0

###############################################################################################################

df.test$interest_level <- "no"

training <- df.train[, cols]
testing  <- df.test[, cols]

all_data <- rbind(training, testing)

head(all_data)


all_data$created <- as.POSIXct( all_data$created)
all_data <- all_data[ order(all_data$created , decreasing = FALSE ),]

all_data$date = as.Date(all_data$created, format="%Y-%m-%d")

all_data <- as.data.table(all_data)
head(all_data, 25)

all_data[, AscOrderRank := 1:.N, by = c("manager_id", "date")]

normalit<-function(m){
  (m - min(m))/(max(m)-min(m))
}


all_data <- all_data %>%
              group_by(manager_id,date) %>%
              mutate(AscOrderRankNorm = normalit(AscOrderRank))

all_data <- as.data.table(all_data)
head(all_data, 25)

all_data$AscOrderRankNorm[is.nan(all_data$AscOrderRankNorm)] <- 0

############################################################################################################

all_data <- all_data[ order(all_data$created , decreasing = TRUE ),]

all_data[, DescOrderRank := 1:.N, by = c("manager_id", "date")]


all_data <- all_data %>%
  group_by(manager_id,date) %>%
  mutate(DescOrderRankNorm = normalit(DescOrderRank))

all_data <- as.data.table(all_data)
head(all_data, 25)

all_data$DescOrderRankNorm[is.nan(all_data$DescOrderRankNorm)] <- 0

head(all_data,10)

#############################################################################################################

all_data <- as.data.frame(all_data)

train.df <- subset(all_data, interest_level != "no")
test.df  <- subset(all_data, interest_level == "no")

final.features <- c("listing_id","distance_lv","distance_lv_partition","AscOrderRank", "AscOrderRankNorm", "DescOrderRank", "DescOrderRankNorm")

write.csv(train.df[,final.features],  './input/Prav_train_features10.csv', row.names=FALSE, quote = FALSE)
write.csv(test.df[,final.features],  './input/Prav_test_features10.csv', row.names=FALSE, quote = FALSE)



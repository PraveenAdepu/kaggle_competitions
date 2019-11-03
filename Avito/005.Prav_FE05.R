
library(tidyverse)
library(lubridate)
library(magrittr)
library(text2vec)
library(tokenizers)
library(stopwords)
library(xgboost)
library(Matrix)
set.seed(2018)

#---------------------------
cat("Loading data...\n")
tr <- read_csv("./input/train.csv") 
te <- read_csv("./input/test.csv")


Prav_5fold_CVindices <- read_csv("./input/Prav_5folds_CVindices_weekdayStratified.csv") 

tr <- inner_join(tr, Prav_5fold_CVindices, by="item_id")

tr[is.na(tr)]<- 0
te[is.na(te)]<- 0

cv = 5

names(X_build)

for (i in 1:cv)
  
{
  
  cat(i ,"fold Processing\n")
  
  cat("X_build fold Processing\n")
  X_build <- subset(tr, CVindices != i)
  cat("X_val fold Processing\n")
  X_val   <- subset(tr, CVindices == i) 
  
  item_seq_category_price_mean_encode <- X_build %>% 
                                            group_by( category_name) %>%
                                            summarise(item_seq_number_category_price_mean_encode = mean(price)
                                                      , item_seq_number_category_target_mean_encode = mean(deal_probability))
  
  region_price_mean_encode <- X_build %>% 
                                              group_by( region) %>%
                                              summarise(region_price_mean_encode = mean(price)
                                                        , region_target_mean_encode = mean(deal_probability))
  
  city_price_mean_encode <- X_build %>% 
                                        group_by( city) %>%
                                        summarise(city_price_mean_encode = mean(price)
                                                  , city_target_mean_encode = mean(deal_probability))
  
  
  
  X_build <- left_join(X_build, item_seq_category_price_mean_encode, by=c("category_name"))
  X_val   <- left_join(X_val  , item_seq_category_price_mean_encode, by=c("category_name"))
  te1     <- left_join(te     , item_seq_category_price_mean_encode, by=c("category_name"))
  
  X_build <- left_join(X_build, region_price_mean_encode, by=c("region"))
  X_val   <- left_join(X_val  , region_price_mean_encode, by=c("region"))
  te1     <- left_join(te1    , region_price_mean_encode, by=c("region"))
  
  X_build <- left_join(X_build, city_price_mean_encode, by=c("city"))
  X_val   <- left_join(X_val  , city_price_mean_encode, by=c("city"))
  te1     <- left_join(te1    , city_price_mean_encode, by=c("city"))
  
  
  X_build[is.na(X_build)] <- 0
  X_val[is.na(X_val)] <- 0
  te1[is.na(te1)] <- 0
  encode_features <- c("item_id","item_seq_number_category_price_mean_encode","item_seq_number_category_target_mean_encode"
                       ,"region_price_mean_encode","region_target_mean_encode"
                       ,"city_price_mean_encode","city_target_mean_encode")
  
  if(i == 1)
  {
    write.csv(X_build[,encode_features], './input/Prav.build.fold1.csv', row.names=FALSE, quote = FALSE)
    write.csv(X_val[,encode_features]  , './input/Prav.val.fold1.csv', row.names=FALSE, quote = FALSE)
    write.csv(te1[,encode_features]    , './input/Prav.test.fold1.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 2)
  {
    write.csv(X_build[,encode_features], './input/Prav.build.fold2.csv', row.names=FALSE, quote = FALSE)
    write.csv(X_val[,encode_features]  , './input/Prav.val.fold2.csv', row.names=FALSE, quote = FALSE)
    write.csv(te1[,encode_features]    , './input/Prav.test.fold2.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 3)
  {
    write.csv(X_build[,encode_features], './input/Prav.build.fold3.csv', row.names=FALSE, quote = FALSE)
    write.csv(X_val[,encode_features]  , './input/Prav.val.fold3.csv', row.names=FALSE, quote = FALSE)
    write.csv(te1[,encode_features]    , './input/Prav.test.fold3.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 4)
  {
    write.csv(X_build[,encode_features], './input/Prav.build.fold4.csv', row.names=FALSE, quote = FALSE)
    write.csv(X_val[,encode_features]  , './input/Prav.val.fold4.csv', row.names=FALSE, quote = FALSE)
    write.csv(te1[,encode_features]    , './input/Prav.test.fold4.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 5)
  {
    write.csv(X_build[,encode_features], './input/Prav.build.fold5.csv', row.names=FALSE, quote = FALSE)
    write.csv(X_val[,encode_features]  , './input/Prav.val.fold5.csv', row.names=FALSE, quote = FALSE)
    write.csv(te1[,encode_features]    , './input/Prav.test.fold5.csv', row.names=FALSE, quote = FALSE)
  }
}

item_seq_category_price_mean_encode <- tr %>% 
  group_by( category_name) %>%
  summarise(item_seq_number_category_price_mean_encode = mean(price)
            , item_seq_number_category_target_mean_encode = mean(deal_probability))

region_price_mean_encode <- tr %>% 
  group_by( region) %>%
  summarise(region_price_mean_encode = mean(price)
            , region_target_mean_encode = mean(deal_probability))

city_price_mean_encode <- tr %>% 
  group_by( city) %>%
  summarise(city_price_mean_encode = mean(price)
            , city_target_mean_encode = mean(deal_probability))


tr1 <- left_join(tr, item_seq_category_price_mean_encode, by=c("category_name"))
tr1 <- left_join(tr1, region_price_mean_encode, by=c("region"))
tr1 <- left_join(tr1, city_price_mean_encode, by=c("city"))

te1     <- left_join(te     , item_seq_category_price_mean_encode, by=c("category_name"))
te1     <- left_join(te1, region_price_mean_encode, by=c("region"))
te1     <- left_join(te1, city_price_mean_encode, by=c("city"))


encode_features <- c("item_id","item_seq_number_category_price_mean_encode","item_seq_number_category_target_mean_encode"
                     ,"region_price_mean_encode","region_target_mean_encode"
                     ,"city_price_mean_encode","city_target_mean_encode")

summary(te1$item_seq_number_category_target_mean_encode)

tr1[is.na(tr1)] <- 0
te1[is.na(te1)] <- 0

write.csv(tr1[,encode_features], './input/Prav.encode_train.csv', row.names=FALSE, quote = FALSE)
write.csv(te1[,encode_features], './input/Prav.encode_test.csv', row.names=FALSE, quote = FALSE)

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

head(tr)
#---------------------------
cat("Preprocessing...\n")
tri <- 1:nrow(tr)
y <- tr$deal_probability

tr_te <- tr %>% 
  select(-deal_probability) %>% 
  bind_rows(te) %>% 
  mutate(no_img = is.na(image) %>% as.integer(),
         no_dsc = is.na(description) %>% as.integer(),
         no_p1 = is.na(param_1) %>% as.integer(), 
         no_p2 = is.na(param_2) %>% as.integer(), 
         no_p3 = is.na(param_3) %>% as.integer(),
         titl_len = str_length(title),
         desc_len = str_length(description),
         titl_cap = str_count(title, "[A-Z??-??]"),
         desc_cap = str_count(description, "[A-Z??-??]"),
         titl_pun = str_count(title, "[[:punct:]]"),
         desc_pun = str_count(description, "[[:punct:]]"),
         titl_dig = str_count(title, "[[:digit:]]"),
         desc_dig = str_count(description, "[[:digit:]]"),
         mday = mday(activation_date),
         wday = wday(activation_date),         
         day = day(activation_date)) %>% 
  select( -image, -title, -description, -activation_date) %>% 
  replace_na(list(image_top_1 = -99, #price = -99, 
                  
                  desc_len = 0, desc_cap = 0, desc_pun = 0, desc_dig = 0, titl_pun = 0, titl_cap = 0, titl_dig = 0, titl_len = 0)) %T>% 
  glimpse()

#rm(tr, te); gc()

tr_te <- tr_te %>%
  group_by(item_seq_number, category_name, city, image_top_1) %>%
  mutate(item_seq_number_category_city_image_price_mean = mean(price, na.rm = TRUE)) %>%
  ungroup() %>%
  
  group_by(item_seq_number,category_name,image_top_1) %>%
  mutate(item_seq_number_category_image_price_mean = mean(price, na.rm = TRUE))%>%
  ungroup()%>%
  
  group_by(item_seq_number, city, image_top_1) %>%
  mutate(item_seq_number_city_image_price_mean = mean(price, na.rm = TRUE)) %>%
  ungroup() %>%
  
  group_by(item_seq_number, image_top_1) %>%
  mutate(item_seq_number_image_price_mean = mean(price, na.rm = TRUE)) %>%
  ungroup() %>%
  
  group_by(category_name, city, image_top_1) %>%
  mutate(category_city_image_price_mean = mean(price, na.rm = TRUE)) %>%
  ungroup() %>%
  
  group_by(category_name,image_top_1) %>%
  mutate(category_image_price_mean = mean(price, na.rm = TRUE))%>%
  ungroup()%>%
  
  group_by(city, image_top_1) %>%
  mutate(city_image_price_mean = mean(price, na.rm = TRUE)) %>%
  ungroup() %>%
  
  mutate(price_per_titl_pun = ifelse(titl_pun == 0, 0 , price/titl_pun) ,
          price_per_desc_pun = ifelse(desc_pun == 0, 0 , price/desc_pun)

         
  )


FE03_features <- c("item_id","no_img"                                , "no_dsc"                                        
                   , "no_p1"                                         , "no_p2"                                       ,   "no_p3"                                         
                   , "titl_len"                                      , "desc_len"                                    ,   "titl_cap"                                      
                   , "desc_cap"                                      , "titl_pun"                                    ,   "desc_pun"                                      
                   , "titl_dig"                                      , "desc_dig"                                    ,   "mday"                                          
                   , "wday"                                          , "day"                                         ,   "item_seq_number_category_city_image_price_mean"
                   , "item_seq_number_category_image_price_mean"     , "item_seq_number_city_image_price_mean"       ,   "item_seq_number_image_price_mean"              
                   , "category_city_image_price_mean"                , "category_image_price_mean"                   ,   "city_image_price_mean"                         
                   , "price_per_titl_pun"                            , "price_per_desc_pun" )

Prav_FE03 <- tr_te[,FE03_features]

head(Prav_FE03)


Prav_FE03[is.na(Prav_FE03)] <- -99

train_Prav_FE03 <- Prav_FE03[tri,]
test_Prav_FE03  <- Prav_FE03[-tri,]

summary(train_Prav_FE03)


write.csv(train_Prav_FE03, './input/Prav_train_FE_03.csv', row.names=FALSE, quote = FALSE)
write.csv(test_Prav_FE03, './input/Prav_test_FE_03.csv'  , row.names=FALSE, quote = FALSE)



summary(train_Prav_FE03)




source("./Models/000.setup.R")

# Load Data ---------------------------------------------------------------


aisles      <- fread("./input/aisles.csv")
departments <- fread("./input/departments.csv")
orderp      <- fread("./input/order_products__prior.csv")
ordert      <- fread("./input/order_products__train.csv")
orders      <- fread("./input/orders.csv")
products    <- read_csv("./input/products.csv")

# trainingFolds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")

# Reshape data ------------------------------------------------------------

head(aisles)
aisles$aisle <- as.factor(aisles$aisle)

head(departments)
departments$department <- as.factor(departments$department)

head(orders)
orders$eval_set <- as.factor(orders$eval_set)

head(products)
products$product_name <- as.factor(products$product_name)

products <- products %>% 
  inner_join(aisles) %>% inner_join(departments) %>% 
  select(-aisle_id, -department_id)
rm(aisles, departments)

head(ordert)
ordert$user_id <- orders$user_id[match(ordert$order_id, orders$order_id)]

head(orderp)
orders_products <- orders %>% inner_join(orderp, by = "order_id")

rm(orderp)
gc()

# temp <- filter(orders_products, user_id == 1)
# head(temp)
# temp <- temp %>%
#           arrange(user_id, order_number, product_id) %>%
#           group_by(user_id, product_id) %>%
#             mutate(product_time = row_number())

# Aisle and Department Features: similar to product features
# user product interaction:#purchases, #reorders, #day since last purchase, #order since last purchase etc.
# User aisle and department interaction: similar to product interaction
# User time interaction: user preferred day of week, user preferred time of day, similar features for products and aisles

# Products ----------------------------------------------------------------
# Product Features: #users, #orders, order frequency, reorder rate, recency, mean add_to_cart_order, etc.

head(orders_products)
head(prd)

orders_products %>%
  filter(product_id == 196) %>%
  group_by(product_id,order_dow) %>%
  summarise(count = n())
  
prd <- orders_products %>%
  # filter(product_id == 196) %>%
  arrange(user_id, order_number, product_id) %>%
  group_by(user_id, product_id) %>%
  mutate(product_time = row_number()) %>%
  ungroup() %>%
  group_by(product_id) %>%
  summarise(
    prod_orders = n(),
    prod_reorders = sum(reordered),
    prod_first_orders = sum(product_time == 1),
    prod_second_orders = sum(product_time >= 2),
    prod_user_count = n_distinct(user_id),
    prod_mean_add_to_cart_order = mean(add_to_cart_order),
    prod_max_dow  =names(which.max(table(order_dow))),
    prod_min_dow  =names(which.min(table(order_dow))),
    prod_max_hour =names(which.max(table(order_hour_of_day))),
    prod_min_hour =names(which.min(table(order_hour_of_day)))
  )

prd$prod_reorder_probability <- prd$prod_second_orders / prd$prod_first_orders
prd$prod_reorder_times       <- 1 + prd$prod_reorders / prd$prod_first_orders
prd$prod_reorder_ratio       <- prd$prod_reorders / prd$prod_orders

prd <- prd %>% select(-prod_reorders, -prod_first_orders, -prod_second_orders)


rm(products)
gc()

# Users -------------------------------------------------------------------
# User Features: #Products purchased, #Orders made, frequency and recency of orders, #Aisle purchased from, #Department purchased from, frequency and recency of reorders, tenure, mean order size, etc.
head(orders)
users <- orders %>%
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  summarise(
    user_orders = max(order_number),                                      #Orders made
    user_period = sum(days_since_prior_order, na.rm = T),                 # not sure of this feature
    user_min_days_since_prior = min(days_since_prior_order, na.rm = T),
    user_max_days_since_prior = max(days_since_prior_order, na.rm = T),
    user_mean_days_since_prior = mean(days_since_prior_order, na.rm = T),
    user_min_dow_since_prior = names(which.min(table(order_dow))),
    user_max_dow_since_prior = names(which.max(table(order_dow))),
    user_min_hour_since_prior = names(which.min(table(order_hour_of_day))),
    user_max_hour_since_prior = names(which.max(table(order_hour_of_day))),
    user_mean_hour_since_prior = mean(order_hour_of_day, na.rm = T),
    user_median_hour_since_prior = median(order_hour_of_day, na.rm = T)
  )
head(orders_products)


us <- orders_products %>%
  group_by(user_id,product_id) %>%
  mutate(  user_product_min_dow  = min(order_dow, na.rm = T)
         , user_product_max_dow = max(order_dow, na.rm = T)
         , user_product_mean_dow = mean(order_dow, na.rm = T)
         , user_product_min_hour  = min(order_hour_of_day, na.rm = T)
         , user_product_max_hour = max(order_hour_of_day, na.rm = T)
         , user_product_mean_hour = mean(order_hour_of_day, na.rm = T)
         , user_product_min_prior_days  = min(days_since_prior_order, na.rm = T)
         , user_product_max_prior_days = max(days_since_prior_order, na.rm = T)
         , user_product_mean_prior_days = mean(days_since_prior_order, na.rm = T)
         
         , user_product_min_add_to_cart_order  = min(add_to_cart_order, na.rm = T)
         , user_product_max_add_to_cart_order = max(add_to_cart_order, na.rm = T)
         , user_product_mean_add_to_cart_order = mean(add_to_cart_order, na.rm = T)
         
         ) %>%
  group_by(user_id) %>%
  summarise(
    user_total_products = n(),
    user_reorder_ratio = sum(reordered == 1) / sum(order_number > 1),
    user_distinct_products = n_distinct(product_id),
    user_product_min_dow_min = min(user_product_min_dow, na.rm = T),
    user_product_max_dow_max = max(user_product_max_dow, na.rm = T),
    user_product_mean_dow_mean = mean(user_product_mean_dow, na.rm = T),
    user_product_min_hour_min = min(user_product_min_hour, na.rm = T) ,
    user_product_max_hour_max = max(user_product_max_hour, na.rm = T) ,
    user_product_mean_hour_mean = mean(user_product_mean_hour, na.rm = T),
    user_product_min_prior_days_min = min(user_product_min_prior_days, na.rm = T) , 
    user_product_max_prior_days_max = max(user_product_max_prior_days, na.rm = T) , 
    user_product_mean_prior_days_mean = mean(user_product_mean_prior_days, na.rm = T),
    user_product_min_add_to_cart_order_min = min(user_product_min_add_to_cart_order, na.rm = T), 
    user_product_max_add_to_cart_order_max = max(user_product_max_add_to_cart_order, na.rm = T), 
    user_product_mean_add_to_cart_order_mean = mean(user_product_mean_add_to_cart_order, na.rm = T)
    
  )

users <- users %>% inner_join(us)
users$user_average_basket <- users$user_total_products / users$user_orders

us <- orders %>%
  filter(eval_set != "prior") %>%
  select(user_id, order_id, eval_set,
         time_since_last_order = days_since_prior_order)

users <- users %>% inner_join(us)

rm(us)
gc()


# Database ----------------------------------------------------------------
head(orders_products)
data <- orders_products %>%
  group_by(user_id, product_id) %>% 
  summarise(
    up_orders                             = n()
    , up_first_order                      = min(order_number)
    , up_last_order                       = max(order_number)
    , up_average_cart_position            = mean(add_to_cart_order)
    , user_product_min_dow                = min(order_dow, na.rm = T)
    , user_product_max_dow                = max(order_dow, na.rm = T)
    , user_product_mean_dow               = mean(order_dow, na.rm = T)
    , user_product_min_hour               = min(order_hour_of_day, na.rm = T)
    , user_product_max_hour               = max(order_hour_of_day, na.rm = T)
    , user_product_mean_hour              = mean(order_hour_of_day, na.rm = T)
    , user_product_min_prior_days         = min(days_since_prior_order, na.rm = T)
    , user_product_max_prior_days         = max(days_since_prior_order, na.rm = T)
    , user_product_mean_prior_days        = mean(days_since_prior_order, na.rm = T)
    
    , user_product_min_add_to_cart_order  = min(add_to_cart_order, na.rm = T)
    , user_product_max_add_to_cart_order  = max(add_to_cart_order, na.rm = T)
    , user_product_mean_add_to_cart_order = mean(add_to_cart_order, na.rm = T)
    
    , user_product_maxRank_dow  =names(which.max(table(order_dow)))
    , user_product_minRank_dow  =names(which.min(table(order_dow)))
    , user_product_maxRank_hour =names(which.max(table(order_hour_of_day)))
    , user_product_minRank_hour =names(which.min(table(order_hour_of_day)))
    
    
    )

rm(orders_products, orders)

data <- data %>% 
  inner_join(prd, by = "product_id") %>%
  inner_join(users, by = "user_id")

data$up_order_rate <- data$up_orders / data$user_orders
data$up_orders_since_last_order <- data$user_orders - data$up_last_order
data$up_order_rate_since_first_order <- data$up_orders / (data$user_orders - data$up_first_order + 1)

data <- data %>% 
  left_join(ordert %>% select(user_id, product_id, reordered), 
            by = c("user_id", "product_id"))

rm(ordert, prd, users)
gc()


# Train / Test datasets ---------------------------------------------------
train <- as.data.frame(data[data$eval_set == "train",])
# train$eval_set <- NULL
# train$user_id <- NULL
# train$product_id <- NULL
# train$order_id <- NULL
train$reordered[is.na(train$reordered)] <- 0

test <- as.data.frame(data[data$eval_set == "test",])
# test$eval_set <- NULL
# test$user_id <- NULL
# test$reordered <- NULL

rm(data)
gc()

head(train)
head(test)

product_embeds <- read.csv("./input/product_vector_features.csv")

user_product_streak <- read.csv("./input/order_streaks.csv")


aisles      <- fread("./input/aisles.csv")
departments <- fread("./input/departments.csv")
products    <- read_csv("./input/products.csv")

head(aisles)
aisles$aisle <- as.factor(aisles$aisle)

head(departments)
departments$department <- as.factor(departments$department)


head(products)
products$product_name <- as.factor(products$product_name)

products <- products %>% 
  inner_join(aisles) %>% inner_join(departments) %>% 
  select(-aisle, -department, -product_name)
rm(aisles, departments)

head(products)


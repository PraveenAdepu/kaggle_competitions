
source("./Models/000.setup.R")

# Load Data ---------------------------------------------------------------


aisles.source      <- fread("./input/aisles.csv")
departments.source <- fread("./input/departments.csv")
orderp      <- fread("./input/order_products__prior.csv")
ordert      <- fread("./input/order_products__train.csv")
orders      <- fread("./input/orders.csv")
products.source    <- read_csv("./input/products.csv")

# trainingFolds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")

# Reshape data ------------------------------------------------------------

head(aisles.source)
aisles.source$aisle <- as.factor(aisles.source$aisle)

head(departments.source)
departments.source$department <- as.factor(departments.source$department)

head(orders)
orders$eval_set <- as.factor(orders$eval_set)

head(products.source)
products.source$product_name <- as.factor(products.source$product_name)

products <- products.source %>% 
  inner_join(aisles.source) %>% inner_join(departments.source) %>% 
  select(-aisle, -department, -product_name)
rm(aisles.source, departments.source)

head(ordert)
ordert$user_id <- orders$user_id[match(ordert$order_id, orders$order_id)]

head(orderp)
orders_products <- orders %>% inner_join(orderp, by = "order_id")
orders_products <- orders_products %>% inner_join(products, by = "product_id")
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


aisles <- orders_products %>%
  # filter(product_id == 196) %>%
  arrange(user_id, order_number, aisle_id) %>%
  group_by(user_id, aisle_id) %>%
  mutate(aisle_time = row_number()) %>%
  ungroup() %>%
  group_by(aisle_id) %>%
  summarise(
    aisle_orders = n(),
    aisle_reorders = sum(reordered),
    aisle_first_orders = sum(aisle_time == 1),
    aisle_second_orders = sum(aisle_time >= 2),
    aisle_user_count = n_distinct(user_id),
    aisle_mean_add_to_cart_order = mean(add_to_cart_order),
    aisle_max_dow  =names(which.max(table(order_dow))),
    aisle_min_dow  =names(which.min(table(order_dow))),
    aisle_max_hour =names(which.max(table(order_hour_of_day))),
    aisle_min_hour =names(which.min(table(order_hour_of_day)))
  )

aisles$aisle_reorder_probability <- aisles$aisle_second_orders / aisles$aisle_first_orders
aisles$aisle_reorder_times       <- 1 + aisles$aisle_reorders / aisles$aisle_first_orders
aisles$aisle_reorder_ratio       <- aisles$aisle_reorders / aisles$aisle_orders

aisles <- aisles %>% select(-aisle_reorders, -aisle_first_orders, -aisle_second_orders)

departments <- orders_products %>%
  # filter(product_id == 196) %>%
  arrange(user_id, order_number, department_id) %>%
  group_by(user_id, department_id) %>%
  mutate(department_time = row_number()) %>%
  ungroup() %>%
  group_by(department_id) %>%
  summarise(
    department_orders = n(),
    department_reorders = sum(reordered),
    department_first_orders = sum(department_time == 1),
    department_second_orders = sum(department_time >= 2),
    department_user_count = n_distinct(user_id),
    department_mean_add_to_cart_order = mean(add_to_cart_order),
    department_max_dow  =names(which.max(table(order_dow))),
    department_min_dow  =names(which.min(table(order_dow))),
    department_max_hour =names(which.max(table(order_hour_of_day))),
    department_min_hour =names(which.min(table(order_hour_of_day)))
  )

departments$department_reorder_probability <- departments$department_second_orders / departments$department_first_orders
departments$department_reorder_times       <- 1 + departments$department_reorders / departments$department_first_orders
departments$department_reorder_ratio       <- departments$department_reorders / departments$department_orders

departments <- departments %>% select(-department_reorders, -department_first_orders, -department_second_orders)

#rm(products)
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
    user_mean_dow_since_prior = mean(order_dow, na.rm = T),
    user_median_dow_since_prior = median(order_dow, na.rm = T),
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
data_user_product <- orders_products %>%
  group_by(user_id, product_id) %>% 
  summarise(
      user_product_up_orders               = n()
    , user_product_up_first_order          = min(order_number)
    , user_product_up_last_order           = max(order_number)
    , user_product_up_average_cart_position= mean(add_to_cart_order)
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

data_user_aisle <- orders_products %>%
  group_by(user_id, aisle_id) %>% 
  summarise(
      user_aisle_up_orders                  = n()
    , user_aisle_up_first_order            = min(order_number)
    , user_aisle_up_last_order             = max(order_number)
    , user_aisle_up_average_cart_position  = mean(add_to_cart_order)
    , user_aisle_min_dow                = min(order_dow, na.rm = T)
    , user_aisle_max_dow                = max(order_dow, na.rm = T)
    , user_aisle_mean_dow               = mean(order_dow, na.rm = T)
    , user_aisle_min_hour               = min(order_hour_of_day, na.rm = T)
    , user_aisle_max_hour               = max(order_hour_of_day, na.rm = T)
    , user_aisle_mean_hour              = mean(order_hour_of_day, na.rm = T)
    , user_aisle_min_prior_days         = min(days_since_prior_order, na.rm = T)
    , user_aisle_max_prior_days         = max(days_since_prior_order, na.rm = T)
    , user_aisle_mean_prior_days        = mean(days_since_prior_order, na.rm = T)
    , user_aisle_min_add_to_cart_order  = min(add_to_cart_order, na.rm = T)
    , user_aisle_max_add_to_cart_order  = max(add_to_cart_order, na.rm = T)
    , user_aisle_mean_add_to_cart_order = mean(add_to_cart_order, na.rm = T)
    , user_aisle_maxRank_dow  =names(which.max(table(order_dow)))
    , user_aisle_minRank_dow  =names(which.min(table(order_dow)))
    , user_aisle_maxRank_hour =names(which.max(table(order_hour_of_day)))
    , user_aisle_minRank_hour =names(which.min(table(order_hour_of_day)))
    
    
  )


data_user_department <- orders_products %>%
  group_by(user_id, department_id) %>% 
  summarise(
      user_department_up_orders                = n()
    , user_department_up_first_order           = min(order_number)
    , user_department_up_last_order            = max(order_number)
    , user_department_up_average_cart_position = mean(add_to_cart_order)
    , user_department_min_dow                = min(order_dow, na.rm = T)
    , user_department_max_dow                = max(order_dow, na.rm = T)
    , user_department_mean_dow               = mean(order_dow, na.rm = T)
    , user_department_min_hour               = min(order_hour_of_day, na.rm = T)
    , user_department_max_hour               = max(order_hour_of_day, na.rm = T)
    , user_department_mean_hour              = mean(order_hour_of_day, na.rm = T)
    , user_department_min_prior_days         = min(days_since_prior_order, na.rm = T)
    , user_department_max_prior_days         = max(days_since_prior_order, na.rm = T)
    , user_department_mean_prior_days        = mean(days_since_prior_order, na.rm = T)
    , user_department_min_add_to_cart_order  = min(add_to_cart_order, na.rm = T)
    , user_department_max_add_to_cart_order  = max(add_to_cart_order, na.rm = T)
    , user_department_mean_add_to_cart_order = mean(add_to_cart_order, na.rm = T)
    , user_department_maxRank_dow  =names(which.max(table(order_dow)))
    , user_department_minRank_dow  =names(which.min(table(order_dow)))
    , user_department_maxRank_hour =names(which.max(table(order_hour_of_day)))
    , user_department_minRank_hour =names(which.min(table(order_hour_of_day)))
    
    
  )

rm(orders_products, orders)

names(data_user_product)

head(products)


data <- data_user_product %>%
            inner_join(products, by = "product_id") %>%
            inner_join(data_user_aisle, by = c("user_id","aisle_id")) %>%
            inner_join(data_user_department, by = c("user_id","department_id")) %>%
            inner_join(prd, by = "product_id") %>%
            inner_join(users, by = "user_id")

# data <- data %>% 
#   inner_join(prd, by = "product_id") %>%
#   inner_join(users, by = "user_id")

names(data)

data$user_product_up_order_rate                   <- data$user_product_up_orders   / data$user_orders
data$user_product_up_orders_since_last_order      <- data$user_orders - data$user_product_up_last_order
data$user_product_up_order_rate_since_first_order <- data$user_product_up_orders   / (data$user_orders - data$user_product_up_first_order + 1)

data$user_aisle_up_order_rate                   <- data$user_aisle_up_orders   / data$user_orders
data$user_aisle_up_orders_since_last_order      <- data$user_orders - data$user_aisle_up_last_order
data$user_aisle_up_order_rate_since_first_order <- data$user_aisle_up_orders   / (data$user_orders - data$user_aisle_up_first_order + 1)

data$user_department_up_order_rate                   <- data$user_department_up_orders   / data$user_orders
data$user_department_up_orders_since_last_order      <- data$user_orders - data$user_department_up_last_order
data$user_department_up_order_rate_since_first_order <- data$user_department_up_orders   / (data$user_orders - data$user_department_up_first_order + 1)


data <- data %>% 
  left_join(ordert %>% select(user_id, product_id, reordered), 
            by = c("user_id", "product_id"))

rm(ordert, prd, users)
gc()

rm(aisles, departments,data_user_aisle, data_user_product, data_user_department)
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




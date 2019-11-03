
rm(list=ls())

setwd("C:/Users/SriPrav/Documents/R/30Caesars")
root_directory = "C:/Users/SriPrav/Documents/R/30Caesars"

source("./Models/loadPackages.R")
source("./Models/helperFunctions.R")
source("./Models/Data.visualization.functions.R")

# 
 # train.source <- read_csv("./input/train_v2.csv", col_types = cols(customer_id = col_character())) # 5,203,955
 # test.source  <- read_csv("./input/test_v2.csv", col_types = cols(customer_id = col_character()))  # 1,378,521
# 
# head(train.source)
# 
# saveRDS(train.source,"./input/train.rds")
# saveRDS(test.source,"./input/test.rds")


train.source <- readRDS("./input/train.rds") # 5,203,955
test.source  <- readRDS("./input/test.rds")  # 1,378,521

head(train.source)

unique(train.source$date)

# "2017-04-01" "2017-03-01" "2017-02-01" "2017-01-01" "2016-12-01" "2016-11-01" "2016-10-01" "2016-09-01" "2016-08-01" "2016-07-01" "2016-06-01"
# "2016-05-01"
unique(test.source$date)
# "2017-05-01" "2017-06-01" "2017-07-01" "2017-08-01"



train.source$date <- as.Date(train.source$date, format="%m%d%Y")
test.source$date  <- as.Date(test.source$date, format="%m%d%Y")
sapply(train.source, class)

# generic.data.visualisations("discrete.bar", train.source, "target")
# generic.data.visualisations("discrete.bar", train.source, "market")

# plot04 <-   ggplot(train.source, aes(target, colour = as.factor(target))) +            
#             geom_bar() + facet_wrap(~market)
# print(plot04)

# train.source$f_411 <- as.integer(train.source$f_41)
# summary(train.source$f_411)
# 
# test.source$f_411 <- as.integer(test.source$f_41)
# summary(test.source$f_411)


train.source$id <- -100

test.source$target <- -1

all.source <- rbind(train.source, test.source)

# names(all.source)
# 
# all.source <- all.source %>%
#                 group_by(f_41) %>%
#                 mutate(f41_count = n())
# 
# length(unique(substr(all.source$f_41, start = 1, stop = 1)))
# length(unique(all.source$f_41))
# 
# all.source$f41_first <- substr(all.source$f_41, start = 1, stop = 1)
# 
# 
# # cust.test <- all.source %>%
# #              filter(customer_id == 133901138589)
# # 
# # cust.test1 <- cust.test %>%
# #               arrange(customer_id, date, market) %>%
# #               group_by(customer_id,market) %>%
# #               mutate(cust_market_lag_date = lag(date)
# #                      
# #                      )
# # 
# # cust.test1 <- cust.test1 %>%
# #                 group_by(customer_id,market) %>%
# #                 mutate(cust_market_lagMonths=round(ifelse(is.na(cust_market_lag_date),0,difftime(date, cust_market_lag_date, units = "days"))/30))
# # 
# # cust.test1 <- cust.test1 %>%
# #                 group_by(customer_id,market) %>%
# #                 mutate(cust_market_lagMonths_conseq = ifelse(cust_market_lagMonths==1,1,0))
# # 
# # cust.test1 <- cust.test1 %>%
# #                group_by(customer_id,market,cust_market_lagMonths_conseq) %>%
# #                mutate(conseq_visit = cumsum(cust_market_lagMonths_conseq))
# # 
# # cust.test1 <- cust.test1 %>%
# #               group_by(customer_id,market,cust_market_lagMonths_conseq) %>%
# #               mutate(conseq_visit_eachtrip = ifelse(conseq_visit <= 1,conseq_visit, conseq_visit-1))
# 
# 
# all.source <- all.source %>%
#   arrange(customer_id, date, market) %>%
#   group_by(customer_id,market) %>%
#   mutate(cust_market_lag_date = lag(date), lag_f2 = lag(f_2)
#          
#   )
# 
# all.source <- all.source %>%
#   group_by(customer_id,market) %>%
#   mutate(cust_market_lagMonths=round(ifelse(is.na(cust_market_lag_date),0,difftime(date, cust_market_lag_date, units = "days"))/30))
# 
# all.source <- all.source %>%
#   group_by(customer_id,market) %>%
#   mutate(cust_market_lagMonths_conseq = ifelse(cust_market_lagMonths==1,1,0))
# 
# all.source <- all.source %>%
#   group_by(customer_id,market,cust_market_lagMonths_conseq) %>%
#   mutate(conseq_visit = cumsum(cust_market_lagMonths_conseq))
# 
# all.source <- all.source %>%
#   group_by(customer_id,market,cust_market_lagMonths_conseq) %>%
#   mutate(conseq_visit_eachtrip = ifelse(conseq_visit <= 1,conseq_visit, conseq_visit-1))
# 
# all.source <- all.source %>%
#   arrange(customer_id, date, market) %>%
#   group_by(customer_id,market) %>%
#   mutate( lag_f2 = lag(f_2)
#          
#   )
# 
# all.source <- all.source %>%
#                group_by(customer_id,market) %>%
#                mutate(cust_market_f2_lagdiff= ifelse(is.na(lag_f2),0,f_2-lag_f2))
# 
# 
# head(all.source$row_mean)
# system.time(
# all.source <- all.source %>%
#                 rowwise() %>%
#                 mutate(row_mean = mean(c(f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10,f_11,f_12,f_13,f_14,f_15,f_20,f_27,f_29,f_30,f_31,f_32,f_34,
#                                          f_35,f_36,f_37,f_38,f_39))
#                        
#                        )
#            )
# 
# system.time(
#   all.source <- all.source %>%
#     rowwise() %>%
#     mutate(row_max = max(c(f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10,f_11,f_12,f_13,f_14,f_15,f_20,f_27,f_29,f_30,f_31,f_32,f_34,
#                          f_35,f_36,f_37,f_38,f_39)),
#            row_min = min(c(f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10,f_11,f_12,f_13,f_14,f_15,f_20,f_27,f_29,f_30,f_31,f_32,f_34,
#                          f_35,f_36,f_37,f_38,f_39)),
#            row_sd = sd(c(f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10,f_11,f_12,f_13,f_14,f_15,f_20,f_27,f_29,f_30,f_31,f_32,f_34,
#                        f_35,f_36,f_37,f_38,f_39))
#            
#            )
# )
# 
# names(all.source)
# summary(all.source$row_sd)


train.source <- all.source %>%
                  filter(target != -1)

test.source <- all.source %>%
                  filter(target == -1)


# unique(train.source$f_0)
# unique(train.source$f_2)
# unique(train.source$f_5)
# 
# summary(train.source$f_13)
# summary(train.source$f_15)
# summary(train.source$f_30)
# cor.columns <- c("f_13","f_15","f_30","target")
# cor(train.source[,cor.columns])
# 
# two.features.concatenation <- function(dataset, feature1, feature2)
# {
#   
#   new.feature <- paste0(feature1,"_",feature2)
#   dataset[[new.feature]] <- paste0(dataset[[feature1]],dataset[[feature2]])
#   
#   return(dataset)
# }
# 
# train.source <- two.features.concatenation(train.source,"f_16","f_17")
# # train.source <- two.features.concatenation(train.source,"f_16","f_19")
# # train.source <- two.features.concatenation(train.source,"f_16","f_21")
# # train.source <- two.features.concatenation(train.source,"f_16","f_23")
# # train.source <- two.features.concatenation(train.source,"f_16","f_24")
# # train.source <- two.features.concatenation(train.source,"f_16","f_25")
# # train.source <- two.features.concatenation(train.source,"f_16","f_26")
# # train.source <- two.features.concatenation(train.source,"f_16","f_40")
# 
# 
# test.source <- two.features.concatenation(test.source,"f_16","f_17")
# # test.source <- two.features.concatenation(test.source,"f_16","f_19")
# # test.source <- two.features.concatenation(test.source,"f_16","f_21")
# # test.source <- two.features.concatenation(test.source,"f_16","f_23")
# # test.source <- two.features.concatenation(test.source,"f_16","f_24")
# # test.source <- two.features.concatenation(test.source,"f_16","f_25")
# # test.source <- two.features.concatenation(test.source,"f_16","f_26")
# # test.source <- two.features.concatenation(test.source,"f_16","f_40")
# 
# train.source <- two.features.concatenation(train.source,"f_17","f_19")
# train.source <- two.features.concatenation(train.source,"f_19","f_21")
# train.source <- two.features.concatenation(train.source,"f_21","f_23")
# train.source <- two.features.concatenation(train.source,"f_23","f_24")
# train.source <- two.features.concatenation(train.source,"f_24","f_25")
# train.source <- two.features.concatenation(train.source,"f_25","f_26")
# train.source <- two.features.concatenation(train.source,"f_26","f_40")
# 
# test.source <- two.features.concatenation(test.source,"f_17","f_19")
# test.source <- two.features.concatenation(test.source,"f_19","f_21")
# test.source <- two.features.concatenation(test.source,"f_21","f_23")
# test.source <- two.features.concatenation(test.source,"f_23","f_24")
# test.source <- two.features.concatenation(test.source,"f_24","f_25")
# test.source <- two.features.concatenation(test.source,"f_25","f_26")
# test.source <- two.features.concatenation(test.source,"f_26","f_40")
# 
# train.source <- two.features.concatenation(train.source,"market","f_41")
# test.source <- two.features.concatenation(test.source,"market","f_41")

# Column Name	Data Type
# customer_id	int
# market	char
# date	date
# f_0	int
# f_2	int
# f_3	float
# f_4	float
# f_5	int
# f_6	float
# f_8	float
# f_10	float
# f_11	float
# f_12	float
# f_13	float
# f_14	float
# f_15	float
# f_17	float
# f_18	float
# f_21	float
# f_22	float
# f_25	float
# f_26	float
# f_27	float
# f_28	float
# f_30	float
# f_32	float
# f_34	float
# f_35	float
# f_36	float
# f_37	float
# f_38	float
# f_39	float
# f_40	float
# f_41	float
# target	int

# f_1	char
# f_7	char  # Check on this data type
# f_9	char
# f_16	char
# f_19	char
# f_20	char
# f_23	char
# f_24	char
# f_29	char
# f_31	char
# f_33	char

char.feature.names <- c("market","f_1","f_9","f_16","f_19","f_20","f_23","f_24","f_29","f_31","f_33")
                        # ,"f_16_f_17"
                        # # ,"f_16_f_19"
                        # # ,"f_16_f_21"
                        # # ,"f_16_f_23"
                        # # ,"f_16_f_24"
                        # # ,"f_16_f_25"
                        # # ,"f_16_f_26"
                        # # ,"f_16_f_40"
                        # ,"f_17_f_19"
                        # ,"f_19_f_21"
                        # ,"f_21_f_23"
                        # ,"f_23_f_24"
                        # ,"f_24_f_25"
                        # ,"f_25_f_26"
                        # ,"f_26_f_40"
                        # ,"market_f_41"
                        # )
# 
# head(train.source[,char.feature.names])
for (f in char.feature.names) {

  cat("char feature : ", f ,"\n")
  cat("unique lenght : ", length(unique(train.source[[f]])),"\n")

}

for (f in char.feature.names) {
  
  cat("char feature : ", f ,"\n")
  cat("unique lenght : ", length(unique(test.source[[f]])),"\n")
  
}


cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in char.feature.names) {
  if (class(train.source[[f]])=="character") {
    cat("char feature : ", f ,"\n")
    levels <- unique(c(train.source[[f]], test.source[[f]]))
    train.source[[f]] <- as.integer(factor(train.source[[f]], levels=levels))
    test.source[[f]]  <- as.integer(factor(test.source[[f]],  levels=levels))
  }
}

# head(trainingSet$f_4)

# cat("train date range : \n")
# unique(train.source$date)
# 
# train.source$f2_4 <- ifelse(train.source$f_4 == 0, -1, train.source$f_2/train.source$f_4)
# test.source$f2_4 <- ifelse(test.source$f_4 == 0, -1, test.source$f_2/test.source$f_4)
# 
# names(train.source)
train.source[is.na(train.source)] <- -1
test.source[is.na(test.source)]   <- -1


X_build <- train.source %>%
                 filter(date < "2017-03-01")

X_val <- train.source %>%
              filter(date >= "2017-03-01")

# all.source %>% group_by(date) %>% summarise(count =n())

#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("market_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("customer_id","market")]


system.time({
  setkey(X_build, customer_id, market)
  X_build_last <- X_build[unique(X_build)[, key(X_build), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","market","market_target_lag1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id","market"))
#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("market_target_lead", 1L) := shift(target, 1L, type = "lead"), by = c("customer_id","market")]


system.time({
  setkey(X_build, customer_id, market)
  X_build_last <- X_build[complete.cases(X_build[ , market_target_lead1]),]
  X_build_last <- X_build_last[unique(X_build_last)[, key(X_build_last), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","market","market_target_lead1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id","market"))
##################################################################################################################

#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("market_target_median", 1L) := median(target), by = c("market")]


system.time({
  setkey(X_build, market)
  X_build_last <- X_build[unique(X_build)[, key(X_build), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("market","market_target_median1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("market"))
#################################################################################################################
##################################################################################################################

#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("customer_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("customer_id")]


system.time({
  setkey(X_build, customer_id)
  X_build_last <- X_build[unique(X_build)[, key(X_build), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","customer_target_lag1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id"))
#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("customer_target_lead", 1L) := shift(target, 1L, type = "lead"), by = c("customer_id")]


system.time({
  setkey(X_build, customer_id)
  X_build_last <- X_build[complete.cases(X_build[ , customer_target_lead1]),]
  X_build_last <- X_build_last[unique(X_build_last)[, key(X_build_last), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","customer_target_lead1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id"))
##################################################################################################################

#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("customer_target_median", 1L) := median(target), by = c("customer_id")]


system.time({
  setkey(X_build, customer_id)
  X_build_last <- X_build[unique(X_build)[, key(X_build), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","customer_target_median1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id"))
#################################################################################################################
# 
# #################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)


X_build$customer_target_lag_target_diff_flag <-   ifelse(X_build$customer_target_lag1 <= X_build$target, 1,-1)

X_build$customer_target_lag_target_diff_flag[is.na(X_build$customer_target_lag_target_diff_flag)] <- 0
X_build[, roll := cumsum(customer_target_lag_target_diff_flag), by = c("customer_id")]

# Rolling sum of last 4 points
# DT[, cum.sum := Reduce(`+`, shift(val, 0:3)), by=id]

system.time({
  setkey(X_build, customer_id)
  X_build_last <- X_build[unique(X_build)[, key(X_build), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","roll")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id"))
#################################################################################################################


##################################################################################################################

#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("f33_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("customer_id","f_33")]



system.time({
  setkey(X_build, customer_id, f_33)
  X_build_last <- X_build[unique(X_build)[, key(X_build), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","f_33","f33_target_lag1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id","f_33"))
#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("f33_target_lead", 1L) := shift(target, 1L, type = "lead"), by = c("customer_id","f_33")]


system.time({
  setkey(X_build, customer_id, f_33)
  X_build_last <- X_build[complete.cases(X_build[ , f33_target_lead1]),]
  X_build_last <- X_build_last[unique(X_build_last)[, key(X_build_last), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","f_33","f33_target_lead1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id","f_33"))
##################################################################################################################
##################################################################################################################
#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("f23_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("customer_id","f_23")]



system.time({
  setkey(X_build, customer_id, f_23)
  X_build_last <- X_build[unique(X_build)[, key(X_build), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","f_23","f23_target_lag1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id","f_23"))
#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("f23_target_lead", 1L) := shift(target, 1L, type = "lead"), by = c("customer_id","f_23")]


system.time({
  setkey(X_build, customer_id, f_23)
  X_build_last <- X_build[complete.cases(X_build[ , f23_target_lead1]),]
  X_build_last <- X_build_last[unique(X_build_last)[, key(X_build_last), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","f_23","f23_target_lead1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id","f_23"))
##################################################################################################################
##################################################################################################################


#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("market_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("customer_id","market")]


system.time({
  setkey(train.source, customer_id, market)
  train.source_last <- train.source[unique(train.source)[, key(train.source), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","market","market_target_lag1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id","market"))
#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("market_target_lead", 1L) := shift(target, 1L, type = "lead"), by = c("customer_id","market")]


system.time({
  setkey(train.source, customer_id, market)
  train.source_last <- train.source[complete.cases(train.source[ , market_target_lead1]),]
  train.source_last <- train.source_last[unique(train.source_last)[, key(train.source_last), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","market","market_target_lead1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id","market"))
##################################################################################################################

#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("market_target_median", 1L) := median(target), by = c("market")]


system.time({
  setkey(train.source, market)
  train.source_last <- train.source[unique(train.source)[, key(train.source), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("market","market_target_median1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("market"))
#################################################################################################################
##################################################################################################################

#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("customer_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("customer_id")]


system.time({
  setkey(train.source, customer_id)
  train.source_last <- train.source[unique(train.source)[, key(train.source), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","customer_target_lag1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id"))
#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("customer_target_lead", 1L) := shift(target, 1L, type = "lead"), by = c("customer_id")]


system.time({
  setkey(train.source, customer_id)
  train.source_last <- train.source[complete.cases(train.source[ , customer_target_lead1]),]
  train.source_last <- train.source_last[unique(train.source_last)[, key(train.source_last), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","customer_target_lead1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id"))
##################################################################################################################

#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("customer_target_median", 1L) := median(target), by = c("customer_id")]


system.time({
  setkey(train.source, customer_id)
  train.source_last <- train.source[unique(train.source)[, key(train.source), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","customer_target_median1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id"))
#################################################################################################################
# 
# #################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)


train.source$customer_target_lag_target_diff_flag <-   ifelse(train.source$customer_target_lag1 <= train.source$target, 1,-1)

train.source$customer_target_lag_target_diff_flag[is.na(train.source$customer_target_lag_target_diff_flag)] <- 0
train.source[, roll := cumsum(customer_target_lag_target_diff_flag), by = c("customer_id")]

# Rolling sum of last 4 points
# DT[, cum.sum := Reduce(`+`, shift(val, 0:3)), by=id]

system.time({
  setkey(train.source, customer_id)
  train.source_last <- train.source[unique(train.source)[, key(train.source), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","roll")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id"))
#################################################################################################################


##################################################################################################################

#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("f33_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("customer_id","f_33")]



system.time({
  setkey(train.source, customer_id, f_33)
  train.source_last <- train.source[unique(train.source)[, key(train.source), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","f_33","f33_target_lag1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id","f_33"))
#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("f33_target_lead", 1L) := shift(target, 1L, type = "lead"), by = c("customer_id","f_33")]


system.time({
  setkey(train.source, customer_id, f_33)
  train.source_last <- train.source[complete.cases(train.source[ , f33_target_lead1]),]
  train.source_last <- train.source_last[unique(train.source_last)[, key(train.source_last), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","f_33","f33_target_lead1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id","f_33"))
##################################################################################################################
##################################################################################################################
#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("f23_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("customer_id","f_23")]



system.time({
  setkey(train.source, customer_id, f_23)
  train.source_last <- train.source[unique(train.source)[, key(train.source), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","f_23","f23_target_lag1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id","f_23"))
#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("f23_target_lead", 1L) := shift(target, 1L, type = "lead"), by = c("customer_id","f_23")]


system.time({
  setkey(train.source, customer_id, f_23)
  train.source_last <- train.source[complete.cases(train.source[ , f23_target_lead1]),]
  train.source_last <- train.source_last[unique(train.source_last)[, key(train.source_last), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","f_23","f23_target_lead1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id","f_23"))
##################################################################################################################
##################################################################################################################

# cat("X_build date range : \n")
# unique(X_build$date)
# 
# cat("X_val date range : \n")
# unique(X_val$date)
# 
# cat("test date range : \n")
# unique(test.source$date)
# 
trainingSet <- train.source
testingSet  <- test.source
# 
# #rm(train.source,test.source); gc()
# 
# sapply(trainingSet, class)
# 
# head(all.source)
# 
# head(train.source)
# 
# customer_test <- all.source %>% filter(customer_id == "219009981584")


rm(list=ls())

setwd("C:/Users/SriPrav/Documents/R/30Caesars")
root_directory = "C:/Users/SriPrav/Documents/R/30Caesars"

source("./Models/loadPackages.R")
source("./Models/helperFunctions.R")
source("./Models/Data.visualization.functions.R")
require(xts)
require(stringr)
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


train.source$id <- -100

test.source$target <- -1

all.source <- rbind(train.source, test.source)

test <- all.source %>% filter(customer_id =="219009981584")

#write.csv(head(all.source,1000), "./input/sample.csv", row.names = FALSE)



char.feature.names <- c("market","f_1","f_9","f_16","f_20","f_23","f_24","f_29","f_31","f_33")

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in char.feature.names) {
  if (class(all.source[[f]])=="character") {
    cat("char feature : ", f ,"\n")
    
    all.source[[f]] <- as.integer(sub('c','',all.source[[f]]))
    
  }
}

sort(unique(all.source[["market"]]))
sort(unique(all.source[["f_1"]]))
sort(unique(all.source[["f_9"]]))
sort(unique(all.source[["f_16"]]))
sort(unique(all.source[["f_20"]]))
sort(unique(all.source[["f_23"]]))
sort(unique(all.source[["f_24"]]))
sort(unique(all.source[["f_29"]]))
sort(unique(all.source[["f_31"]]))
sort(unique(all.source[["f_33"]]))

sort(unique(all.source[["f_7"]]))


sort(unique(str_sub(all.source[["f_7"]],1,1)))
all.source$f7_first <- as.integer(str_sub(all.source[["f_7"]],1,1))

sort(unique(str_sub(all.source[["f_7"]],-2,-1)))
all.source$f7_second <- as.integer(str_sub(all.source[["f_7"]],-2,-1))


sapply(all.source, class)



two.way.division.interaction <- function(dataset, feature_1, feature_2)
                                        {
                                          
                                        new_feature <- paste0(paste0(feature_1,"_",feature_2),"_two_interaction_div")
                                        dataset[[new_feature]] <- ifelse(is.na(dataset[[feature_2]]) | dataset[[feature_2]] == 0, NA, dataset[[feature_2]]/dataset[[feature_1]])
                                        
                                        return(dataset)  
                                        }

two.way.substraction.interaction <- function(dataset, feature_1, feature_2)
                                        {
                                          
                                          new_feature <- paste0(paste0(feature_1,"_",feature_2),"_two_interaction_sub")
                                          dataset[[new_feature]] <- ifelse(is.na(dataset[[feature_1]]) | is.na(dataset[[feature_2]]), NA, dataset[[feature_1]] - dataset[[feature_2]])
                                          
                                          return(dataset)  
                                        }

all.source <- two.way.division.interaction(all.source, "f_13","f_15")
all.source <- two.way.division.interaction(all.source, "f_13","f_30")
all.source <- two.way.division.interaction(all.source, "f_13","f_3")
all.source <- two.way.division.interaction(all.source, "f_13","f_8")
all.source <- two.way.division.interaction(all.source, "f_13","f_17")
all.source <- two.way.division.interaction(all.source, "f_15","f_30")
all.source <- two.way.division.interaction(all.source, "f_15","f_3")
all.source <- two.way.division.interaction(all.source, "f_15","f_8")
all.source <- two.way.division.interaction(all.source, "f_15","f_17")
all.source <- two.way.division.interaction(all.source, "f_30","f_3")
all.source <- two.way.division.interaction(all.source, "f_30","f_8")
all.source <- two.way.division.interaction(all.source, "f_30","f_17")
all.source <- two.way.division.interaction(all.source, "f_3","f_8")
all.source <- two.way.division.interaction(all.source, "f_3","f_17")
all.source <- two.way.division.interaction(all.source, "f_8","f_17")

all.source <- two.way.substraction.interaction(all.source, "f_13","f_15")
all.source <- two.way.substraction.interaction(all.source, "f_13","f_30")
all.source <- two.way.substraction.interaction(all.source, "f_13","f_3")
all.source <- two.way.substraction.interaction(all.source, "f_13","f_8")
all.source <- two.way.substraction.interaction(all.source, "f_13","f_17")
all.source <- two.way.substraction.interaction(all.source, "f_15","f_30")
all.source <- two.way.substraction.interaction(all.source, "f_15","f_3")
all.source <- two.way.substraction.interaction(all.source, "f_15","f_8")
all.source <- two.way.substraction.interaction(all.source, "f_15","f_17")
all.source <- two.way.substraction.interaction(all.source, "f_30","f_3")
all.source <- two.way.substraction.interaction(all.source, "f_30","f_8")
all.source <- two.way.substraction.interaction(all.source, "f_30","f_17")
all.source <- two.way.substraction.interaction(all.source, "f_3","f_8")
all.source <- two.way.substraction.interaction(all.source, "f_3","f_17")
all.source <- two.way.substraction.interaction(all.source, "f_8","f_17")

head(all.source$f_13_f_15_two_interaction_div)

test1 <- all.source %>% filter(customer_id =="219009981584")



all.source <- as.data.table(all.source)


all.source[customer_id == "144112433441",]


setorder(setDT(all.source), date)

all.source[, paste0("customer_market_lag", 1L) := shift(market, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_market_lead", 1L) := shift(market, 1L, type = "lead"), by = c("customer_id")]

all.source[, paste0("customer_f33_lag", 1L) := shift(f_33, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f33_lead", 1L) := shift(f_33, 1L, type = "lead"), by = c("customer_id")]

all.source[, paste0("customer_f23_lag", 1L) := shift(f_23, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f23_lead", 1L) := shift(f_23, 1L, type = "lead"), by = c("customer_id")]

all.source[, paste0("customer_target_lag", 4L) := shift(target, 4L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_target_lag", 5L) := shift(target, 5L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_target_lag", 6L) := shift(target, 6L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_target_lag", 7L) := shift(target, 7L, type = "lag"), by = c("customer_id")]

all.source[, paste0("customer_visit_date_lag", 1L) := shift(date, 1L, type = "lag"), by = c("customer_id")]

#all.source[, "customer_visit_date_lag1_diff" := difftime(all.source$date,   all.source$customer_visit_date_lag1, units = "months")]

all.source[is.na(customer_visit_date_lag1), customer_visit_date_lag1 := as.Date("2016-05-01")]

#all.source[, customer_visit_date_lag1_diff := lengths(Map(seq, customer_visit_date_lag1, date, by = "months")) -1]


all.source$customer_visit_date_lag1_diff <-  (as.yearmon(strptime(all.source$date, format = "%Y-%m-%d"))-as.yearmon(strptime(all.source$customer_visit_date_lag1, format = "%Y-%m-%d")))*12


all.source[, paste0("customer_f0_lag", 1L) := shift(f_0, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f0_lead", 1L) := shift(f_0, 1L, type = "lead"), by = c("customer_id")]

all.source[, paste0("customer_f2_lag", 1L) := shift(f_2, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f2_lead", 1L) := shift(f_2, 1L, type = "lead"), by = c("customer_id")]

all.source[, paste0("customer_f5_lag", 1L) := shift(f_5, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f5_lead", 1L) := shift(f_5, 1L, type = "lead"), by = c("customer_id")]


all.source[, paste0("customer_f3_lag", 1L) := shift(f_3, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f4_lag", 1L) := shift(f_4, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f6_lag", 1L) := shift(f_6, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f8_lag", 1L) := shift(f_8, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f10_lag", 1L) := shift(f_10, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f11_lag", 1L) := shift(f_11, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f12_lag", 1L) := shift(f_12, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f13_lag", 1L) := shift(f_13, 1L, type = "lag"), by = c("customer_id")]

all.source[, paste0("customer_f14_lag", 1L) := shift(f_14, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f15_lag", 1L) := shift(f_15, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f17_lag", 1L) := shift(f_17, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f18_lag", 1L) := shift(f_18, 1L, type = "lag"), by = c("customer_id")]

all.source[, paste0("customer_f21_lag", 1L) := shift(f_21, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f22_lag", 1L) := shift(f_22, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f25_lag", 1L) := shift(f_25, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f26_lag", 1L) := shift(f_26, 1L, type = "lag"), by = c("customer_id")]

all.source[, paste0("customer_f27_lag", 1L) := shift(f_27, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f28_lag", 1L) := shift(f_28, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f30_lag", 1L) := shift(f_30, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f32_lag", 1L) := shift(f_32, 1L, type = "lag"), by = c("customer_id")]

all.source[, paste0("customer_f34_lag", 1L) := shift(f_34, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f35_lag", 1L) := shift(f_35, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f36_lag", 1L) := shift(f_36, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f37_lag", 1L) := shift(f_37, 1L, type = "lag"), by = c("customer_id")]

all.source[, paste0("customer_f38_lag", 1L) := shift(f_38, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f39_lag", 1L) := shift(f_39, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f40_lag", 1L) := shift(f_40, 1L, type = "lag"), by = c("customer_id")]
all.source[, paste0("customer_f41_lag", 1L) := shift(f_41, 1L, type = "lag"), by = c("customer_id")]

all.source <- as.data.frame(all.source)

#c("market","f_1","f_9","f_16","f_19","f_20","f_23","f_24","f_29","f_31","f_33")

all.source <- all.source %>% 
  group_by(customer_id) %>%
  mutate(customer_count = n()) %>%
  ungroup() %>%
  group_by(customer_id, date) %>%
  mutate(customer_date_count = n()) %>%
  ungroup() %>%
  group_by(market, date)%>%
  mutate(market_date_count = n()) %>%
  ungroup() %>%
  group_by(customer_id,market, date)%>%
  mutate(customer_market_date_count = n()) %>%
  ungroup() %>%
  group_by(customer_id,f_1, date)%>%
  mutate(customer_f1_date_count = n()) %>%
  ungroup() %>%
  group_by(customer_id,f_9, date)%>%
  mutate(customer_f9_date_count = n()) %>%
  ungroup() %>%
  group_by(customer_id,f_16, date)%>%
  mutate(customer_f16_date_count = n()) %>%
  ungroup() %>%
  group_by(customer_id,f_19, date)%>%
  mutate(customer_f19_date_count = n()) %>%
  ungroup() %>%
  group_by(customer_id,f_20, date)%>%
  mutate(customer_f20_date_count = n()) %>%
  ungroup() %>%
  group_by(customer_id,f_23, date)%>%
  mutate(customer_f23_date_count = n()) %>%
  ungroup() %>%
  group_by(customer_id,f_24, date)%>%
  mutate(customer_f24_date_count = n()) %>%
  ungroup() %>%
  group_by(customer_id,f_29, date)%>%
  mutate(customer_f29_date_count = n()) %>%
  ungroup() %>%
  group_by(customer_id,f_31, date)%>%
  mutate(customer_f31_date_count = n()) %>%
  ungroup() %>%
  group_by(customer_id,f_33, date)%>%
  mutate(customer_f33_date_count = n()) %>% 
  ungroup() %>%
  group_by(customer_id,f7_first, date)%>%
  mutate(customer_f7first_date_count = n()) %>% 
  ungroup() %>%
  group_by(customer_id,f7_second, date)%>%
  mutate(customer_f7second_date_count = n())%>% 
  ungroup() %>%
  group_by(customer_id,f7_first ,f7_second, date)%>%
  mutate(customer_f7firstsecond_date_count = n())

system.time(
  all.source <- all.source %>%
    rowwise() %>%
    mutate(row_max = max(c(f_3,f_4,f_6,f_8,f_10,f_11,f_12,f_13,f_14,f_15,f_17,f_18,f_21,f_22,f_25
                           ,f_26,f_27,f_28,f_30,f_32,f_34,f_35,f_36,f_37,f_38,f_39,f_40,f_41)),
           row_min = min(c(f_3,f_4,f_6,f_8,f_10,f_11,f_12,f_13,f_14,f_15,f_17,f_18,f_21,f_22,f_25
                           ,f_26,f_27,f_28,f_30,f_32,f_34,f_35,f_36,f_37,f_38,f_39,f_40,f_41)),
           row_mean = mean(c(f_3,f_4,f_6,f_8,f_10,f_11,f_12,f_13,f_14,f_15,f_17,f_18,f_21,f_22,f_25
                             ,f_26,f_27,f_28,f_30,f_32,f_34,f_35,f_36,f_37,f_38,f_39,f_40,f_41)),
           row_sd = sd(c(f_3,f_4,f_6,f_8,f_10,f_11,f_12,f_13,f_14,f_15,f_17,f_18,f_21,f_22,f_25
                         ,f_26,f_27,f_28,f_30,f_32,f_34,f_35,f_36,f_37,f_38,f_39,f_40,f_41)),
           row_skewness = skewness(c(f_3,f_4,f_6,f_8,f_10,f_11,f_12,f_13,f_14,f_15,f_17,f_18,f_21,f_22,f_25
                                     ,f_26,f_27,f_28,f_30,f_32,f_34,f_35,f_36,f_37,f_38,f_39,f_40,f_41)),
           row_kurtosis = kurtosis(c(f_3,f_4,f_6,f_8,f_10,f_11,f_12,f_13,f_14,f_15,f_17,f_18,f_21,f_22,f_25
                                     ,f_26,f_27,f_28,f_30,f_32,f_34,f_35,f_36,f_37,f_38,f_39,f_40,f_41))
           
    )
)

all.source <- as.data.table(all.source)



two.features.concatenation <- function(dataset, feature1, feature2)
{
  
  new.feature <- paste0(feature1,"_",feature2)
  dataset[[new.feature]] <- paste0(dataset[[feature1]],dataset[[feature2]])
  
  return(dataset)
}


# c("market","f_1","f_9","f_16","f_19","f_20","f_23","f_24","f_29","f_31","f_33") 

all.source <- two.features.concatenation(all.source,"market","f_33")
all.source <- two.features.concatenation(all.source,"market","f_23")
all.source <- two.features.concatenation(all.source,"f_33","f_23")

head(all.source)

train.source <- all.source %>%
  filter(target != -1)

test.source <- all.source %>%
  filter(target == -1)

gc()
dim(all.source)
dim(train.source)

length(unique(all.source$market_f_33))
length(unique(all.source$market_f_23))
length(unique(all.source$f_33_f_23))



# Column Name	Data Type
# customer_id	int
# market	char
# date	date
# f_0	int
# f_2	int
# f_5	int
# f_3	  float  
# f_4	  float
# f_6	  float
# f_8	  float
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




for (f in char.feature.names) {
  
  cat("char feature : ", f ,"\n")
  cat("unique lenght : ", length(unique(train.source[[f]])),"\n")
  
}

for (f in char.feature.names) {
  
  cat("char feature : ", f ,"\n")
  cat("unique lenght : ", length(unique(test.source[[f]])),"\n")
  
}



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
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("f0_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("customer_id","f_0")]


system.time({
  setkey(X_build, customer_id, f_0)
  X_build_last <- X_build[unique(X_build)[, key(X_build), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","f_0","f0_target_lag1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id","f_0"))
#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("f0_target_lead", 1L) := shift(target, 1L, type = "lead"), by = c("customer_id","f_0")]


system.time({
  setkey(X_build, customer_id, f_0)
  X_build_last <- X_build[complete.cases(X_build[ , f0_target_lead1]),]
  X_build_last <- X_build_last[unique(X_build_last)[, key(X_build_last), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","f_0","f0_target_lead1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id","f_0"))
##################################################################################################################
#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("f2_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("customer_id","f_2")]


system.time({
  setkey(X_build, customer_id, f_2)
  X_build_last <- X_build[unique(X_build)[, key(X_build), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","f_2","f2_target_lag1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id","f_2"))
#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("f2_target_lead", 1L) := shift(target, 1L, type = "lead"), by = c("customer_id","f_2")]


system.time({
  setkey(X_build, customer_id, f_2)
  X_build_last <- X_build[complete.cases(X_build[ , f2_target_lead1]),]
  X_build_last <- X_build_last[unique(X_build_last)[, key(X_build_last), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","f_2","f2_target_lead1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id","f_2"))
##################################################################################################################

#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("f5_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("customer_id","f_5")]


system.time({
  setkey(X_build, customer_id, f_5)
  X_build_last <- X_build[unique(X_build)[, key(X_build), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","f_5","f5_target_lag1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id","f_5"))
#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("f5_target_lead", 1L) := shift(target, 1L, type = "lead"), by = c("customer_id","f_5")]


system.time({
  setkey(X_build, customer_id, f_5)
  X_build_last <- X_build[complete.cases(X_build[ , f5_target_lead1]),]
  X_build_last <- X_build_last[unique(X_build_last)[, key(X_build_last), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("customer_id","f_5","f5_target_lead1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("customer_id","f_5"))
##################################################################################################################


#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("marketf33_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("market_f_33")]


system.time({
  setkey(X_build, market_f_33)
  X_build_last <- X_build[unique(X_build)[, key(X_build), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("market_f_33","marketf33_target_lag1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("market_f_33"))
#################################################################################################################


#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("marketf23_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("market_f_23")]


system.time({
  setkey(X_build, market_f_23)
  X_build_last <- X_build[unique(X_build)[, key(X_build), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("market_f_23","marketf23_target_lag1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("market_f_23"))
#################################################################################################################
#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

X_build[, paste0("f33_f23_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("f_33_f_23")]


system.time({
  setkey(X_build, f_33_f_23)
  X_build_last <- X_build[unique(X_build)[, key(X_build), with=FALSE], mult="last"]
})

X_build[customer_id == "144112433441",]
X_build_last[customer_id == "144112433441",]

X_build_last <- as.data.frame(X_build_last)

last.columns <- c("f_33_f_23","f33_f23_target_lag1")

X_build_last <- X_build_last[,last.columns]

X_val <- left_join(X_val,X_build_last, by=c("f_33_f_23"))
#################################################################################################################

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


#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("f0_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("customer_id","f_0")]


system.time({
  setkey(train.source, customer_id, f_0)
  train.source_last <- train.source[unique(train.source)[, key(train.source), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","f_0","f0_target_lag1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id","f_0"))
#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("f0_target_lead", 1L) := shift(target, 1L, type = "lead"), by = c("customer_id","f_0")]


system.time({
  setkey(train.source, customer_id, f_0)
  train.source_last <- train.source[complete.cases(train.source[ , f0_target_lead1]),]
  train.source_last <- train.source_last[unique(train.source_last)[, key(train.source_last), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","f_0","f0_target_lead1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id","f_0"))
##################################################################################################################
#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("f2_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("customer_id","f_2")]


system.time({
  setkey(train.source, customer_id, f_2)
  train.source_last <- train.source[unique(train.source)[, key(train.source), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","f_2","f2_target_lag1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id","f_2"))
#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("f2_target_lead", 1L) := shift(target, 1L, type = "lead"), by = c("customer_id","f_2")]


system.time({
  setkey(train.source, customer_id, f_2)
  train.source_last <- train.source[complete.cases(train.source[ , f2_target_lead1]),]
  train.source_last <- train.source_last[unique(train.source_last)[, key(train.source_last), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","f_2","f2_target_lead1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id","f_2"))
##################################################################################################################

#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("f5_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("customer_id","f_5")]


system.time({
  setkey(train.source, customer_id, f_5)
  train.source_last <- train.source[unique(train.source)[, key(train.source), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","f_5","f5_target_lag1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id","f_5"))
#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("f5_target_lead", 1L) := shift(target, 1L, type = "lead"), by = c("customer_id","f_5")]


system.time({
  setkey(train.source, customer_id, f_5)
  train.source_last <- train.source[complete.cases(train.source[ , f5_target_lead1]),]
  train.source_last <- train.source_last[unique(train.source_last)[, key(train.source_last), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("customer_id","f_5","f5_target_lead1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("customer_id","f_5"))
##################################################################################################################


#################################################################################################################
X_build <- as.data.table(X_build )
setorder(setDT(X_build), date)

train.source[, paste0("marketf33_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("market_f_33")]


system.time({
  setkey(train.source, market_f_33)
  train.source_last <- train.source[unique(train.source)[, key(train.source), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("market_f_33","marketf33_target_lag1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("market_f_33"))
#################################################################################################################


#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("marketf23_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("market_f_23")]


system.time({
  setkey(train.source, market_f_23)
  train.source_last <- train.source[unique(train.source)[, key(train.source), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("market_f_23","marketf23_target_lag1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("market_f_23"))
#################################################################################################################
#################################################################################################################
train.source <- as.data.table(train.source )
setorder(setDT(train.source), date)

train.source[, paste0("f33_f23_target_lag", 1L) := shift(target, 1L, type = "lag"), by = c("f_33_f_23")]


system.time({
  setkey(train.source, f_33_f_23)
  train.source_last <- train.source[unique(train.source)[, key(train.source), with=FALSE], mult="last"]
})

train.source[customer_id == "144112433441",]
train.source_last[customer_id == "144112433441",]

train.source_last <- as.data.frame(train.source_last)

last.columns <- c("f_33_f_23","f33_f23_target_lag1")

train.source_last <- train.source_last[,last.columns]

test.source <- left_join(test.source,train.source_last, by=c("f_33_f_23"))
#################################################################################################################


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



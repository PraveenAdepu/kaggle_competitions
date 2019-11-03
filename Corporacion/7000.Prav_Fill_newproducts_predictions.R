

test <- fread("./input/test.csv")
test$date <- as.Date(parse_date_time(test$date,'%y-%m-%d'))
test$store_item_nbr <- paste(test$store_nbr, test$item_nbr, sep="_")

test <- as.data.frame(test)

test_only_records <- readRDS("./input/test_only_records.RDS")

mean01 <- read_csv("./submissions/Prav_mean01.csv")

test_only_records$store_nbr <- NULL
test_only_records$item_nbr  <- NULL
head(test_only_records)
testing <- left_join(test, test_only_records, by="store_item_nbr")
head(testing)

testing$train_record[is.na(testing$train_record)]<- 1


testingSet <- left_join(testing, mean01, by="id")

head(testingSet)

testingSet %>% filter(train_record == 0) %>% summarise(n()) # 711056 records

items <- read_csv("./input/items.csv")

testingSet <- left_join(testingSet, items, by="item_nbr")

testingSet_predictions <- testingSet %>% filter(train_record == 1)
testingSet_newproducts <- testingSet %>% filter(train_record == 0)

head(testingSet_newproducts)

##################################################################################################
testingSet %>% filter(store_nbr ==1 & item_nbr == 103501)

testingSet_predictions_class_means <- testingSet_predictions %>% 
                                          #filter(store_nbr ==1 & class == 3008) %>%
                                          group_by( date, store_nbr,class) %>%
                                          summarise(mean_unit_sales = mean(unit_sales))

testingSet_newproducts_preds <- left_join(testingSet_newproducts, testingSet_predictions_class_means, by = c("date","store_nbr","class"))

head(testingSet_newproducts_preds)
testingSet_newproducts_preds$mean_unit_sales[is.na(testingSet_newproducts_preds$mean_unit_sales)] <- 0
testingSet_newproducts_preds %>% filter(store_nbr ==1 & item_nbr == 105576)
sum(testingSet_newproducts_preds$mean_unit_sales)

testingSet_newproducts_preds <- testingSet_newproducts_preds %>% select(id, mean_unit_sales) %>% dplyr::rename(unit_sales = mean_unit_sales)

testingSet_predictions <- testingSet_predictions %>% select(id, unit_sales)

testing_preds <- rbind(testingSet_predictions,testingSet_newproducts_preds)
head(mean01)
head(testing_preds)
testing_sub <- left_join(mean01, testing_preds, by="id")
head(testing_sub)
testing_sub$unit_sales.x <- NULL
names(testing_sub)[2] <- "unit_sales"
summary(testing_sub$unit_sales)
sum(testing_sub$unit_sales)
write.csv(testing_sub,"./submissions/Prav_mean01_newproducts_process.csv", row.names = FALSE, quote = FALSE)




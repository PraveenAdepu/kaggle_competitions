

stores_01_05 <- read_csv("./submissions/stores01-05.csv")
stores_06_10 <- read_csv("./submissions/stores06-10.csv")
stores_11_15 <- read_csv("./submissions/stores11-15.csv")
stores_16_20 <- read_csv("./submissions/stores16-20.csv")
stores_21_25 <- read_csv("./submissions/stores21-25.csv")
stores_26_30 <- read_csv("./submissions/stores26-30.csv")
stores_31_35 <- read_csv("./submissions/stores31-35.csv")
stores_36_40 <- read_csv("./submissions/stores36-40.csv")
stores_41_45 <- read_csv("./submissions/stores41-45.csv")
stores_46_50 <- read_csv("./submissions/stores46-50.csv")
stores_51_54 <- read_csv("./submissions/stores51-54.csv")
#mean01 <- read_csv("./submissions/Prav_nnetar01.csv")

#sum(stores_11_15$unit_sales)

predictions <- rbind(stores_01_05, stores_06_10)
predictions <- rbind(predictions, stores_11_15)
predictions <- rbind(predictions, stores_16_20)
predictions <- rbind(predictions, stores_21_25)
predictions <- rbind(predictions, stores_26_30)
predictions <- rbind(predictions, stores_31_35)
predictions <- rbind(predictions, stores_36_40)
predictions <- rbind(predictions, stores_41_45)
predictions <- rbind(predictions, stores_46_50)
predictions <- rbind(predictions, stores_51_54)

head(predictions) #2606624

test <-fread('./input/test.csv') #3370464

test$date  <- as.Date(parse_date_time(test$date,'%y-%m-%d'))

head(test)

# mean01 %>% filter(id == 127642592)
# t1 <- stores_11_15 %>% filter(store_nbr == 11 & item_nbr == 103501)


testingSet <- left_join(test, predictions, by=c("store_nbr","item_nbr","date"))

head(testingSet)
is.data.frame(testingSet)
submission <- testingSet[,c("id","unit_sales")]

summary(submission$unit_sales)

submission$unit_sales[is.na(submission$unit_sales)] <- 0
submission$unit_sales[submission$unit_sales < 0] <- 0

write.csv(submission,"./submissions/prav_prophet01.csv", row.names = FALSE, quote = FALSE)

mean01 <- read_csv("./submissions/Prav_mean01.csv")


all_ensemble <- left_join(submission, mean01, by = "id")


head(all_ensemble)

all_ensemble <- all_ensemble %>% filter( unit_sales.x <= 20000)
ensemble.features <- setdiff(names(all_ensemble),"id")
cor(all_ensemble[, ensemble.features])

sum(all_ensemble$unit_sales.x)
sum(all_ensemble$unit_sales.y)

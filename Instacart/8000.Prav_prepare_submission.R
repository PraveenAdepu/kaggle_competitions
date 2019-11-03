model.name = "xgb06"
cv = 5

score.cutoff <- data.frame(fold=integer(),
                 OrderSize=integer(),
                 cutoff=double(),
                 meanf1core=double()
                 )

for (fold in 1:cv)
{

  cat("fold : ", fold , "processing \n")
X_val   <- subset(trainingSet, CVindices == fold, select = -c( CVindices))

sub_file <- paste0("./submissions/prav.",model.name,".fold",fold,".csv")
cat("processing file : ", sub_file)

model    <- read_csv(sub_file)

cols <- c("user_id","order_id","product_id","reordered")

model <- left_join(model,X_val[, cols], by=c("user_id","order_id","product_id"))

model <- model %>%
          group_by(order_id) %>%
          mutate(OrderSize = n()) %>%
          ungroup()
for (ordersize in unique(model$OrderSize)) {
  cat("ordersize " , ordersize, "\n", sep = "")        
for(i in seq(0.15,0.25, by = 0.01))
{
dt <- data.frame(user_id=model$user_id, purch=model$reordered, pred=model$pred, OrderSize = model$OrderSize)
f1score <- dt %>%
  filter(OrderSize == ordersize) %>%
  group_by(user_id) %>%
  summarise(f1score=f1Score(purch, pred, cutoff=i))


presentScore <- data.frame(fold=fold, OrderSize = ordersize,  cutoff=i, meanf1core =mean(f1score$f1score, na.rm = TRUE))
score.cutoff <- rbind(score.cutoff, presentScore)

cat("fold " , i  , " F1 score - including NA : " , mean(f1score$f1score, na.rm = TRUE), "\n", sep = "")
# f1score[is.na(f1score)] <- 0
# cat("fold " , i  , " F1 score - NA replace with 0 : " , mean(f1score$f1score), "\n", sep = "")

}

}

}


head(score.cutoff)

final.score.cutoff <- score.cutoff %>%
    arrange(fold, OrderSize, meanf1core) %>%
    group_by(fold, OrderSize) %>% 
    filter(meanf1core == max(meanf1core)) %>% 
    filter(1:n() == 1)
  
head(final.score.cutoff)    
OrderSize.cutoff.foldsMeanF1score <- final.score.cutoff %>%
                                    #filter(OrderSize == 2) %>%
                                    group_by(OrderSize) %>%
                                    mutate(foldsMeanF1score = mean(meanf1core)) %>%
                                    filter(cutoff == mean(cutoff)) %>% 
                                    filter(1:n() == 1) %>%
                                    select(OrderSize,cutoff,foldsMeanF1score)


head(OrderSize.cutoff.foldsMeanF1score)

summary(OrderSize.cutoff.foldsMeanF1score$OrderSize)

####################################################################################################################

fold1    <- read_csv("./submissions/prav.xgb06.fold1.csv")
fold2    <- read_csv("./submissions/prav.xgb06.fold2.csv")
fold3    <- read_csv("./submissions/prav.xgb06.fold3.csv")
fold4    <- read_csv("./submissions/prav.xgb06.fold4.csv")
fold5    <- read_csv("./submissions/prav.xgb06.fold5.csv")


model <- rbind(fold1, fold2, fold3, fold4, fold5)

cols <- c("user_id","order_id","product_id","reordered")

model <- left_join(model,trainingSet[, cols], by=c("user_id","order_id","product_id"))

model <- model %>%
  group_by(order_id) %>%
  mutate(OrderSize = n()) %>%
  ungroup()
fold = 1
for (ordersize in unique(model$OrderSize)) {
  cat("ordersize " , ordersize, "\n", sep = "")        
  for(i in seq(0.10,0.85, by = 0.01))
  {
    dt <- data.frame(user_id=model$user_id, purch=model$reordered, pred=model$pred, OrderSize = model$OrderSize)
    f1score <- dt %>%
      filter(OrderSize == ordersize) %>%
      group_by(user_id) %>%
      summarise(f1score=f1Score(purch, pred, cutoff=i))
    
    
    presentScore <- data.frame(fold=fold, OrderSize = ordersize,  cutoff=i, meanf1core =mean(f1score$f1score, na.rm = TRUE))
    score.cutoff <- rbind(score.cutoff, presentScore)
    
    cat("fold " , i  , " F1 score - including NA : " , mean(f1score$f1score, na.rm = TRUE), "\n", sep = "")
    # f1score[is.na(f1score)] <- 0
    # cat("fold " , i  , " F1 score - NA replace with 0 : " , mean(f1score$f1score), "\n", sep = "")
    
  }
  
}

final.score.cutoff <- score.cutoff %>%
  arrange(fold, OrderSize, meanf1core) %>%
  group_by(fold, OrderSize) %>% 
  filter(meanf1core == max(meanf1core)) %>% 
  filter(1:n() == 1)

head(final.score.cutoff)  
summary(final.score.cutoff$cutoff) 

 final.score.cutoff %>%
  arrange(cutoff)
#########################################################################################################################################

xgb01.full           <- read_csv("./submissions/prav.xgb06.full.csv")
sample_submission    <- read_csv("./input/sample_submission.csv")

xgb01.full <- xgb01.full %>%
  group_by(order_id) %>%
  mutate(OrderSize = n()) %>%
  ungroup()


full.model <- left_join(xgb01.full, final.score.cutoff, by="OrderSize")

full.model <- full.model %>%
                  mutate(cutoff = ifelse(is.na(cutoff),0.17,cutoff)
                         , reordered = ifelse(pred >= cutoff,1,0)
                          )

head(full.model)


submission <- full.model %>%
              filter(reordered == 1) %>%
              group_by(order_id) %>%
              summarise(
                products = paste(product_id, collapse = " ")
              )

missing <- data.frame(
  order_id = unique(full.model$order_id[!full.model$order_id %in% submission$order_id]),
  products = "None"
)

submission <- submission %>% 
                    bind_rows(missing) %>% 
                    arrange(order_id)
head(submission,10)

write.csv(submission, './submissions/prav.xgb06.full-sub.csv', row.names=FALSE, quote = FALSE)


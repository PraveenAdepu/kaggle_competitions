
MA_preds    <- read_csv("./submissions/Prav_MAfull-test.csv")
MA_preds$id <- NULL
xgb_preds   <-  read_csv("./submissions/Prav.xgb02.full.csv")

ET02_preds   <-  read_csv("./submissions/Prav.ET_knn_02.full.csv")
KNN_preds   <-  read_csv("./submissions/Prav.ET_knn.full.csv")

ET02_preds$visitor_pred_knn <- NULL
KNN_preds$visitor_pred_ET <- NULL

### get & process the sample submission file
sample_sub <- fread("./input/sample_submission.csv")
sample_sub$visitors <- NULL
sample_sub$store_id <- substr(sample_sub$id, 1, 20)
sample_sub$visit_date <- substr(sample_sub$id, 22, 31)
sample_sub$visit_date <- as.Date(parse_date_time(sample_sub$visit_date,'%y-%m-%d'))


head(sample_sub)
head(MA_preds)
head(xgb_preds)
head(ET02_preds)
head(KNN_preds)

names(MA_preds) <- c("air_store_id","visit_date","visitors_MA_pred")

### generate the final submission file
submission <- left_join(sample_sub, MA_preds, c("store_id" = "air_store_id", 'visit_date' = 'visit_date'))
submission <- left_join(submission, xgb_preds, c("store_id" = "air_store_id", 'visit_date' = 'visit_date'))
submission <- left_join(submission, ET02_preds, c("store_id" = "air_store_id", 'visit_date' = 'visit_date'))
submission <- left_join(submission, KNN_preds, c("store_id" = "air_store_id", 'visit_date' = 'visit_date'))

head(submission)

summary(submission$visitors_MA_pred)
summary(submission$visitors_pred)
submission$ET02knn    <- 0.5 * submission$visitor_pred_ET + 0.5 * submission$visitor_pred_knn
submission$ET02knnMA  <- 0.5 * submission$ET02knn + 0.5 * submission$visitors_MA_pred
submission$visitors   <- 0.5 * submission$ET02knnMA  + 0.5 * submission$visitors_pred

summary(submission$visitors)
#submission$visitors[is.na(submission$visitors)] <- 0
is.data.frame(submission)
final.columns <- c('id', 'visitors')
final_sub <- submission[,final.columns]
head(final_sub)
write.csv(final_sub, "./submissions/Prav_xgb02_ET02knnMA_mean.csv", row.names = FALSE)
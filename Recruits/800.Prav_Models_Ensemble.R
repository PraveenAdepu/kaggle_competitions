
MA_preds          <- read_csv("./submissions/Prav_MA_validation_preds.csv")
prophet_preds     <- read_csv("./submissions/Prav_prophet_validation_preds.csv")
forecast_preds    <-  read_csv("./submissions/Prav_forecastHybrid_validation_preds.csv")
forecastMix_preds <-  read_csv("./submissions/Prav_forecastMix_validation_preds.csv")
ET_knn_preds      <-  read_csv("./submissions/Prav_ET_knn_validation_preds.csv")
xgb_preds      <-  read_csv("./submissions/Prav_xgb_validation_preds.csv")
xgb02_preds      <-  read_csv("./submissions/Prav_xgb02_validation_preds.csv")
ET_knn_02_preds      <-  read_csv("./submissions/Prav_ET_knn_02_validation_preds.csv")
head(MA_preds)
head(prophet_preds)
head(forecast_preds)
head(forecastMix_preds)
head(ET_knn_preds)
head(xgb_preds)
head(xgb02_preds)
head(ET_knn_02_preds)
prophet_preds$visitors <- NULL
forecast_preds$visitors <- NULL


names(prophet_preds) <- c("air_store_id","visit_date","visitor_prophet")
names(forecast_preds) <- c("air_store_id","visit_date","visitor_forecast")
names(xgb02_preds) <- c("air_store_id","visit_date","visitor_pred_xgb02")
names(ET_knn_02_preds) <- c("air_store_id","visit_date","visitor_pred_ET02","visitor_pred_knn02")

LSTM03_fold1 <- read_csv("./submissions/Prav.LSTM03.fold1.csv")
LSTM03_fold2 <- read_csv("./submissions/Prav.LSTM03.fold2.csv")
LSTM03_fold3 <- read_csv("./submissions/Prav.LSTM03.fold3.csv")
LSTM03_fold4 <- read_csv("./submissions/Prav.LSTM03.fold4.csv")
LSTM03_fold5 <- read_csv("./submissions/Prav.LSTM03.fold5.csv")

LSTM03_preds <- rbind(LSTM03_fold1,LSTM03_fold2,LSTM03_fold3,LSTM03_fold4,LSTM03_fold5) # ,LSTM02_fold3,LSTM02_fold4,LSTM02_fold5
names(LSTM03_preds) <- c("air_store_id","visit_date","visitor_LSTM03")



LSTM04_fold1 <- read_csv("./submissions/Prav.LSTM04.fold1.csv")
LSTM04_fold2 <- read_csv("./submissions/Prav.LSTM04.fold2.csv")
LSTM04_fold3 <- read_csv("./submissions/Prav.LSTM04.fold3.csv")
LSTM04_fold4 <- read_csv("./submissions/Prav.LSTM04.fold4.csv")
LSTM04_fold5 <- read_csv("./submissions/Prav.LSTM04.fold5.csv")
LSTM04_preds <- rbind(LSTM04_fold1,LSTM04_fold2,LSTM04_fold3,LSTM04_fold4,LSTM04_fold5)
names(LSTM04_preds) <- c("air_store_id","visit_date","visitor_LSTM04")



CNN01_fold1 <- read_csv("./submissions/Prav.CNN01.fold1.csv")
CNN01_fold2 <- read_csv("./submissions/Prav.CNN01.fold2.csv")
CNN01_fold3 <- read_csv("./submissions/Prav.CNN01.fold3.csv")
CNN01_fold4 <- read_csv("./submissions/Prav.CNN01.fold4.csv")
CNN01_fold5 <- read_csv("./submissions/Prav.CNN01.fold5.csv")
CNN01_preds <- rbind(CNN01_fold1,CNN01_fold2,CNN01_fold3,CNN01_fold4,CNN01_fold5)
names(CNN01_preds) <- c("air_store_id","visit_date","visitor_CNN01")


CNN02_fold1 <- read_csv("./submissions/Prav.CNN02.fold1.csv")
CNN02_fold2 <- read_csv("./submissions/Prav.CNN02.fold2.csv")
CNN02_fold3 <- read_csv("./submissions/Prav.CNN02.fold3.csv")
CNN02_fold4 <- read_csv("./submissions/Prav.CNN02.fold4.csv")
CNN02_fold5 <- read_csv("./submissions/Prav.CNN02.fold5.csv")
CNN02_preds <- rbind(CNN02_fold1,CNN02_fold2,CNN02_fold3,CNN02_fold4,CNN02_fold5)
names(CNN02_preds) <- c("air_store_id","visit_date","visitor_CNN02")

all_preds <- inner_join(MA_preds,prophet_preds,by = c("air_store_id","visit_date"))

all_preds <- inner_join(all_preds,forecast_preds,by = c("air_store_id","visit_date"))
all_preds <- inner_join(all_preds,forecastMix_preds,by = c("air_store_id","visit_date"))
all_preds <- inner_join(all_preds,ET_knn_preds,by = c("air_store_id","visit_date"))
all_preds <- inner_join(all_preds,xgb_preds,by = c("air_store_id","visit_date"))
all_preds <- inner_join(all_preds,xgb02_preds,by = c("air_store_id","visit_date"))
all_preds <- inner_join(all_preds, LSTM03_preds, by = c("air_store_id","visit_date"))
all_preds <- inner_join(all_preds, LSTM04_preds, by = c("air_store_id","visit_date"))
all_preds <- inner_join(all_preds, CNN01_preds, by = c("air_store_id","visit_date"))
all_preds <- inner_join(all_preds, CNN02_preds, by = c("air_store_id","visit_date"))
all_preds <- inner_join(all_preds,ET_knn_02_preds,by = c("air_store_id","visit_date"))

summary(all_preds$visitor_forecastMix);summary(all_preds$visitor_CNN01);summary(all_preds$visitor_CNN02);summary(all_preds$visitor_pred_xgb02)

all_preds$visitor_CNN01 <- ifelse(all_preds$visitor_CNN01 < 0, 0 , all_preds$visitor_CNN01)
all_preds$visitor_CNN02 <- ifelse(all_preds$visitor_CNN02 < 0, 0 , all_preds$visitor_CNN02)

score(log1p(all_preds$visitors), log1p(all_preds$visitors_pred),"rmse")      # CV - 0.53 , LB - 0.49
score(log1p(all_preds$visitors), log1p(all_preds$visitor_prophet),"rmse")
score(log1p(all_preds$visitors), log1p(all_preds$visitor_forecast),"rmse")
score(log1p(all_preds$visitors), log1p(all_preds$visitor_forecastMix),"rmse")
score(log1p(all_preds$visitors), log1p(all_preds$visitor_pred_ET),"rmse")    # 0.556
score(log1p(all_preds$visitors), log1p(all_preds$visitor_pred_knn),"rmse")   # 0.569
score(log1p(all_preds$visitors), log1p(all_preds$visitor_pred_xgb),"rmse")   # 0.5272433
score(log1p(all_preds$visitors), log1p(all_preds$visitor_pred_xgb02),"rmse") # 0.5246513
score(log1p(all_preds$visitors), log1p(all_preds$visitor_LSTM03),"rmse")
score(log1p(all_preds$visitors), log1p(all_preds$visitor_LSTM04),"rmse")
score(log1p(all_preds$visitors), log1p(all_preds$visitor_CNN01),"rmse")
score(log1p(all_preds$visitors), log1p(all_preds$visitor_CNN02),"rmse")
score(log1p(all_preds$visitors), log1p(all_preds$visitor_pred_ET02),"rmse")    # 0.556
score(log1p(all_preds$visitors), log1p(all_preds$visitor_pred_knn02),"rmse")   # 0.569

cor(all_preds[,c("visitors_pred","visitor_prophet","visitor_forecast","visitor_forecastMix","visitor_pred_ET","visitor_pred_knn","visitor_pred_xgb","visitor_pred_xgb02",
                 "visitor_LSTM03","visitor_LSTM04","visitor_CNN01","visitor_CNN02","visitor_pred_ET02","visitor_pred_knn02")])

score(log1p(all_preds$visitors), log1p(0.5 *all_preds$visitor_LSTM04+ 0.5 *all_preds$visitor_LSTM03),"rmse")
score(log1p(all_preds$visitors), log1p(0.5 *all_preds$visitor_CNN01+ 0.5 *all_preds$visitor_LSTM03),"rmse")
score(log1p(all_preds$visitors), log1p(0.5 *all_preds$visitor_CNN02+ 0.5 *all_preds$visitor_LSTM03),"rmse")
score(log1p(all_preds$visitors), log1p(0.5 *all_preds$visitor_CNN01+ 0.5 *all_preds$visitor_CNN02),"rmse")
score(log1p(all_preds$visitors), log1p(0.5 *all_preds$visitor_forecastMix+ 0.5 *all_preds$visitor_LSTM03),"rmse")
score(log1p(all_preds$visitors), log1p(0.5 *all_preds$visitor_pred_ET+ 0.5 *all_preds$visitor_pred_knn),"rmse") # CV - 0.540 , LB - 0.501
score(log1p(all_preds$visitors), log1p(0.5 *all_preds$visitor_pred_ET02+ 0.5 *all_preds$visitor_pred_knn),"rmse") # CV - 0.540 , LB - 0.501



all_preds$ET02knn     <- 0.5 *all_preds$visitor_pred_ET02 + 0.5 *all_preds$visitor_pred_knn
all_preds$ET02knnMA    <- 0.5 *all_preds$ET02knn + 0.5 *all_preds$visitors_pred
all_preds$LSTM03CCN01 <- 0.5 *all_preds$visitor_LSTM03 + 0.5 *all_preds$visitor_CNN01
score(log1p(all_preds$visitors), log1p(all_preds$ET02knnMA),"rmse") # CV - 0.540 , LB - 0.501
score(log1p(all_preds$visitors), log1p(0.5 *all_preds$visitor_pred_xgb02+ 0.5 *all_preds$ET02knnMA),"rmse") # CV - 0.520 , LB - 
score(log1p(all_preds$visitors), log1p(0.5 *all_preds$visitors_pred+ 0.5 *all_preds$ETknn),"rmse") # CV - 0.540 , LB - 0.501
score(log1p(all_preds$visitors), log1p(0.5 *all_preds$visitors_pred+ 0.5 *all_preds$visitor_pred_xgb),"rmse") # CV - 0.540 , LB - 0.501
score(log1p(all_preds$visitors), log1p(0.5 *all_preds$visitors_pred+ 0.5 *all_preds$LSTM03CCN01),"rmse") # CV - 0.540 , LB - 0.501
score(log1p(all_preds$visitors), log1p(0.5 *all_preds$visitors_pred+ 0.5 *all_preds$ETknnXgb),"rmse") # CV - 0.540 , LB - 0.501

head(all_preds)


CNN01_preds %>% filter(visitor_CNN01 < 0 )

test <- all_preds %>% filter(air_store_id == "air_00a91d42b08b08d9")


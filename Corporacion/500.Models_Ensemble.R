
setwd("C:/Users/SriPrav/Documents/R/34Corporacion")
root_directory = "C:/Users/SriPrav/Documents/R/34Corporacion"

lgbm <- read_csv("./submissions/lgb.csv")
mean_ref <- read_csv("./submissions/Prav_mean_baseline.csv")

head(lgbm); head(mean_ref)

models <- rbind(lgbm, mean_ref)

models <- models %>%
            group_by(id) %>%
            summarise(unit_sales = mean(unit_sales))

head(models)

write.csv(models, "./submissions/Prav_lgb_mean_ensemble.csv",row.names = FALSE, quote = FALSE)

lgbm02 <- read_csv("./submissions/Prav_lgbm02.csv")

all.models <- left_join(models, lgbm02, by="id")
cor.features <- setdiff(names(all.models),"id")
cor(all.models[,cor.features])

all.models$unit_sales <- 0.5 * all.models$unit_sales.x + 0.5 * all.models$unit_sales.y

head(all.models)

sub.cols <- c("id","unit_sales")

write.csv(all.models[,sub.cols], "./submissions/Prav_lgb2_mean_ensemble.csv",row.names = FALSE, quote = FALSE)

model01 <- read_csv("./submissions/Prav_lgb2_mean_ensemble.csv")
model02 <- read_csv("./submissions/Prav_lgbm03.csv")

all.models <- left_join(model01, model02, by="id")
cor.features <- setdiff(names(all.models),"id")
cor(all.models[,cor.features])

all.models$unit_sales <- 0.5 * all.models$unit_sales.x + 0.5 * all.models$unit_sales.y

head(all.models)

sub.cols <- c("id","unit_sales")

write.csv(all.models[,sub.cols], "./submissions/Prav_lgb3_mean_ensemble.csv",row.names = FALSE, quote = FALSE)
##########################################################################################################################

model01 <- read_csv("./submissions/Prav_lgb3_mean_ensemble.csv")
model02 <- read_csv("./submissions/ensemble_ma_lgbm_cat.csv")

all.models <- left_join(model01, model02, by="id")
cor.features <- setdiff(names(all.models),"id")
cor(all.models[,cor.features])

all.models$unit_sales <- 0.5 * all.models$unit_sales.x + 0.5 * all.models$unit_sales.y

head(all.models)

sub.cols <- c("id","unit_sales")

write.csv(all.models[,sub.cols], "./submissions/Prav_lgb3_ref_ensemble.csv",row.names = FALSE, quote = FALSE)

##########################################################################################################################

model01 <- read_csv("./submissions/Prav_lgb3_ref_ensemble.csv")
model02 <- read_csv("./submissions/lgb.csv")

all.models <- left_join(model01, model02, by="id")
cor.features <- setdiff(names(all.models),"id")
cor(all.models[,cor.features])

all.models$unit_sales <- 0.5 * all.models$unit_sales.x + 0.5 * all.models$unit_sales.y

head(all.models)

sub.cols <- c("id","unit_sales")

write.csv(all.models[,sub.cols], "./submissions/Prav_lgb3ref_ref_ensemble.csv",row.names = FALSE, quote = FALSE)


##########################################################################################################################

model01 <- read_csv("./submissions/Prav_mean01_ets.csv")
model01$unit_sales <- ifelse(model01$unit_sales <=1000,model01$unit_sales, 1000)

model02 <- read_csv("./submissions/Prav_nn03.csv")

all.models <- left_join(model01, model02, by="id")
cor.features <- setdiff(names(all.models),"id")


cor(all.models[,cor.features])

all.models$unit_sales <- 0.5 * all.models$unit_sales.x  + 0.5 * all.models$unit_sales.y

head(all.models)
tail(all.models,100)

sub.cols <- c("id","unit_sales")

write.csv(all.models[,sub.cols], "./submissions/Prav_ets_nn03.csv",row.names = FALSE, quote = FALSE)

##########################################################################################################################

model01 <- read_csv("./submissions/Prav_ets_nn03.csv")

model02 <- read_csv("./submissions/Prav_lgbm04.csv")

all.models <- left_join(model01, model02, by="id")
cor.features <- setdiff(names(all.models),"id")


cor(all.models[,cor.features])

all.models$unit_sales <- 0.5 * all.models$unit_sales.x  + 0.5 * all.models$unit_sales.y

head(all.models)

sub.cols <- c("id","unit_sales")

write.csv(all.models[,sub.cols], "./submissions/Prav_etsnn03_lgb04.csv",row.names = FALSE, quote = FALSE)

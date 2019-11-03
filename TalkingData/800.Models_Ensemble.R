rm(list=ls())

setwd("C:/Users/SriPrav/Documents/R/47TalkingData")
root_directory = "C:/Users/SriPrav/Documents/R/47TalkingData"


require.packages <- c(
  "data.table",
  "Matrix",
  "xgboost",
  "sqldf",
  "plyr",
  "dplyr",
  "ROCR",
  "Metrics",
  "pROC",
  "caret",
  "readr",
  "moments",
  "forecast",
  "reshape2",
  "foreach",
  "date",
  "lubridate",
  "ggplot2",
  "ggpmisc",
  "arules",
  "arulesViz",
  "extraTrees",
  "ranger",
  "randomForest",
  "knitr",
  "rmarkdown",
  "rJava",
  "tm",
  "prophet",
  "doParallel",
  "forecastHybrid"
)



##########################################################################################################################
# Function to install and load required packages
##########################################################################################################################
install.missing.packages <- function(x) {
  for (i in x) {
    #  require returns TRUE invisibly if it was able to load package
    if (!require(i , character.only = TRUE)) {
      #  If package was not able to be loaded then re-install
      install.packages(i , dependencies = TRUE)
      #  Load package after installing
      require(i , character.only = TRUE)
    }
  }
}

##########################################################################################################################
# Try function to install and load packages
##########################################################################################################################

install.missing.packages(require.packages)

##########################################################################################################################
# Try function to install and load packages
##########################################################################################################################


lgbm     <- read_csv("./submissions/Prav_lgbm_baseline01.csv")
ftrl <- read_csv("./submissions/wordbatch_fm_ftrl.csv")

head(lgbm); head(ftrl)


all.models <- left_join(lgbm, ftrl, by="click_id")
cor.features <- setdiff(names(all.models),"click_id")
cor(all.models[,cor.features])

cor(rank(all.models$is_attributed.x)
          ,rank(all.models$is_attributed.y)
          )

tmp     <- 100*rank(all.models$is_attributed.x) + 100*rank(all.models$is_attributed.y) 

tmp      <- tmp/max(tmp) 

submission <- data.frame(click_id = all.models$click_id, is_attributed= tmp) 

head(all.models); head(submission)

summary(submission$is_attributed)

write.csv(submission, file="./submissions/Prav_lgbm_ftrl_baseline.csv",row.names=FALSE)


###########################################################################################################################
###########################################################################################################################


lgbm     <- read_csv("./submissions/Prav_lgbm_baseline01.csv")
ftrl <- read_csv("./submissions/Prav_lgbm_01.csv")
lgbm02     <- read_csv("./submissions/Prav_lgbm_02.csv")

head(lgbm); head(ftrl)


all.models <- left_join(lgbm, ftrl, by="click_id")
all.models <- left_join(all.models, lgbm02, by="click_id")

cor.features <- setdiff(names(all.models),"click_id")
cor(all.models[,cor.features])

cor(rank(all.models$is_attributed.x)
    ,rank(all.models$is_attributed.y)
    ,rank(all.models$is_attributed)
)

tmp     <- 100*rank(all.models$is_attributed.x) + 100*rank(all.models$is_attributed.y) + 100*rank(all.models$is_attributed)

tmp      <- tmp/max(tmp) 

submission <- data.frame(click_id = all.models$click_id, is_attributed= tmp) 

head(all.models); head(submission)

summary(submission$is_attributed)

write.csv(submission, file="./submissions/Prav_lgbm_baseline_0102.csv",row.names=FALSE)


###########################################################################################################################
###########################################################################################################################


lgbm     <- read_csv("./submissions/Prav_lgbm_baseline_0102.csv")
ftrl <- read_csv("./submissions/wordbatch_fm_ftrl.csv")

head(lgbm); head(ftrl)


all.models <- left_join(lgbm, ftrl, by="click_id")
cor.features <- setdiff(names(all.models),"click_id")
cor(all.models[,cor.features])

cor(rank(all.models$is_attributed.x)
    ,rank(all.models$is_attributed.y)
)

tmp     <- 100*rank(all.models$is_attributed.x) + 100*rank(all.models$is_attributed.y) 

tmp      <- tmp/max(tmp) 

submission <- data.frame(click_id = all.models$click_id, is_attributed= tmp) 

head(all.models); head(submission)

summary(submission$is_attributed)

write.csv(submission, file="./submissions/Prav_lgbm_baseline_0102_ftrl.csv",row.names=FALSE)


###########################################################################################################################
###########################################################################################################################


###########################################################################################################################
###########################################################################################################################

all.models$is_attributed <- 0.5 * all.models$is_attributed.x + 0.5 * all.models$is_attributed.y

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

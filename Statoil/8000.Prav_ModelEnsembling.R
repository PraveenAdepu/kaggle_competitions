
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
  "tm"
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



fold1 <- read_csv("C:/Users/SriPrav/Documents/R/35Statoil/submissions/Prav.nn01.fold1-test.csv")
fold2 <- read_csv("C:/Users/SriPrav/Documents/R/35Statoil/submissions/Prav.nn01.fold2-test.csv") 
fold3 <- read_csv("C:/Users/SriPrav/Documents/R/35Statoil/submissions/Prav.nn01.fold3-test.csv")
fold4 <- read_csv("C:/Users/SriPrav/Documents/R/35Statoil/submissions/Prav.nn01.fold4-test.csv")

test <- left_join(fold1, fold2, by="id")
test <- left_join(test, fold3, by="id")
test <- left_join(test, fold4, by="id")

test$is_iceberg <- (test$is_iceberg.x + test$is_iceberg.y + test$is_iceberg.x.x + test$is_iceberg.y.y ) /4

cols <- c("id","is_iceberg")

write.csv(test[,cols],"C:/Users/SriPrav/Documents/R/35Statoil/submissions/Prav.nn01.fold1-4-test.csv", row.names = FALSE, quote = FALSE)


model01 <- read_csv("C:/Users/SriPrav/Documents/R/35Statoil/submissions/Prav.xgb001.fold1-4-test.csv") 
model02 <- read_csv("C:/Users/SriPrav/Documents/R/35Statoil/submissions/Prav.nn01.fold1-4-test.csv") 

all_ensemble <- left_join(model01, model02, by = "id")
head(all_ensemble)
ensemble.features <- setdiff(names(all_ensemble),"id")
cor(all_ensemble[, ensemble.features])

all_ensemble$is_iceberg   <- 0.5 * all_ensemble$is_iceberg.x+ 0.5 * all_ensemble$is_iceberg.y

cols <- c("id","is_iceberg")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, 'C:/Users/SriPrav/Documents/R/35Statoil/submissions/Prav_xgb01_nn01_Mean.csv', row.names=FALSE, quote = FALSE)

x2 <- pi * 100^(-1:3)
round(x2, 3)
signif(x2, 3)



fold1 <- read_csv("C:/Users/SriPrav/Documents/R/35Statoil/submissions/Prav.xgb001.fold1-test.csv")
fold2 <- read_csv("C:/Users/SriPrav/Documents/R/35Statoil/submissions/Prav.xgb001.fold2-test.csv") 
fold3 <- read_csv("C:/Users/SriPrav/Documents/R/35Statoil/submissions/Prav.xgb001.fold3-test.csv")
fold4 <- read_csv("C:/Users/SriPrav/Documents/R/35Statoil/submissions/Prav.xgb001.fold4-test.csv")

test <- left_join(fold1, fold2, by="id")
test <- left_join(test, fold3, by="id")
test <- left_join(test, fold4, by="id")

test$is_iceberg <- (test$is_iceberg.x + test$is_iceberg.y + test$is_iceberg.x.x + test$is_iceberg.y.y ) /4

cols <- c("id","is_iceberg")

write.csv(test[,cols],"C:/Users/SriPrav/Documents/R/35Statoil/submissions/Prav.xgb001.fold1-4-test.csv", row.names = FALSE, quote = FALSE)



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

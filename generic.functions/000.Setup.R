


# rm(list=ls())

#source('./Models/000.Setup.R')


##########################################################################################################################
# Select required packages
##########################################################################################################################

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
  "markdown",
  "rJava"
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


###########################################################################################################################
# Metric function forecast validation #####################################################################################
###########################################################################################################################
score <- function(a,b,metric)
  
{    
  switch(metric,           
         accuracy = sum(abs(a-b)<=0.5)/length(a),           
         auc = auc(a,b),           
         logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),           
         mae = sum(abs(a-b))/length(a),           
         precision = length(a[a==b])/length(a),           
         rmse = sqrt(sum((a-b)^2)/length(a)),           
         rmspe = sqrt(sum(((a-b)/a)^2)/length(a)))    
}

metric = "auc"
###########################################################################################################################
# Metric function forecast validation #####################################################################################
###########################################################################################################################




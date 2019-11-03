
rm(list=ls())


require("markdown")
require("rmarkdown")

setwd("C:/Users/SriPrav/Documents/R/40Recruit")
root_directory = "C:/Users/SriPrav/Documents/R/40Recruit"

source("./models/loadPackages.R")

#rmarkdown::render("./TimeSeries.Rmd")

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

metric = "rmse"
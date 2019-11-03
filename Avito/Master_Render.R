
rm(list=ls())

Sys.getlocale(category = "LC_ALL")
#> Sys.getlocale(category = "LC_ALL")
#[1] "LC_COLLATE=English_Australia.1252;LC_CTYPE=English_Australia.1252;LC_MONETARY=English_Australia.1252;LC_NUMERIC=C;LC_TIME=English_Australia.1252"
Sys.setlocale('LC_ALL', 'russian');
#Sys.setlocale('LC_ALL', 'English_Australia');
#[1] "LC_COLLATE=Russian_Russia.1251;LC_CTYPE=Russian_Russia.1251;LC_MONETARY=Russian_Russia.1251;LC_NUMERIC=C;LC_TIME=Russian_Russia.1251"



require("markdown")
require("rmarkdown")

setwd("C:/Users/SriPrav/Documents/R/48Avito")
root_directory = "C:/Users/SriPrav/Documents/R/48Avito"

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
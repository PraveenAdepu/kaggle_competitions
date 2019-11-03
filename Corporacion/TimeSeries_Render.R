
rm(list=ls())


require("markdown")
require("rmarkdown")

setwd("C:/Users/SriPrav/Documents/R/34Corporacion")
root_directory = "C:/Users/SriPrav/Documents/R/34Corporacion"

source("./models/loadPackages.R")

rmarkdown::render("./TimeSeries.Rmd")


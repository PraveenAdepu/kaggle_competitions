
rm(list=ls())

setwd("C:/Users/SriPrav/Documents/R/53Santander")
root_directory = "C:/Users/SriPrav/Documents/R/53Santander"

require("markdown")

rmarkdown::render("./Models/Prav_Santander_EDA.R")
rmarkdown::render("./Models/Prav_Santander_Model05.R")

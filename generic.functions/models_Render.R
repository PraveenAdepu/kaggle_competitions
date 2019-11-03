# rm(list=ls())

require("markdown")

setwd("C:\\Users\\SriPrav\\Documents\\R\\25Brightstar")
root_directory = "C:\\Users\\SriPrav\\Documents\\R\\25Brightstar"

rmarkdown::render("./slot_models.Rmd") 

rmarkdown::render("./repayment_models.Rmd") 


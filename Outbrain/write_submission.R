rm(list = ls(all = T)); gc()

### Needs data.table package
#setwd('directory_of_data')

library(data.table)
#library(ggplot2)

sigmoid <- function(x){
  1 / (1 + exp(-x))
}

### read the raw score from VW
probs_file = 'prob_nn_250.txt'

p_nn = fread(probs_file)$V1

# take a look at the distribution
hist(p_nn)

### read the clicks_test.csv file
clicks_test = fread('clicks_test.csv')
clicks_test[, clicked:=p_nn] # create a new column with p_nn, they are both ordered

### write submission
setkey(clicks_test, "clicked")
submission <- clicks_test[,.(ad_id=paste(rev(ad_id), collapse=" ")), by=display_id]
setkey(submission, "display_id")

write.csv(submission, 
          file = "btb_with_vw.csv",
          row.names = F, quote=FALSE)


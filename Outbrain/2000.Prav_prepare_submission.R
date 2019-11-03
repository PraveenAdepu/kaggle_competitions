submission_proba  <- fread("./submissions/Prav_sub_proba_04.csv") 
names(submission_proba)
head(submission_proba,10)

setorderv( submission_proba, c("display_id","clicked"), c(1,-1)  );gc() #Sort by -prob
head(submission_proba,10)

submission <- submission_proba[,.(ad_id=paste0(ad_id,collapse=" ")), keyby="display_id" ];gc()#Build submission
#6,245,533
head(submission,10)

write.csv(submission,   './submissions/Prav_FTRL_04.csv', row.names=FALSE, quote = FALSE)




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




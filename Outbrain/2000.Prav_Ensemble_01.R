rm(X1, X2, X)

testfile <- read_csv("./input/testingSet12.csv")

idcols <- c("display_id","ad_id")

testingIdCols <- testfile[, idcols]

X1 <- read_csv("./submissions/prav_fullmodel13_ffm.csv", col_names =FALSE) # prav.xgb13.full.csv
X2 <- read_csv("./submissions/prav_fulltrain20_ffm.csv", col_names =FALSE) # submission_keras_02.csv

names(X1)[1] <- "clicked13"
names(X2)[1] <- "clicked20"


submission_proba_ffm <- cbind(testingIdCols, X1,X2)
head(submission_proba_ffm,10)

cor(submission_proba_ffm$clicked13  ,submission_proba_ffm$clicked20     ,method = "pearson")
cor(submission_proba_ffm$clicked13  ,submission_proba_ffm$clicked20      ,method = "spearman")

submission_proba_ffm$clicked <- submission_proba_ffm$clicked12 * 0.5 + submission_proba_ffm$clicked11 * 0.5
head(submission_proba_ffm,10)

submission_proba_ffm$clicked12 <- NULL
submission_proba_ffm$clicked11 <- NULL
head(submission_proba_ffm,10)

submission_proba_ffm <- as.data.table(submission_proba_ffm)

setorderv( submission_proba_ffm, c("display_id","clicked"), c(1,-1)  );gc() #Sort by -prob
head(submission_proba_ffm,10)

submission_ffm <- submission_proba_ffm[,.(ad_id=paste0(ad_id,collapse=" ")), keyby="display_id" ];gc()#Build submission
#6,245,533
head(submission_ffm,10)

write.csv(submission_ffm,   './submissions/Prav_Ensemble_FFM11and12.csv', row.names=FALSE, quote = FALSE)
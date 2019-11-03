
####################################################################################################
## This is feature engineering script

####################################################################################################

train <- read_csv("./input/train.csv")
test  <- read_csv("./input/test.csv")

cvindices <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")

train <- left_join(train, cvindices, by = c("id","qid1","qid2"))


test$qid1         <- 0
test$qid2         <- 0
test$CVindices    <- 0
test$is_duplicate <- -1

names(test)[1] <- "id"

all_data <- rbind(train,test)

head(all_data)

###################################################################################################################

word_not_match <- function(firsttitle1,secondtitle2){

  str          <- gsub("\\?","", tolower(secondtitle2))
  firsttitle1  <- gsub("\\?","", tolower(firsttitle1))
  
  str          <- gsub("[[:punct:]]","", str)
  firsttitle1  <- gsub("[[:punct:]]","", firsttitle1)
  
  firsttitle1  <- unlist(strsplit(firsttitle1," "))
  ntitle1      <- length(firsttitle1)
  
  for(i in 1:length(firsttitle1))
    {
    pattern <- firsttitle1[i]
    pattern  <- paste0(paste0("\\b",pattern),"\\b")
    str = gsub(pattern,"", str)
    }
  
  return(c(ntitle1,str))
}


train_all_words1 <- as.data.frame( t(mapply(word_not_match, all_data$question1,all_data$question2)))
all_data$q2_NotMatching_q1   <- train_all_words1[,2]

train_all_words2 <- as.data.frame( t(mapply(word_not_match,all_data$question2,all_data$question1)))
all_data$q1_NotMatching_q2   <- train_all_words2[,2]

###################################################################################################################
library(qdap)
library(stringr)



train.features <- all_data[all_data$CVindices != 0, ]
test.features  <- all_data[all_data$CVindices == 0, ]


require.features <- c("id", "q2_NotMatching_q1","q1_NotMatching_q2")


train.features <- train.features[, require.features]

train.features$q2_NotMatching_q1 <-  str_trim(clean(train.features$q2_NotMatching_q1))
train.features$q1_NotMatching_q2 <-  str_trim(clean(train.features$q1_NotMatching_q2))


write.csv(train.features,"./input/train_features_20.csv", row.names=F, quote=F, sep=",")


test.features <- test.features[, require.features]
test.features$q2_NotMatching_q1 <-  str_trim(clean(test.features$q2_NotMatching_q1))
test.features$q1_NotMatching_q2 <-  str_trim(clean(test.features$q1_NotMatching_q2))

write.csv(test.features,"./input/test_features_20.csv", row.names=F, quote=F, sep=",")

#####################################################################################################################
train.features$qNonMatch_dist_soundex            <-  stringdist(train.features$q2_NotMatching_q1, train.features$q1_NotMatching_q2, method = c("soundex"))
train.features$qNonMatch_dist_jarowinkler        <- jarowinkler(train.features$q2_NotMatching_q1, train.features$q1_NotMatching_q2)
train.features$qNonMatch_dist_lcs                <-  stringdist(train.features$q2_NotMatching_q1, train.features$q1_NotMatching_q2, method = c("lcs"))
train.features$qNonMatch_dist_lv                 <-  stringdist(train.features$q2_NotMatching_q1, train.features$q1_NotMatching_q2, method = 'lv') 
train.features$qNonMatch_dist_cosine             <-  stringdist(train.features$q2_NotMatching_q1, train.features$q1_NotMatching_q2, method = 'cosine') 
train.features$qNonMatch_2gram_jaccard           <-  stringdist(train.features$q2_NotMatching_q1, train.features$q1_NotMatching_q2, method='jaccard')

test.features$qNonMatch_dist_soundex            <-  stringdist(test.features$q2_NotMatching_q1, test.features$q1_NotMatching_q2, method = c("soundex"))
test.features$qNonMatch_dist_jarowinkler        <- jarowinkler(test.features$q2_NotMatching_q1, test.features$q1_NotMatching_q2)
test.features$qNonMatch_dist_lcs                <-  stringdist(test.features$q2_NotMatching_q1, test.features$q1_NotMatching_q2, method = c("lcs"))
test.features$qNonMatch_dist_lv                 <-  stringdist(test.features$q2_NotMatching_q1, test.features$q1_NotMatching_q2, method = 'lv') 
test.features$qNonMatch_dist_cosine             <-  stringdist(test.features$q2_NotMatching_q1, test.features$q1_NotMatching_q2, method = 'cosine') 
test.features$qNonMatch_2gram_jaccard           <-  stringdist(test.features$q2_NotMatching_q1, test.features$q1_NotMatching_q2, method='jaccard')

measure.features <- c("id", "qNonMatch_dist_soundex","qNonMatch_dist_jarowinkler","qNonMatch_dist_lcs","qNonMatch_dist_lv","qNonMatch_dist_cosine","qNonMatch_2gram_jaccard")

write.csv(train.features[,measure.features],"./input/train_features_21.csv", row.names=F, quote=F, sep=",")

write.csv(test.features[,measure.features],"./input/test_features_21.csv", row.names=F, quote=F, sep=",")


#############################################################################################################

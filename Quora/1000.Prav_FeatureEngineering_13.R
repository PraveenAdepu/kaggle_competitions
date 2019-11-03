train <- read_csv("./input/train.csv")
test  <- read_csv("./input/test.csv")

train$q1qm <- as.numeric(str_count(train$question1, "\\?"))
train$q2qm <- as.numeric(str_count(train$question2, "\\?"))
train$qmatchqm <- as.numeric(train$q1qm == train$q2qm)
train$qratioqm <- ifelse(train$qmatchqm == 1, 1, 
                        1 - abs(train$q1qm - train$q2qm)/(train$q1qm + train$q2qm))

train$q1math <- as.numeric(str_count(train$question1, "\\[math]"))
train$q2math <- as.numeric(str_count(train$question2, "\\[math]"))
train$qmatchmath <- as.numeric(train$q1math == train$q2math)
train$qratiomath <- ifelse(train$qmatchmath == 1, 1, 
                          1 - abs(train$q1math - train$q2math)/(train$q1math + train$q2math))

train$q1stop <- as.numeric(str_count(train$question1, "\\."))
train$q2stop <- as.numeric(str_count(train$question2, "\\."))
train$qmatchstop <- as.numeric(train$q1stop == train$q2stop)
train$qratiostop <- ifelse(train$qmatchstop == 1, 1, 
                          1 - abs(train$q1stop - train$q2stop)/(train$q1stop + train$q2stop))

train$q1num <- as.numeric(str_count(train$question1, "\\d+"))
train$q2num <- as.numeric(str_count(train$question2, "\\d+"))
train$qmatchnum <- as.numeric(train$q1num == train$q2num)
train$qrationum <- ifelse(train$qmatchnum == 1, 1, 
                         1 - abs(train$q1num - train$q2num)/(train$q1num + train$q2num))

train$q1what <- as.numeric(str_count(tolower(train$question1), "\\what"))
train$q2what <- as.numeric(str_count(tolower(train$question2), "\\what"))
train$qmatchwhat <- as.numeric(train$q1what == train$q2what)
train$qratiowhat <- ifelse(train$qmatchwhat == 1, 1, 
                          1 - abs(train$q1what - train$q2what)/(train$q1what + train$q2what))

train$q1when <- as.numeric(str_count(tolower(train$question1), "\\when"))
train$q2when <- as.numeric(str_count(tolower(train$question2), "\\when"))
train$qmatchwhen <- as.numeric(train$q1when == train$q2when)
train$qratiowhen <- ifelse(train$qmatchwhen == 1, 1, 
                           1 - abs(train$q1when - train$q2when)/(train$q1when + train$q2when))
train$q1why <- as.numeric(str_count(tolower(train$question1), "\\why"))
train$q2why <- as.numeric(str_count(tolower(train$question2), "\\why"))
train$qmatchwhy <- as.numeric(train$q1why == train$q2why)
train$qratiowhy <- ifelse(train$qmatchwhy == 1, 1, 
                          1 - abs(train$q1why - train$q2why)/(train$q1why + train$q2why))
train$q1how <- as.numeric(str_count(tolower(train$question1), "\\how"))
train$q2how <- as.numeric(str_count(tolower(train$question2), "\\how"))
train$qmatchhow <- as.numeric(train$q1how == train$q2how)
train$qratiohow <- ifelse(train$qmatchhow == 1, 1, 
                          1 - abs(train$q1how - train$q2how)/(train$q1how + train$q2how))
#########################################################################################################
test$q1qm <- as.numeric(str_count(test$question1, "\\?"))
test$q2qm <- as.numeric(str_count(test$question2, "\\?"))
test$qmatchqm <- as.numeric(test$q1qm == test$q2qm)
test$qratioqm <- ifelse(test$qmatchqm == 1, 1, 
                        1 - abs(test$q1qm - test$q2qm)/(test$q1qm + test$q2qm))

test$q1math <- as.numeric(str_count(test$question1, "\\[math]"))
test$q2math <- as.numeric(str_count(test$question2, "\\[math]"))
test$qmatchmath <- as.numeric(test$q1math == test$q2math)
test$qratiomath <- ifelse(test$qmatchmath == 1, 1, 
                          1 - abs(test$q1math - test$q2math)/(test$q1math + test$q2math))

test$q1stop <- as.numeric(str_count(test$question1, "\\."))
test$q2stop <- as.numeric(str_count(test$question2, "\\."))
test$qmatchstop <- as.numeric(test$q1stop == test$q2stop)
test$qratiostop <- ifelse(test$qmatchstop == 1, 1, 
                          1 - abs(test$q1stop - test$q2stop)/(test$q1stop + test$q2stop))

test$q1num <- as.numeric(str_count(test$question1, "\\d+"))
test$q2num <- as.numeric(str_count(test$question2, "\\d+"))
test$qmatchnum <- as.numeric(test$q1num == test$q2num)
test$qrationum <- ifelse(test$qmatchnum == 1, 1, 
                         1 - abs(test$q1num - test$q2num)/(test$q1num + test$q2num))

test$q1what <- as.numeric(str_count(tolower(test$question1), "\\what"))
test$q2what <- as.numeric(str_count(tolower(test$question2), "\\what"))
test$qmatchwhat <- as.numeric(test$q1what == test$q2what)
test$qratiowhat <- ifelse(test$qmatchwhat == 1, 1, 
                          1 - abs(test$q1what - test$q2what)/(test$q1what + test$q2what))

test$q1when <- as.numeric(str_count(tolower(test$question1), "\\when"))
test$q2when <- as.numeric(str_count(tolower(test$question2), "\\when"))
test$qmatchwhen <- as.numeric(test$q1when == test$q2when)
test$qratiowhen <- ifelse(test$qmatchwhen == 1, 1, 
                          1 - abs(test$q1when - test$q2when)/(test$q1when + test$q2when))
test$q1why <- as.numeric(str_count(tolower(test$question1), "\\why"))
test$q2why <- as.numeric(str_count(tolower(test$question2), "\\why"))
test$qmatchwhy <- as.numeric(test$q1why == test$q2why)
test$qratiowhy <- ifelse(test$qmatchwhy == 1, 1, 
                         1 - abs(test$q1why - test$q2why)/(test$q1why + test$q2why))
test$q1how <- as.numeric(str_count(tolower(test$question1), "\\how"))
test$q2how <- as.numeric(str_count(tolower(test$question2), "\\how"))
test$qmatchhow <- as.numeric(test$q1how == test$q2how)
test$qratiohow <- ifelse(test$qmatchhow == 1, 1, 
                         1 - abs(test$q1how - test$q2how)/(test$q1how + test$q2how))
##################################################################################################
head(train$question1,10)
head(train$q1what,10)

features <- c("q1qm","q2qm","qmatchqm","qratiomath","q1stop","q2stop","qmatchstop","qratiostop","q1num"
              ,"q2num","qmatchnum","qrationum","q1what","q2what","qmatchwhat","qratiowhat"
              ,"q1when","q2when","qmatchwhen","qratiowhen","q1why","q2why","qmatchwhy","qratiowhy"
              ,"q1how","q2how","qmatchhow","qratiohow")

for (i in features) {
  cat("feature : ", i,"\n")
  cat("AUC : " ,score(train$is_duplicate, train[[i]] , metric))
  
}

#################################################################################################



train.sentiment_question1 <- get_nrc_sentiment(train$question1)
train.sentiment_question2 <- get_nrc_sentiment(train$question2)

test.sentiment_question1 <- get_nrc_sentiment(test$question1)
test.sentiment_question2 <- get_nrc_sentiment(test$question2)

colnames(train.sentiment_question1) <- paste("q1", colnames(train.sentiment_question1), sep = "_")
colnames(train.sentiment_question2) <- paste("q2", colnames(train.sentiment_question2), sep = "_")

colnames(test.sentiment_question1) <- paste("q1", colnames(test.sentiment_question1), sep = "_")
colnames(test.sentiment_question2) <- paste("q2", colnames(test.sentiment_question2), sep = "_")

train <- cbind(train, train.sentiment_question1, train.sentiment_question2)
test  <- cbind(test, test.sentiment_question1, test.sentiment_question2)


excludeSet <- c("qid1","qid2","question1","question2","is_duplicate" )
featureSet <- setdiff(names(train),excludeSet)

write.csv(train[,featureSet],  './input/train_features_53.csv', row.names=FALSE, quote = FALSE)

featureSet <- setdiff(names(test),excludeSet)
write.csv(test[,featureSet],  './input/test_features_53.csv', row.names=FALSE, quote = FALSE)


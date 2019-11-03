
Sys.getlocale(category = "LC_ALL")
#> Sys.getlocale(category = "LC_ALL")
#[1] "LC_COLLATE=English_Australia.1252;LC_CTYPE=English_Australia.1252;LC_MONETARY=English_Australia.1252;LC_NUMERIC=C;LC_TIME=English_Australia.1252"
Sys.setlocale('LC_ALL', 'russian');
#Sys.setlocale('LC_ALL', 'English_Australia');
#[1] "LC_COLLATE=Russian_Russia.1251;LC_CTYPE=Russian_Russia.1251;LC_MONETARY=Russian_Russia.1251;LC_NUMERIC=C;LC_TIME=Russian_Russia.1251"


train <- read_csv("./input/train.csv")
test  <- read_csv("./input/test.csv")


names(train)
names(test)

# title , description

train$title_nchar         <- nchar(train$title)
train$description_nchar   <- nchar(train$description)

test$title_nchar         <- nchar(test$title)
test$description_nchar   <- nchar(test$description)

summary(train$title_nchar)
summary(train$description_nchar)

head(train$title)

head(subset(train[c("title","description")], is.na(train$description_nchar)))

require(stringdist)

train$titlediscription_distance_lcs     <- stringdist(train$title, train$description, method = "lcs")
train$titlediscription_distance_jaccard <- stringdist(train$title, train$description, method = "jaccard")
train$titlediscription_distance_jw      <- stringdist(train$title, train$description, method = "jw")
train$titlediscription_distance_sound   <- stringdist(train$title, train$description, method = "soundex")
train$titlediscription_distance_lv      <- stringdist(train$title, train$description, method = "lv")
train$titlediscription_distance_cosine  <- stringdist(train$title, train$description, method = "cosine")

test$titlediscription_distance_lcs     <- stringdist(test$title, test$description, method = "lcs")
test$titlediscription_distance_jaccard <- stringdist(test$title, test$description, method = "jaccard")
test$titlediscription_distance_jw      <- stringdist(test$title, test$description, method = "jw")
test$titlediscription_distance_sound   <- stringdist(test$title, test$description, method = "soundex")
test$titlediscription_distance_lv      <- stringdist(test$title, test$description, method = "lv")
test$titlediscription_distance_cosine  <- stringdist(test$title, test$description, method = "cosine")


# stringdist(a, b, method = c("osa", "lv", "dl", "hamming", "lcs", "qgram",
#                             "cosine", "jaccard", "jw", "soundex"))

require("stringr")
require("stringi")

train$titleEndsWithdescription        <- ifelse(stri_extract_last_words(train$title) == stri_extract_last_words(train$description), 1, 0 )
train$titleStartsWithdescription      <- ifelse(stri_extract_first_words(train$title) == stri_extract_first_words(train$description), 1, 0 )
train$titleStartsWithdescriptionEnd   <- ifelse(stri_extract_first_words(train$title) == stri_extract_last_words(train$description), 1, 0 )
train$titleEndsWithdescriptionStart   <- ifelse(stri_extract_last_words(train$title) == stri_extract_first_words(train$description), 1, 0 )


test$titleEndsWithdescription        <- ifelse(stri_extract_last_words(test$title) == stri_extract_last_words(test$description), 1, 0 )
test$titleStartsWithdescription      <- ifelse(stri_extract_first_words(test$title) == stri_extract_first_words(test$description), 1, 0 )
test$titleStartsWithdescriptionEnd   <- ifelse(stri_extract_first_words(test$title) == stri_extract_last_words(test$description), 1, 0 )
test$titleEndsWithdescriptionStart   <- ifelse(stri_extract_last_words(test$title) == stri_extract_first_words(test$description), 1, 0 )


train$titleEndsWithdescriptionSound     <- stringdist(stri_extract_last_words(train$title)  , stri_extract_last_words(train$description),   method='soundex' )
train$titleStartsWithdescriptionSound   <- stringdist(stri_extract_first_words(train$title) , stri_extract_first_words(train$description),  method='soundex' )

test$titleEndsWithdescriptionSound     <- stringdist(stri_extract_last_words(test$title)  , stri_extract_last_words(test$description),   method='soundex' )
test$titleStartsWithdescriptionSound   <- stringdist(stri_extract_first_words(test$title) , stri_extract_first_words(test$description),  method='soundex' )


###############################################################################################################################################################

require(gbm)
require(readr)
require(xgboost)
require(car)
require(dplyr)
require(caret)
require(randomForest)
require(stringdist)
require(sqldf)
require(RecordLinkage)
require(e1071)
require(tm)
require(RTextTools)
require(SnowballC) 
require(parallel)  
require(tau)       
require(stringr)
require(data.table)
require(stringi)
require(geosphere)
require(plyr)
require(data.table)
require(splitstackshape)
require(ngram)

word_match <- function(firsttitle1,secondtitle2){
  n_title      <- 0
  firsttitle1  <- unlist(strsplit(firsttitle1," "))
  ntitle1 <- length(firsttitle1)
  #if(ntitle1 > 0) {
  for(i in 1:length(firsttitle1)){
    
    #pattern <- paste("(^| )",firsttitle1[i],"($| )",sep="")
    pattern     <- firsttitle1[i]
    n_title     <- n_title  + ifelse(grepl(pattern, secondtitle2,fixed = TRUE,ignore.case=TRUE)>= 1, 1,0 )
    
  }
  
  return(c(ntitle1,n_title))
}

word_strength <- function(firsttitle1,secondtitle2){
  n_title      <- 0
  firsttitle1  <- unlist(strsplit(firsttitle1," "))
  ntitle1 <- length(firsttitle1)
  #if(ntitle1 > 0) {
  for(i in 1:length(firsttitle1)){
    
    #pattern <- paste("(^| )",firsttitle1[i],"($| )",sep="")
    pattern     <- firsttitle1[i]
    n_title     <- n_title  + ifelse(jarowinkler(pattern,secondtitle2)>=0.3,  1  , 0 )
    
  }
  
  return(c(ntitle1,n_title))
}

stem_text<- function(text, language = 'ru', mc.cores = 1) {
  # stem each word in a block of text
  stem_string <- function(str, language) {
    str <- tokenize(x = str)
    str <- wordStem(str, language = language)
    str <- paste(str, collapse = "")
    return(str)
  }
  
  # stem each text block in turn
  x <- mclapply(X = text, FUN = stem_string, language, mc.cores = mc.cores)
  
  # return stemed text blocks
  return(unlist(x))
}

train$title       <- stem_text(train$title, language = 'ru', mc.cores = 1)
train$description <- stem_text(train$description, language = 'ru', mc.cores = 1)

test$title       <- stem_text(test$title, language = 'ru', mc.cores = 1)
test$description <- stem_text(test$description, language = 'ru', mc.cores = 1)


train_all_words <- as.data.frame( t(mapply(word_match,train$title,train$description)))
train$titleWordsMatchedinDescription  <- train_all_words[,2]

test_all_words <- as.data.frame( t(mapply(word_match,test$title,test$description)))
test$titleWordsMatchedinDescription  <- test_all_words[,2]


train_all_words <- as.data.frame( t(mapply(word_strength,train$title,train$description)))
train$titleWordsMatchedStrengthinDescription  <- train_all_words[,2]

test_all_words <- as.data.frame( t(mapply(word_strength,test$title,test$description)))
test$titleWordsMatchedStrengthinDescription  <- test_all_words[,2]


train$titledescription_jaccard_2gram  <- stringdist(train$title, train$description, method='jaccard', q=2)
train$titledescription_jaccard_3gram  <- stringdist(train$title, train$description, method='jaccard', q=3)
train$titledescription_jaccard_4gram  <- stringdist(train$title, train$description, method='jaccard', q=4)
train$titledescription_jaccard_5gram  <- stringdist(train$title, train$description, method='jaccard', q=5)

test$titledescription_jaccard_2gram  <- stringdist(test$title, test$description, method='jaccard', q=2)
test$titledescription_jaccard_3gram  <- stringdist(test$title, test$description, method='jaccard', q=3)
test$titledescription_jaccard_4gram  <- stringdist(test$title, test$description, method='jaccard', q=4)
test$titledescription_jaccard_5gram  <- stringdist(test$title, test$description, method='jaccard', q=5)

allTwoword_match <- function(firsttitle1,secondtitle2){
  wordmatch         <- 0
  wordscombinations <- 0
  unlistfirsttitle1  <- unlist(strsplit(firsttitle1," "))
  ntitle1 <- length(unlistfirsttitle1)
  #if(ntitle1 > 0) {
  for(i in 2){
    ng1 <- ngram_asweka(firsttitle1, min = i, max = i, sep = " ")
    ng2 <- ngram_asweka(secondtitle2, min = i, max = i, sep = " ")
    wordmatch <- wordmatch + length(which(ng1 %in% ng2))
    
  }
  
  return(c(ntitle1,wordmatch))
}

train_allTwoword_match <- as.data.frame( t(mapply(allTwoword_match,train$title,train$description)))

train$TitleTwoWordsMatchedinDescription   <- train_allTwoword_match[,2]

test_allTwoword_match <- as.data.frame( t(mapply(allTwoword_match,test$title,test$description)))

test$TitleTwoWordsMatchedinDescription   <- test_allTwoword_match[,2]

WordPunctCount <- function(firsttitle1,secondtitle2){
  SpecialCharCount1  <- 0
  SpecialCharCount2  <- 0
  {
    SpecialCharCount1  <- str_count(firsttitle1) - str_count(str_replace_all(firsttitle1, "[[:punct:]]", ""))
    SpecialCharCount2  <- str_count(secondtitle2) - str_count(str_replace_all(secondtitle2, "[[:punct:]]", ""))
  }
  return(c(SpecialCharCount1,SpecialCharCount2 ))
}

train_WordPunct <- as.data.frame( t(mapply(WordPunctCount,train$title,train$description)))
train$titlePunctCount       <-  train_WordPunct[,1]
train$descriptionPunctCount <-  train_WordPunct[,2]

test_WordPunct <- as.data.frame( t(mapply(WordPunctCount,test$title,test$description)))
test$titlePunctCount       <-  test_WordPunct[,1]
test$descriptionPunctCount <-  test_WordPunct[,2]

rm(test_all_words, test_allTwoword_match, test_WordPunct, train_all_words, train_allTwoword_match, train_WordPunct); gc()

######################################################################################################################################

library(qdap)

CountEnglishWords <- function(inuptString1,inuptString2){
  inuptString1 <- paste("a " , inuptString1)
  inuptString2 <- paste("a " , inuptString2)
  EnglishWordsCount1  <- 0
  EnglishWordsCount2  <- 0
  {
    EnglishWordsCount1  <-   sum( word_count(str_extract_all(str_replace_all(inuptString1, "[[:punct:]]", ""), "[a-z]+"))) - 1
    EnglishWordsCount2  <-   sum( word_count(str_extract_all(str_replace_all(inuptString2, "[[:punct:]]", ""), "[a-z]+"))) - 1
  }
  return(c(EnglishWordsCount1,EnglishWordsCount2 ))
}

# validate function results with mix of both languages
train_CountEnglishWords  <- as.data.frame( t(mapply(CountEnglishWords,train$title,train$description)))

train$title_EnglishWords       <-  train_CountEnglishWords[,1] 
train$description_EnglishWords <-  train_CountEnglishWords[,2]


test_CountEnglishWords  <- as.data.frame( t(mapply(CountEnglishWords,test$title,test$description)))

test$title_EnglishWords       <-  test_CountEnglishWords[,1] 
test$description_EnglishWords <-  test_CountEnglishWords[,2]

###################################################################################################################################
# write features to csv files for models

derived.features <- c("item_id","title_nchar"                 ,  "description_nchar"                     , "titlediscription_distance_lcs"         
                      ,"titlediscription_distance_jaccard"    ,  "titlediscription_distance_jw"          , "titlediscription_distance_sound"       
                      ,"titleEndsWithdescription"             ,  "titleStartsWithdescription"            , "titleStartsWithdescriptionEnd"         
                      ,"titleEndsWithdescriptionStart"        ,  "titleEndsWithdescriptionSound"         , "titleStartsWithdescriptionSound"       
                      ,"titleWordsMatchedinDescription"       ,  "titleWordsMatchedStrengthinDescription", "titledescription_jaccard_2gram"        
                      ,"titledescription_jaccard_3gram"       ,  "titledescription_jaccard_4gram"        , "titledescription_jaccard_5gram"        
                      ,"titlediscription_distance_lv"         ,  "titlediscription_distance_cosine"      , "TitleTwoWordsMatchedinDescription"     
                      ,"titlePunctCount"                      ,  "descriptionPunctCount"                 , "title_EnglishWords"                    
                      ,"description_EnglishWords")

train_features <- train[,derived.features]
test_features  <- test[,derived.features]

train_features[is.na(train_features)] <- -99
test_features[is.na(test_features)]   <- -99


train_features$titledescription_jaccard_2gram <- ifelse(train_features$titledescription_jaccard_2gram == Inf, -99, train_features$titledescription_jaccard_2gram)
train_features$titledescription_jaccard_3gram <- ifelse(train_features$titledescription_jaccard_3gram == Inf, -99, train_features$titledescription_jaccard_3gram)
train_features$titledescription_jaccard_4gram <- ifelse(train_features$titledescription_jaccard_4gram == Inf, -99, train_features$titledescription_jaccard_4gram)
train_features$titledescription_jaccard_5gram <- ifelse(train_features$titledescription_jaccard_5gram == Inf, -99, train_features$titledescription_jaccard_5gram)

test_features$titledescription_jaccard_2gram <- ifelse(test_features$titledescription_jaccard_2gram == Inf, -99, test_features$titledescription_jaccard_2gram)
test_features$titledescription_jaccard_3gram <- ifelse(test_features$titledescription_jaccard_3gram == Inf, -99, test_features$titledescription_jaccard_3gram)
test_features$titledescription_jaccard_4gram <- ifelse(test_features$titledescription_jaccard_4gram == Inf, -99, test_features$titledescription_jaccard_4gram)
test_features$titledescription_jaccard_5gram <- ifelse(test_features$titledescription_jaccard_5gram == Inf, -99, test_features$titledescription_jaccard_5gram)


summary(train_features)
summary(test_features)


write.csv(train_features, './input/Prav_train_FE_01.csv', row.names=FALSE, quote = FALSE)
write.csv(test_features, './input/Prav_test_FE_01.csv'  , row.names=FALSE, quote = FALSE)

rm(train_CountEnglishWords, test_CountEnglishWords); gc()

###################################################################################################################################

# tf-idf feature engineering

library(tidyverse)
library(lubridate)
library(magrittr)n
library(text2vec)
library(tokenizers)
library(stopwords)
library(xgboost)
library(Matrix)


features <- c("item_id","title","description")



tr <- train[,features]
te  <- test[,features]

cat("Preprocessing...\n")
tri <- 1:nrow(tr)

tr_te <- tr %>% 
  bind_rows(te) %>%
  mutate(txt = paste(title, description, sep = " ")
        )

rm(tr, te); gc()

#---------------------------
cat("Parsing text...\n")
it <- tr_te %$%
  str_to_lower(txt) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  tokenize_word_stems(language = "russian") %>% 
  itoken()

vect <- create_vocabulary(it, ngram = c(1, 1), stopwords = stopwords("ru")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.4, vocab_term_max = 6500) %>% 
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <-  create_dtm(it, vect) %>% 
  fit_transform(m_tfidf)

rm(it, vect, m_tfidf); gc()

dim(tfidf)



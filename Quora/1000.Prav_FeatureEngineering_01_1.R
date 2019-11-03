
####################################################################################################
## This is feature engineering script

####################################################################################################
# Read data clean files which prepared from python feature engineering file
####################################################################################################

train <- read_csv("./input/train_qs_clean.csv")
test  <- read_csv("./input/test_qs_clean.csv")
####################################################################################################
# Use Porterstem questions for this feature extraction
####################################################################################################

train$lemmatizer_question1 <- NULL
train$lemmatizer_question2 <- NULL

test$lemmatizer_question1 <- NULL
test$lemmatizer_question2 <- NULL
####################################################################################################

names(train)[2] <- "question1"
names(train)[3] <- "question2"

names(test)[2] <- "question1"
names(test)[3] <- "question2"

###################################################################################################


cvindices <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")

cvindices$qid1 <- NULL
cvindices$qid2 <- NULL

train <- left_join(train, cvindices, by = c("id"))

test$CVindices    <- 0


names(test)[1] <- "id"

names(train)
names(test)

all_data <- rbind(train,test)

head(all_data)

###################################################################################################################

# all_data$q_dist_hamming            <- stringdist(all_data$question1, all_data$question2, method = c("hamming")) 
# summary(all_data$q_dist_hamming)
# all_data$q_dist_hamming  <- NULL # Not good

all_data$q_dist_soundex            <- stringdist(all_data$question1, all_data$question2, method = c("soundex"))
summary(all_data$q_dist_soundex)

all_data$q_dist_jarowinkler        <- jarowinkler(all_data$question1, all_data$question2)
summary(all_data$q_dist_jarowinkler)

all_data$q_dist_lcs            <- stringdist(all_data$question1, all_data$question2, method = c("lcs"))
summary(all_data$q_dist_lcs)

all_data$q1_nchar          <- nchar(all_data$question1)
all_data$q2_nchar          <- nchar(all_data$question2)
summary(all_data$q1_nchar)
summary(all_data$q2_nchar)

all_data$q1_EndsWith_q2     <- ifelse(stri_extract_last_words(all_data$question1) == stri_extract_last_words(all_data$question2), 1, 0 )
summary(all_data$q1_EndsWith_q2)

all_data$q1_EndsWith_Sound_q2     <- stringdist(stri_extract_last_words(all_data$question1)  , stri_extract_last_words(all_data$question2),   method='soundex' )
summary(all_data$q1_EndsWith_Sound_q2)

all_data$q1_StartsWith_q2           <- ifelse(stri_extract_first_words(all_data$question1) == stri_extract_first_words(all_data$question2), 1, 0 )
all_data$q1_StartsWith_Sound_q2     <- stringdist(stri_extract_first_words(all_data$question1)  , stri_extract_first_words(all_data$question2),   method='soundex' )
summary(all_data$q1_StartsWith_q2)
summary(all_data$q1_StartsWith_Sound_q2)


# temp <- all_data[all_data$CVindices != 0, ]
# 
# auc(temp$is_duplicate, temp$q_dist_soundex)
# # auc(temp$is_duplicate, temp$q_dist_jarowinkler) # long time to score
# auc(temp$is_duplicate, temp$q_dist_lcs)
# auc(temp$is_duplicate, temp$q1_nchar)
# auc(temp$is_duplicate, temp$q2_nchar)
# auc(temp$is_duplicate, temp$q1_EndsWith_q2)
# auc(temp$is_duplicate, temp$q1_EndsWith_Sound_q2)
# auc(temp$is_duplicate, temp$q1_StartsWith_q2)
# auc(temp$is_duplicate, temp$q1_StartsWith_Sound_q2)

all_data$q_nchar_ratios_pmax      <- pmax(all_data$q1_nchar/ all_data$q2_nchar, all_data$q2_nchar/all_data$q1_nchar)
all_data$q_nchar_pmin             <- pmin(all_data$q1_nchar, all_data$q2_nchar, na.rm=TRUE)
all_data$q_nchar_pmax             <- pmax(all_data$q1_nchar, all_data$q2_nchar, na.rm=TRUE)

summary(all_data$q_nchar_ratios_pmax)
summary(all_data$q_nchar_pmin)
summary(all_data$q_nchar_pmax)

# temp <- all_data[all_data$CVindices != 0, ]
# 
# auc(temp$is_duplicate, temp$q_nchar_ratios_pmax)
# auc(temp$is_duplicate, temp$q_nchar_pmin)
# auc(temp$is_duplicate, temp$q_nchar_pmax)




###################################################################################################################
###################################################################################################################


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

stem_text<- function(text, language = 'en', mc.cores = 1) {
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

###################################################################################################################

# head(all_data$question1)
# head(all_data$question1_stem_text)
# all_data$question1_stem_text       <- stem_text(all_data$question1, language = 'en', mc.cores = 1)
# all_data$question2_stem_text       <- stem_text(all_data$question2, language = 'en', mc.cores = 1)

# Error in wordStem(str, language = language) : 
#   There is a limit of 255characters on the number of characters in a word being stemmed 
###################################################################################################################

train_all_words <- as.data.frame( t(mapply(word_match,all_data$question1,all_data$question2)))
all_data$q1_nwords              <- train_all_words[,1]
all_data$q1_nwords_matched_q2   <- train_all_words[,2]

all_data$q1_MatchedWords_ratio_to_q2 <- ifelse(all_data$q1_nwords == 0 , 0 , all_data$q1_nwords_matched_q2/all_data$q1_nwords)


# temp <- all_data[all_data$CVindices != 0, ]
# 
# auc(temp$is_duplicate, temp$q1_nwords)
# auc(temp$is_duplicate, temp$q1_nwords_matched_q2)
# auc(temp$is_duplicate, temp$q1_MatchedWords_ratio_to_q2)

train_all_words <- as.data.frame( t(mapply(word_match,all_data$question2,all_data$question1)))
all_data$q2_nwords              <- train_all_words[,1]
all_data$q2_nwords_matched_q1   <- train_all_words[,2]

all_data$q2_MatchedWords_ratio_to_q1 <- ifelse(all_data$q2_nwords == 0 , 0 , all_data$q2_nwords_matched_q1/all_data$q2_nwords)

# temp <- all_data[all_data$CVindices != 0, ]
# 
# auc(temp$is_duplicate, temp$q2_nwords)
# auc(temp$is_duplicate, temp$q2_nwords_matched_q1)
# auc(temp$is_duplicate, temp$q2_MatchedWords_ratio_to_q1)


all_data$q_MatchedWords_ratio_to_q_ratios_pmax      <- pmax(all_data$q1_MatchedWords_ratio_to_q2/ all_data$q2_MatchedWords_ratio_to_q1, all_data$q2_MatchedWords_ratio_to_q1/all_data$q1_MatchedWords_ratio_to_q2)
all_data$q_MatchedWords_ratio_to_q_pmin             <- pmin(all_data$q1_MatchedWords_ratio_to_q2, all_data$q2_MatchedWords_ratio_to_q1, na.rm=TRUE)
all_data$q_MatchedWords_ratio_to_q_pmax             <- pmax(all_data$q1_MatchedWords_ratio_to_q2, all_data$q2_MatchedWords_ratio_to_q1, na.rm=TRUE)

# temp <- all_data[all_data$CVindices != 0, ]
# 
# auc(temp$is_duplicate, temp$q_MatchedWords_ratio_to_q_ratios_pmax)
# auc(temp$is_duplicate, temp$q_MatchedWords_ratio_to_q_pmin)
# auc(temp$is_duplicate, temp$q_MatchedWords_ratio_to_q_pmax)


###################################################################################################################
###################################################################################################################

all_data$q_2gram_jaccard  <- stringdist(all_data$question1,all_data$question2, method='jaccard', q=2)
all_data$q_3gram_jaccard  <- stringdist(all_data$question1,all_data$question2, method='jaccard', q=3)
all_data$q_4gram_jaccard  <- stringdist(all_data$question1,all_data$question2, method='jaccard', q=4)
all_data$q_5gram_jaccard  <- stringdist(all_data$question1,all_data$question2, method='jaccard', q=5)


# temp <- all_data[all_data$CVindices != 0, ]
# 
# auc(temp$is_duplicate, temp$q_2gram_jaccard)
# auc(temp$is_duplicate, temp$q_3gram_jaccard)
# auc(temp$is_duplicate, temp$q_4gram_jaccard)
# auc(temp$is_duplicate, temp$q_5gram_jaccard)


###################################################################################################################
###################################################################################################################

# allTwoword_match <- function(firsttitle1,secondtitle2){
#   wordmatch         <- 0
#   wordscombinations <- 0
#   unlistfirsttitle1  <- unlist(strsplit(firsttitle1," "))
#   ntitle1 <- length(unlistfirsttitle1)
#   if(ntitle1 > 0) {
#   for(i in 2){
#     ng1 <- ngram_asweka(firsttitle1, min = i, max = i, sep = " ")
#     ng2 <- ngram_asweka(secondtitle2, min = i, max = i, sep = " ")
#     wordmatch <- wordmatch + length(which(ng1 %in% ng2))
#     
#   }
#   }
#   
#   return(c(ntitle1,wordmatch))
# }
# 
# train_allTwoword_match <- as.data.frame( t(mapply(allTwoword_match, all_data$question1,all_data$question2)))
# 
# all_data$q1_2pairwords_matched_q2   <- train_allTwoword_match[,2]
# all_data$q1_2pairwords_matched_q2_ratio <- ifelse((all_data$q1_nwords -1) == 0, 0 , all_data$q1_pairwords_matched_q2/(all_data$q1_nwords -1))




###################################################################################################################
###################################################################################################################



all_data$q_dist_lv     <-  stringdist(all_data$question1,all_data$question2, method = 'lv') #'cosine'
all_data$q_dist_cosine <-  stringdist(all_data$question1,all_data$question2, method = 'cosine') #'cosine'

# temp <- all_data[all_data$CVindices != 0, ]
# 
# auc(temp$is_duplicate, temp$q_dist_lv)
# auc(temp$is_duplicate, temp$q_dist_cosine)
##############################################################################################
##############################################################################################


WordPunctCount <- function(firsttitle1,secondtitle2){
  SpecialCharCount1  <- 0
  SpecialCharCount2  <- 0
  {
    SpecialCharCount1  <- str_count(firsttitle1) - str_count(str_replace_all(firsttitle1, "[[:punct:]]", ""))
    SpecialCharCount2  <- str_count(secondtitle2) - str_count(str_replace_all(secondtitle2, "[[:punct:]]", ""))
  }
  return(c(SpecialCharCount1,SpecialCharCount2 ))
}

train_WordPunct <- as.data.frame( t(mapply(WordPunctCount,all_data$question1,all_data$question2)))
all_data$q1_PunctCount <-  train_WordPunct[,1]
all_data$q2_PunctCount <-  train_WordPunct[,2]


all_data$q_PunctCount_ratios_pmax      <- pmax(all_data$q1_PunctCount/ all_data$q2_PunctCount, all_data$q2_PunctCount/all_data$q1_PunctCount)
all_data$q_PunctCount_pmin             <- pmin(all_data$q1_PunctCount, all_data$q2_PunctCount, na.rm=TRUE)
all_data$q_PunctCount_pmax             <- pmax(all_data$q1_PunctCount, all_data$q2_PunctCount, na.rm=TRUE)

# temp <- all_data[all_data$CVindices != 0, ]
# 
# auc(temp$is_duplicate, temp$q1_PunctCount)
# auc(temp$is_duplicate, temp$q2_PunctCount)
# auc(temp$is_duplicate, temp$q_PunctCount_ratios_pmax)
# auc(temp$is_duplicate, temp$q_PunctCount_pmin)
# auc(temp$is_duplicate, temp$q_PunctCount_pmax)


##############################################################################################
##############################################################################################

train.features <- all_data[all_data$CVindices != 0, ]
test.features  <- all_data[all_data$CVindices == 0, ]

non.features <- c("qid1","qid2", "question1","question2","is_duplicate","CVindices")

train.feature.names <- setdiff(names(train.features), non.features)
train.features <- train.features[, train.feature.names]

write.csv(train.features,"./input/train_features_01_1.csv", row.names=F)

test.non.features <- c("qid1","qid2", "question1","question2","is_duplicate","CVindices")

test.feature.names <- setdiff(names(test.features), test.non.features)
test.features <- test.features[, test.feature.names]

colnames(test.features)[which(names(test.features) == "id")] <- "test_id"

write.csv(test.features,"./input/test_features_01_1.csv", row.names=F)

head(train.features)
head(test.features)

##############################################################################################
##############################################################################################
gc()
#######################################################
Sys.time()
save.image(file = "Quora_01.RData" , safe = TRUE)
Sys.time()
# load("Quora_01.RData")
# Sys.time()
########################################################


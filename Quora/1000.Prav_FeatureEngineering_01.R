
####################################################################################################
## This is feature engineering script

####################################################################################################



train <- read_csv("./input/train.csv")
test  <- read_csv("./input/test.csv")

cvindices <- read_csv("./CVSchema/CVindices_5folds.csv")

train <- left_join(train, cvindices, by = c("id","qid1","qid2"))


test$qid1         <- 0
test$qid2         <- 0
test$CVindices    <- 0
test$is_duplicate <- -1

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


temp <- all_data[all_data$CVindices != 0, ]

auc(temp$is_duplicate, temp$q_dist_soundex)
# auc(temp$is_duplicate, temp$q_dist_jarowinkler) # long time to score
auc(temp$is_duplicate, temp$q_dist_lcs)
auc(temp$is_duplicate, temp$q1_nchar)
auc(temp$is_duplicate, temp$q2_nchar)
auc(temp$is_duplicate, temp$q1_EndsWith_q2)
auc(temp$is_duplicate, temp$q1_EndsWith_Sound_q2)
auc(temp$is_duplicate, temp$q1_StartsWith_q2)
auc(temp$is_duplicate, temp$q1_StartsWith_Sound_q2)

all_data$q_nchar_ratios_pmax      <- pmax(all_data$q1_nchar/ all_data$q2_nchar, all_data$q2_nchar/all_data$q1_nchar)
all_data$q_nchar_pmin             <- pmin(all_data$q1_nchar, all_data$q2_nchar, na.rm=TRUE)
all_data$q_nchar_pmax             <- pmax(all_data$q1_nchar, all_data$q2_nchar, na.rm=TRUE)

summary(all_data$q_nchar_ratios_pmax)
summary(all_data$q_nchar_pmin)
summary(all_data$q_nchar_pmax)

temp <- all_data[all_data$CVindices != 0, ]

auc(temp$is_duplicate, temp$q_nchar_ratios_pmax)
auc(temp$is_duplicate, temp$q_nchar_pmin)
auc(temp$is_duplicate, temp$q_nchar_pmax)




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


temp <- all_data[all_data$CVindices != 0, ]

auc(temp$is_duplicate, temp$q1_nwords)
auc(temp$is_duplicate, temp$q1_nwords_matched_q2)
auc(temp$is_duplicate, temp$q1_MatchedWords_ratio_to_q2)

train_all_words <- as.data.frame( t(mapply(word_match,all_data$question2,all_data$question1)))
all_data$q2_nwords              <- train_all_words[,1]
all_data$q2_nwords_matched_q1   <- train_all_words[,2]

all_data$q2_MatchedWords_ratio_to_q1 <- ifelse(all_data$q2_nwords == 0 , 0 , all_data$q2_nwords_matched_q1/all_data$q2_nwords)

temp <- all_data[all_data$CVindices != 0, ]

auc(temp$is_duplicate, temp$q2_nwords)
auc(temp$is_duplicate, temp$q2_nwords_matched_q1)
auc(temp$is_duplicate, temp$q2_MatchedWords_ratio_to_q1)


all_data$q_MatchedWords_ratio_to_q_ratios_pmax      <- pmax(all_data$q1_MatchedWords_ratio_to_q2/ all_data$q2_MatchedWords_ratio_to_q1, all_data$q2_MatchedWords_ratio_to_q1/all_data$q1_MatchedWords_ratio_to_q2)
all_data$q_MatchedWords_ratio_to_q_pmin             <- pmin(all_data$q1_MatchedWords_ratio_to_q2, all_data$q2_MatchedWords_ratio_to_q1, na.rm=TRUE)
all_data$q_MatchedWords_ratio_to_q_pmax             <- pmax(all_data$q1_MatchedWords_ratio_to_q2, all_data$q2_MatchedWords_ratio_to_q1, na.rm=TRUE)

temp <- all_data[all_data$CVindices != 0, ]

auc(temp$is_duplicate, temp$q_MatchedWords_ratio_to_q_ratios_pmax)
auc(temp$is_duplicate, temp$q_MatchedWords_ratio_to_q_pmin)
auc(temp$is_duplicate, temp$q_MatchedWords_ratio_to_q_pmax)


###################################################################################################################
###################################################################################################################

all_data$q_2gram_jaccard  <- stringdist(all_data$question1,all_data$question2, method='jaccard', q=2)
all_data$q_3gram_jaccard  <- stringdist(all_data$question1,all_data$question2, method='jaccard', q=3)
all_data$q_4gram_jaccard  <- stringdist(all_data$question1,all_data$question2, method='jaccard', q=4)
all_data$q_5gram_jaccard  <- stringdist(all_data$question1,all_data$question2, method='jaccard', q=5)


temp <- all_data[all_data$CVindices != 0, ]

auc(temp$is_duplicate, temp$q_2gram_jaccard)
auc(temp$is_duplicate, temp$q_3gram_jaccard)
auc(temp$is_duplicate, temp$q_4gram_jaccard)
auc(temp$is_duplicate, temp$q_5gram_jaccard)


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

temp <- all_data[all_data$CVindices != 0, ]

auc(temp$is_duplicate, temp$q_dist_lv)
auc(temp$is_duplicate, temp$q_dist_cosine)
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

temp <- all_data[all_data$CVindices != 0, ]

auc(temp$is_duplicate, temp$q1_PunctCount)
auc(temp$is_duplicate, temp$q2_PunctCount)
auc(temp$is_duplicate, temp$q_PunctCount_ratios_pmax)
auc(temp$is_duplicate, temp$q_PunctCount_pmin)
auc(temp$is_duplicate, temp$q_PunctCount_pmax)


##############################################################################################
##############################################################################################

train.features <- all_data[all_data$CVindices != 0, ]
test.features  <- all_data[all_data$CVindices == 0, ]

non.features <- c( "question1","question2","is_duplicate","CVindices")

train.feature.names <- setdiff(names(train.features), non.features)
train.features <- train.features[, train.feature.names]

write.csv(train.features,"./input/train_features_01.csv", row.names=F, quote=F, sep=",")

test.non.features <- c("qid1","qid2", "question1","question2","is_duplicate","CVindices")

test.feature.names <- setdiff(names(test.features), test.non.features)
test.features <- test.features[, test.feature.names]
write.csv(test.features,"./input/test_features_01.csv", row.names=F, quote=F, sep=",")


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

head(all_data$question1); head(all_data$question2); head(all_data$is_duplicate)

head(all_data$q_dist_soundex)
head(all_data$q_2gram_jaccard)
head(all_data$q_5gram_jaccard)
head(all_data$q_dist_lcs)
head(all_data$q1_MatchedWords_ratio_to_q2)

##########################################################################################################

trainItemPairs$sameLat          <- as.numeric(trainItemPairs$lat1 == trainItemPairs$lat2)
trainItemPairs$sameLon          <- as.numeric(trainItemPairs$lon1 == trainItemPairs$lon2)
trainItemPairs$sameLocation     <- as.numeric(trainItemPairs$locationID1 == trainItemPairs$locationID2)
trainItemPairs$sameregion       <- as.numeric(trainItemPairs$regionID1 == trainItemPairs$regionID2)
trainItemPairs$priceDifference        <- (trainItemPairs$price1 - trainItemPairs$price2)
trainItemPairs$sameTitle        <- as.numeric(trainItemPairs$title1 == trainItemPairs$title2)
trainItemPairs$sameDescription  <- as.numeric(trainItemPairs$description1 == trainItemPairs$description2)
trainItemPairs$imageArrayDiff   <- as.numeric(trainItemPairs$immages_array1_count - trainItemPairs$immages_array2_count)

trainItemPairs$SimilarityTitle        <- jarowinkler(trainItemPairs$title1,trainItemPairs$title2)
trainItemPairs$SimilarityDescription  <- jarowinkler(trainItemPairs$description1,trainItemPairs$description2)

trainItemPairs$samemetro  <- as.numeric(trainItemPairs$metroID1 == trainItemPairs$metroID2)
trainItemPairs$SimilarityJSON        <- jarowinkler(trainItemPairs$attrsJSON1,trainItemPairs$attrsJSON2)
trainItemPairs$distance = sqrt((trainItemPairs$lat1-trainItemPairs$lat2)^2+(trainItemPairs$lon1-trainItemPairs$lon2)^2)

trainItemPairs$nchartitle1          <- nchar(trainItemPairs$title1)
trainItemPairs$nchartitle2          <- nchar(trainItemPairs$title2)
trainItemPairs$nchardescription1    <- nchar(trainItemPairs$description1)
trainItemPairs$nchardescription2    <- nchar(trainItemPairs$description2)
trainItemPairs$priceDiff            <- pmax(trainItemPairs$price1/trainItemPairs$price2, trainItemPairs$price2/trainItemPairs$price1)

trainItemPairs$priceMin             <- pmin(trainItemPairs$price1, trainItemPairs$price2, na.rm=TRUE)
trainItemPairs$priceMax             <- pmax(trainItemPairs$price1, trainItemPairs$price2, na.rm=TRUE)
trainItemPairs$titleStringDistance2 <- (stringdist(trainItemPairs$title1, trainItemPairs$title2, method = "lcs") / 
                                          pmax(trainItemPairs$nchartitle1, trainItemPairs$nchartitle2, na.rm=TRUE))

trainItemPairs$title1StartsWithTitle2 <- as.numeric(substr(trainItemPairs$title1, 1, nchar(trainItemPairs$title2)) == trainItemPairs$title2)
trainItemPairs$title2StartsWithTitle1 <- as.numeric(substr(trainItemPairs$title2, 1, nchar(trainItemPairs$title1)) == trainItemPairs$title1)

trainItemPairs$titleCharDiff      <- pmax(trainItemPairs$nchartitle1/ trainItemPairs$nchartitle2, trainItemPairs$nchartitle2/trainItemPairs$nchartitle1)
trainItemPairs$titleCharMin       <- pmin(trainItemPairs$nchartitle1, trainItemPairs$nchartitle2, na.rm=TRUE)
trainItemPairs$titleCharMax       <- pmax(trainItemPairs$nchartitle1, trainItemPairs$nchartitle2, na.rm=TRUE)

trainItemPairs$descriptionCharDiff <- pmax(trainItemPairs$nchardescription1/ trainItemPairs$nchardescription2, trainItemPairs$nchardescription2/trainItemPairs$nchardescription1)
trainItemPairs$descriptionCharMin  <- pmin(trainItemPairs$nchardescription1, trainItemPairs$nchardescription2, na.rm=TRUE)
trainItemPairs$descriptionCharMax  <- pmax(trainItemPairs$nchardescription1, trainItemPairs$nchardescription2, na.rm=TRUE)

trainItemPairs$title1EndsWithTitle2     <- ifelse(stri_extract_last_words(trainItemPairs$title1) == stri_extract_last_words(trainItemPairs$title2), 1, 0 )
trainItemPairs$title1StartingWithTitle2 <- ifelse(stri_extract_first_words(trainItemPairs$title1) == stri_extract_first_words(trainItemPairs$title2), 1, 0 )

trainItemPairs$title2EndsWithTitle1     <- ifelse(stri_extract_last_words(trainItemPairs$title2) == stri_extract_last_words(trainItemPairs$title1), 1, 0 )
trainItemPairs$title2StartingWithTitle1 <- ifelse(stri_extract_first_words(trainItemPairs$title2) == stri_extract_first_words(trainItemPairs$title1), 1, 0 )


trainItemPairs$title1SoundexWithTitle2     <- stringdist(trainItemPairs$title1,trainItemPairs$title2,method='soundex')

trainItemPairs$title1EndSoundexWithTitle2     <- stringdist(stri_extract_last_words(trainItemPairs$title1)  , stri_extract_last_words(trainItemPairs$title2),   method='soundex' )
trainItemPairs$title1StartSoundexWithTitle2   <- stringdist(stri_extract_first_words(trainItemPairs$title1) , stri_extract_first_words(trainItemPairs$title2) , method='soundex' )


testItemPairs$sameLat         <- as.numeric(testItemPairs$lat1 == testItemPairs$lat2)
testItemPairs$sameLon         <- as.numeric(testItemPairs$lon1 == testItemPairs$lon2)
testItemPairs$sameLocation    <- as.numeric(testItemPairs$locationID1 == testItemPairs$locationID2)
testItemPairs$sameregion      <- as.numeric(testItemPairs$regionID1 == testItemPairs$regionID2)
testItemPairs$priceDifference       <- (testItemPairs$price1 - testItemPairs$price2)
testItemPairs$sameTitle       <- as.numeric(testItemPairs$title1 == testItemPairs$title2)
testItemPairs$sameDescription <- as.numeric(testItemPairs$description1 == testItemPairs$description2)
testItemPairs$imageArrayDiff   <- as.numeric(testItemPairs$immages_array1_count - testItemPairs$immages_array2_count)

testItemPairs$SimilarityTitle        <- jarowinkler(testItemPairs$title1,testItemPairs$title2)
testItemPairs$SimilarityDescription  <- jarowinkler(testItemPairs$description1,testItemPairs$description2)

testItemPairs$samemetro  <- as.numeric(testItemPairs$metroID1 == testItemPairs$metroID2)
testItemPairs$SimilarityJSON        <- jarowinkler(testItemPairs$attrsJSON1,testItemPairs$attrsJSON2)
testItemPairs$distance = sqrt((testItemPairs$lat1-testItemPairs$lat2)^2+(testItemPairs$lon1-testItemPairs$lon2)^2)

testItemPairs$nchartitle1          <- nchar(testItemPairs$title1)
testItemPairs$nchartitle2          <- nchar(testItemPairs$title2)
testItemPairs$nchardescription1    <- nchar(testItemPairs$description1)
testItemPairs$nchardescription2    <- nchar(testItemPairs$description2)
testItemPairs$nchartitle1          <- nchar(testItemPairs$title1)
testItemPairs$nchartitle2          <- nchar(testItemPairs$title2)
testItemPairs$nchardescription1    <- nchar(testItemPairs$description1)
testItemPairs$nchardescription2    <- nchar(testItemPairs$description2)
testItemPairs$priceDiff            <- pmax(testItemPairs$price1/testItemPairs$price2, testItemPairs$price2/testItemPairs$price1)
testItemPairs$priceMin             <- pmin(testItemPairs$price1, testItemPairs$price2, na.rm=TRUE)
testItemPairs$priceMax             <- pmax(testItemPairs$price1, testItemPairs$price2, na.rm=TRUE)
testItemPairs$titleStringDistance2 <- (stringdist(testItemPairs$title1, testItemPairs$title2, method = "lcs") / 
                                         pmax(testItemPairs$nchartitle1, testItemPairs$nchartitle2, na.rm=TRUE))

testItemPairs$title1StartsWithTitle2 <- as.numeric(substr(testItemPairs$title1, 1, nchar(testItemPairs$title2)) == testItemPairs$title2)
testItemPairs$title2StartsWithTitle1 <- as.numeric(substr(testItemPairs$title2, 1, nchar(testItemPairs$title1)) == testItemPairs$title1)

testItemPairs$titleCharDiff       <- pmax(testItemPairs$nchartitle1/ testItemPairs$nchartitle2, testItemPairs$nchartitle2/testItemPairs$nchartitle1)
testItemPairs$titleCharMin        <- pmin(testItemPairs$nchartitle1, testItemPairs$nchartitle2, na.rm=TRUE)
testItemPairs$titleCharMax        <- pmax(testItemPairs$nchartitle1, testItemPairs$nchartitle2, na.rm=TRUE)

testItemPairs$descriptionCharDiff <- pmax(testItemPairs$nchardescription1/ testItemPairs$nchardescription2, testItemPairs$nchardescription2/testItemPairs$nchardescription1)
testItemPairs$descriptionCharMin  <- pmin(testItemPairs$nchardescription1, testItemPairs$nchardescription2, na.rm=TRUE)
testItemPairs$descriptionCharMax  <- pmax(testItemPairs$nchardescription1, testItemPairs$nchardescription2, na.rm=TRUE)

testItemPairs$title1EndsWithTitle2     <- ifelse(stri_extract_last_words(testItemPairs$title1) == stri_extract_last_words(testItemPairs$title2), 1, 0 )
testItemPairs$title1StartingWithTitle2 <- ifelse(stri_extract_first_words(testItemPairs$title1) == stri_extract_first_words(testItemPairs$title2), 1, 0 )
testItemPairs$title1SoundexWithTitle2  <- stringdist(testItemPairs$title1,testItemPairs$title2,method='soundex')

testItemPairs$title2EndsWithTitle1     <-  ifelse(stri_extract_last_words(testItemPairs$title2) ==  stri_extract_last_words(testItemPairs$title1), 1, 0 )
testItemPairs$title2StartingWithTitle1 <- ifelse(stri_extract_first_words(testItemPairs$title2) == stri_extract_first_words(testItemPairs$title1), 1, 0 )

testItemPairs$title1EndSoundexWithTitle2     <- stringdist(stri_extract_last_words(testItemPairs$title1)  , stri_extract_last_words(testItemPairs$title2),   method='soundex' )
testItemPairs$title1StartSoundexWithTitle2   <- stringdist(stri_extract_first_words(testItemPairs$title1) , stri_extract_first_words(testItemPairs$title2) , method='soundex' )

trainItemPairs$priceDiff            <- ifelse(is.na(trainItemPairs$priceDiff) | trainItemPairs$priceDiff == Inf, -9999, trainItemPairs$priceDiff)
trainItemPairs$titleStringDistance2 <- ifelse(is.na(trainItemPairs$titleStringDistance2) | trainItemPairs$titleStringDistance2 == Inf, -9999, trainItemPairs$titleStringDistance2)
trainItemPairs$titleCharDiff        <- ifelse(is.na(trainItemPairs$titleCharDiff)| trainItemPairs$titleCharDiff == Inf, -9999, trainItemPairs$titleCharDiff)
trainItemPairs$descriptionCharDiff  <- ifelse(is.na(trainItemPairs$descriptionCharDiff)| trainItemPairs$descriptionCharDiff == Inf, -9999, trainItemPairs$descriptionCharDiff)

testItemPairs$priceDiff            <- ifelse(is.na(testItemPairs$priceDiff) | testItemPairs$priceDiff == Inf, -9999, testItemPairs$priceDiff)
testItemPairs$titleStringDistance2 <- ifelse(is.na(testItemPairs$titleStringDistance2) | testItemPairs$titleStringDistance2 == Inf, -9999, testItemPairs$titleStringDistance2)
testItemPairs$titleCharDiff        <- ifelse(is.na(testItemPairs$titleCharDiff)| testItemPairs$titleCharDiff == Inf, -9999, testItemPairs$titleCharDiff)
testItemPairs$descriptionCharDiff  <- ifelse(is.na(testItemPairs$descriptionCharDiff)| testItemPairs$descriptionCharDiff == Inf, -9999, testItemPairs$descriptionCharDiff)

###################################################################################################

###################################################################################################


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

trainItemPairs$title1       <- stem_text(trainItemPairs$title1, language = 'ru', mc.cores = 1)
trainItemPairs$title2       <- stem_text(trainItemPairs$title2, language = 'ru', mc.cores = 1)
trainItemPairs$description1 <- stem_text(trainItemPairs$description1, language = 'ru', mc.cores = 1)
trainItemPairs$description2 <- stem_text(trainItemPairs$description2, language = 'ru', mc.cores = 1)

testItemPairs$title1       <- stem_text(testItemPairs$title1, language = 'ru', mc.cores = 1)
testItemPairs$title2       <- stem_text(testItemPairs$title2, language = 'ru', mc.cores = 1)
testItemPairs$description1 <- stem_text(testItemPairs$description1, language = 'ru', mc.cores = 1)
testItemPairs$description2 <- stem_text(testItemPairs$description2, language = 'ru', mc.cores = 1)


train_all_words <- as.data.frame( t(mapply(word_match,trainItemPairs$title1,trainItemPairs$title2)))
trainItemPairs$Title1Length                 <- train_all_words[,1]
trainItemPairs$Title1WordsMatchedinTitle2   <- train_all_words[,2]
trainItemPairs$Title1WordsMatchedinTitle2ratio <- ifelse(trainItemPairs$Title1Length == 0 , 0 , trainItemPairs$Title1WordsMatchedinTitle2/trainItemPairs$Title1Length)

train_all_words <- as.data.frame( t(mapply(word_match,trainItemPairs$title2,trainItemPairs$title1)))
trainItemPairs$Title2Length                 <- train_all_words[,1]
trainItemPairs$Title2WordsMatchedinTitle1   <- train_all_words[,2]
trainItemPairs$Title2WordsMatchedinTitle1ratio <- ifelse(trainItemPairs$Title2Length == 0 , 0 , trainItemPairs$Title2WordsMatchedinTitle1/trainItemPairs$Title2Length)



train_all_words <- as.data.frame( t(mapply(word_match,trainItemPairs$title1,trainItemPairs$description2)))
trainItemPairs$Title1InDescLength           <- train_all_words[,1]
trainItemPairs$Title1WordsMatchedinDesc2    <- train_all_words[,2]
trainItemPairs$Title1WordsMatchedinDesc2ratio <- ifelse(trainItemPairs$Title1InDescLength == 0 , 0 , trainItemPairs$Title1WordsMatchedinDesc2/trainItemPairs$Title1InDescLength)


train_all_words <- as.data.frame( t(mapply(word_match,trainItemPairs$title2,trainItemPairs$description1)))
trainItemPairs$Title2InDescLength           <- train_all_words[,1]
trainItemPairs$Title2WordsMatchedinDesc1    <- train_all_words[,2]
trainItemPairs$Title2WordsMatchedinDesc1ratio <- ifelse(trainItemPairs$Title2InDescLength == 0 , 0 , trainItemPairs$Title2WordsMatchedinDesc1/trainItemPairs$Title2InDescLength)

train_all_words <- as.data.frame( t(mapply(word_match,trainItemPairs$attrsJSON1,trainItemPairs$attrsJSON2)))
trainItemPairs$attrsJSON1Length                      <- train_all_words[,1]
trainItemPairs$attrsJSON1WordsMatchedinattrsJSON2    <- train_all_words[,2]
trainItemPairs$attrsJSON1WordsMatchedinattrsJSON2ratio        <- ifelse(trainItemPairs$attrsJSON1Length == 0 , 0 , trainItemPairs$attrsJSON1WordsMatchedinattrsJSON2/trainItemPairs$attrsJSON1Length)

trainItemPairs$attrsJSON1WordsMatchedinattrsJSON2ratio <- round(trainItemPairs$attrsJSON1WordsMatchedinattrsJSON2ratio,2)



trainItemPairs$Title1WordsMatchedinTitle2ratio <- round(trainItemPairs$Title1WordsMatchedinTitle2ratio,2)
trainItemPairs$Title2WordsMatchedinTitle1ratio <- round(trainItemPairs$Title2WordsMatchedinTitle1ratio,2)

trainItemPairs$Title1WordsMatchedinDesc2ratio  <- round(trainItemPairs$Title1WordsMatchedinDesc2ratio,2)
trainItemPairs$Title2WordsMatchedinDesc1ratio  <- round(trainItemPairs$Title2WordsMatchedinDesc1ratio,2)
# trainItemPairs$Title1WordsToTitle2MatchRatio <- ifelse(trainItemPairs$Title1Length == 0, 0, trainItemPairs$Title1WordsMatchedinTitle2/trainItemPairs$Title1Length)
# 
test_all_words <- as.data.frame(t(mapply(word_match,testItemPairs$title1,testItemPairs$title2)))
testItemPairs$Title1Length                 <- test_all_words[,1]
testItemPairs$Title1WordsMatchedinTitle2   <- test_all_words[,2]
testItemPairs$Title1WordsMatchedinTitle2ratio <- ifelse(testItemPairs$Title1Length == 0 , 0 , testItemPairs$Title1WordsMatchedinTitle2/testItemPairs$Title1Length)

test_all_words <- as.data.frame(t(mapply(word_match,testItemPairs$title2,testItemPairs$title1)))
testItemPairs$Title2Length                 <- test_all_words[,1]
testItemPairs$Title2WordsMatchedinTitle1   <- test_all_words[,2]
testItemPairs$Title2WordsMatchedinTitle1ratio <- ifelse(testItemPairs$Title2Length == 0 , 0 , testItemPairs$Title2WordsMatchedinTitle1/testItemPairs$Title2Length)

test_all_words <- as.data.frame( t(mapply(word_match,testItemPairs$title1,testItemPairs$description2)))
testItemPairs$Title1InDescLength           <- test_all_words[,1]
testItemPairs$Title1WordsMatchedinDesc2    <- test_all_words[,2]
testItemPairs$Title1WordsMatchedinDesc2ratio <- ifelse(testItemPairs$Title1InDescLength == 0 , 0 , testItemPairs$Title1WordsMatchedinDesc2/testItemPairs$Title1InDescLength)


test_all_words <- as.data.frame( t(mapply(word_match,testItemPairs$title2,testItemPairs$description1)))
testItemPairs$Title2InDescLength           <- test_all_words[,1]
testItemPairs$Title2WordsMatchedinDesc1    <- test_all_words[,2]
testItemPairs$Title2WordsMatchedinDesc1ratio <- ifelse(testItemPairs$Title2InDescLength == 0 , 0 , testItemPairs$Title2WordsMatchedinDesc1/testItemPairs$Title2InDescLength)

testItemPairs$Title1WordsMatchedinTitle2ratio <- round(testItemPairs$Title1WordsMatchedinTitle2ratio,2)
testItemPairs$Title2WordsMatchedinTitle1ratio <- round(testItemPairs$Title2WordsMatchedinTitle1ratio,2)

testItemPairs$Title1WordsMatchedinDesc2ratio  <- round(testItemPairs$Title1WordsMatchedinDesc2ratio,2)
testItemPairs$Title2WordsMatchedinDesc1ratio  <- round(testItemPairs$Title2WordsMatchedinDesc1ratio,2)

train_all_words <- as.data.frame( t(mapply(word_match,trainItemPairs$description1,trainItemPairs$description2)))
trainItemPairs$description1Length                 <- train_all_words[,1]
trainItemPairs$description1WordsMatchedinDesc2    <- train_all_words[,2]
trainItemPairs$description1WordsMatchedinDesc2ratio <- ifelse(trainItemPairs$description1Length == 0 , 0 , trainItemPairs$description1WordsMatchedinDesc2/trainItemPairs$description1Length)

trainItemPairs$description1WordsMatchedinDesc2ratio <- round(trainItemPairs$description1WordsMatchedinDesc2ratio , 2)

test_all_words <- as.data.frame( t(mapply(word_match,testItemPairs$description1,testItemPairs$description2)))
testItemPairs$description1Length                 <- test_all_words[,1]
testItemPairs$description1WordsMatchedinDesc2    <- test_all_words[,2]
testItemPairs$description1WordsMatchedinDesc2ratio <- ifelse(testItemPairs$description1Length == 0 , 0 , testItemPairs$description1WordsMatchedinDesc2/testItemPairs$description1Length)

testItemPairs$description1WordsMatchedinDesc2ratio <- round(testItemPairs$description1WordsMatchedinDesc2ratio , 2)

test_all_words <- as.data.frame( t(mapply(word_match,testItemPairs$attrsJSON1,testItemPairs$attrsJSON2)))
testItemPairs$attrsJSON1Length                      <- test_all_words[,1]
testItemPairs$attrsJSON1WordsMatchedinattrsJSON2    <- test_all_words[,2]
testItemPairs$attrsJSON1WordsMatchedinattrsJSON2ratio        <- ifelse(testItemPairs$attrsJSON1Length == 0 , 0 , testItemPairs$attrsJSON1WordsMatchedinattrsJSON2/testItemPairs$attrsJSON1Length)

testItemPairs$attrsJSON1WordsMatchedinattrsJSON2ratio <- round(testItemPairs$attrsJSON1WordsMatchedinattrsJSON2ratio,2)

trainItemPairs$Title1WordsMatchedinTitle2ratioDiff      <- pmax(trainItemPairs$Title1WordsMatchedinTitle2ratio/ trainItemPairs$Title2WordsMatchedinTitle1ratio, trainItemPairs$Title2WordsMatchedinTitle1ratio/trainItemPairs$Title1WordsMatchedinTitle2ratio)
trainItemPairs$Title1WordsMatchedinTitle2ratioMin       <- pmin(trainItemPairs$Title1WordsMatchedinTitle2, trainItemPairs$Title2WordsMatchedinTitle1, na.rm=TRUE)
trainItemPairs$Title1WordsMatchedinTitle2ratioMax       <- pmax(trainItemPairs$Title1WordsMatchedinTitle2, trainItemPairs$Title2WordsMatchedinTitle1, na.rm=TRUE)

trainItemPairs$Title1WordsMatchedinDesc2ratioDiff      <- pmax(trainItemPairs$Title1WordsMatchedinDesc2ratio/ trainItemPairs$Title2WordsMatchedinDesc1ratio, trainItemPairs$Title2WordsMatchedinDesc1ratio/trainItemPairs$Title1WordsMatchedinDesc2ratio)
trainItemPairs$Title1WordsMatchedinDesc2ratioMin       <- pmin(trainItemPairs$Title1WordsMatchedinDesc2, trainItemPairs$Title2WordsMatchedinDesc1, na.rm=TRUE)
trainItemPairs$Title1WordsMatchedinDesc2ratioMax       <- pmax(trainItemPairs$Title1WordsMatchedinDesc2, trainItemPairs$Title2WordsMatchedinDesc1, na.rm=TRUE)

trainItemPairs$Title1WordsMatchedinTitle2Diff      <- pmax(trainItemPairs$Title1WordsMatchedinTitle2/ trainItemPairs$Title2WordsMatchedinTitle1, trainItemPairs$Title2WordsMatchedinTitle1/trainItemPairs$Title1WordsMatchedinTitle2)
trainItemPairs$Title1WordsMatchedinDesc2Diff       <- pmax(trainItemPairs$Title1WordsMatchedinDesc2/ trainItemPairs$Title2WordsMatchedinDesc1, trainItemPairs$Title2WordsMatchedinDesc1/trainItemPairs$Title1WordsMatchedinDesc2)

testItemPairs$Title1WordsMatchedinTitle2Diff      <- pmax(testItemPairs$Title1WordsMatchedinTitle2/ testItemPairs$Title2WordsMatchedinTitle1, testItemPairs$Title2WordsMatchedinTitle1/testItemPairs$Title1WordsMatchedinTitle2)
testItemPairs$Title1WordsMatchedinDesc2Diff       <- pmax(testItemPairs$Title1WordsMatchedinDesc2/  testItemPairs$Title2WordsMatchedinDesc1,  testItemPairs$Title2WordsMatchedinDesc1/ testItemPairs$Title1WordsMatchedinDesc2)


testItemPairs$Title1WordsMatchedinDesc2ratioDiff      <- pmax(testItemPairs$Title1WordsMatchedinDesc2ratio/ testItemPairs$Title2WordsMatchedinDesc1ratio, testItemPairs$Title2WordsMatchedinDesc1ratio/testItemPairs$Title1WordsMatchedinDesc2ratio)
testItemPairs$Title1WordsMatchedinDesc2ratioMin       <- pmin(testItemPairs$Title1WordsMatchedinDesc2, testItemPairs$Title2WordsMatchedinDesc1, na.rm=TRUE)
testItemPairs$Title1WordsMatchedinDesc2ratioMax       <- pmax(testItemPairs$Title1WordsMatchedinDesc2, testItemPairs$Title2WordsMatchedinDesc1, na.rm=TRUE)

testItemPairs$Title1WordsMatchedinTitle2ratioDiff      <- pmax(testItemPairs$Title1WordsMatchedinTitle2ratio/ testItemPairs$Title2WordsMatchedinTitle1ratio, testItemPairs$Title2WordsMatchedinTitle1ratio/testItemPairs$Title1WordsMatchedinTitle2ratio)
testItemPairs$Title1WordsMatchedinTitle2ratioMin       <- pmin(testItemPairs$Title1WordsMatchedinTitle2, testItemPairs$Title2WordsMatchedinTitle1, na.rm=TRUE)
testItemPairs$Title1WordsMatchedinTitle2ratioMax       <- pmax(testItemPairs$Title1WordsMatchedinTitle2, testItemPairs$Title2WordsMatchedinTitle1, na.rm=TRUE)

trainItemPairs$Title1WordsMatchedinTitle2ratioDiff    <- ifelse(is.na(trainItemPairs$Title1WordsMatchedinTitle2ratioDiff) | trainItemPairs$Title1WordsMatchedinTitle2ratioDiff == Inf, -9999, trainItemPairs$Title1WordsMatchedinTitle2ratioDiff)
trainItemPairs$Title1WordsMatchedinDesc2ratioDiff     <- ifelse(is.na(trainItemPairs$Title1WordsMatchedinDesc2ratioDiff) | trainItemPairs$Title1WordsMatchedinDesc2ratioDiff == Inf, -9999, trainItemPairs$Title1WordsMatchedinDesc2ratioDiff)
trainItemPairs$Title1WordsMatchedinTitle2Diff         <- ifelse(is.na(trainItemPairs$Title1WordsMatchedinTitle2Diff) | trainItemPairs$Title1WordsMatchedinTitle2Diff == Inf, -9999, trainItemPairs$Title1WordsMatchedinTitle2Diff)
trainItemPairs$Title1WordsMatchedinDesc2Diff          <- ifelse(is.na(trainItemPairs$Title1WordsMatchedinDesc2Diff) | trainItemPairs$Title1WordsMatchedinDesc2Diff == Inf, -9999, trainItemPairs$Title1WordsMatchedinDesc2Diff)

testItemPairs$Title1WordsMatchedinTitle2ratioDiff     <- ifelse(is.na(testItemPairs$Title1WordsMatchedinTitle2ratioDiff) | testItemPairs$Title1WordsMatchedinTitle2ratioDiff == Inf, -9999, testItemPairs$Title1WordsMatchedinTitle2ratioDiff)
testItemPairs$Title1WordsMatchedinDesc2ratioDiff      <- ifelse(is.na(testItemPairs$Title1WordsMatchedinDesc2ratioDiff) | testItemPairs$Title1WordsMatchedinDesc2ratioDiff == Inf, -9999, testItemPairs$Title1WordsMatchedinDesc2ratioDiff)
testItemPairs$Title1WordsMatchedinTitle2Diff          <- ifelse(is.na(testItemPairs$Title1WordsMatchedinTitle2Diff) | testItemPairs$Title1WordsMatchedinTitle2Diff == Inf, -9999, testItemPairs$Title1WordsMatchedinTitle2Diff)
testItemPairs$Title1WordsMatchedinDesc2Diff           <- ifelse(is.na(testItemPairs$Title1WordsMatchedinDesc2Diff) | testItemPairs$Title1WordsMatchedinDesc2Diff == Inf, -9999, testItemPairs$Title1WordsMatchedinDesc2Diff)


train_all_wordsstrength <- as.data.frame( t(mapply(word_strength,trainItemPairs$title1,trainItemPairs$title2)))
trainItemPairs$Title1LengthStrength             <- train_all_wordsstrength[,1]
trainItemPairs$Title1WordsStrengthinTitle2      <- train_all_wordsstrength[,2]
trainItemPairs$Title1WordsStrengthinTitle2ratio <- ifelse(trainItemPairs$Title1LengthStrength == 0 , 0 , trainItemPairs$Title1WordsStrengthinTitle2/trainItemPairs$Title1LengthStrength)

test_all_wordsstrength <- as.data.frame( t(mapply(word_strength,testItemPairs$title1,testItemPairs$title2)))
testItemPairs$Title1LengthStrength             <- test_all_wordsstrength[,1]
testItemPairs$Title1WordsStrengthinTitle2      <- test_all_wordsstrength[,2]
testItemPairs$Title1WordsStrengthinTitle2ratio <- ifelse(testItemPairs$Title1LengthStrength == 0 , 0 , testItemPairs$Title1WordsStrengthinTitle2/testItemPairs$Title1LengthStrength)
###########################################################################################################


trainItemPairs$title12gramtitle2  <- stringdist(trainItemPairs$title1, trainItemPairs$title2, method='jaccard', q=2)
testItemPairs$title12gramtitle2  <- stringdist(testItemPairs$title1, testItemPairs$title2, method='jaccard', q=2)

trainItemPairs$title12gramtitle2 <- ifelse(is.na(trainItemPairs$title12gramtitle2) | trainItemPairs$title12gramtitle2 == Inf, -9999, trainItemPairs$title12gramtitle2)
testItemPairs$title12gramtitle2 <- ifelse(is.na(testItemPairs$title12gramtitle2) | testItemPairs$title12gramtitle2 == Inf, -9999, testItemPairs$title12gramtitle2)

trainItemPairs$title13gramtitle2  <- stringdist(trainItemPairs$title1, trainItemPairs$title2, method='jaccard', q=3)
testItemPairs$title13gramtitle2  <- stringdist(testItemPairs$title1, testItemPairs$title2, method='jaccard', q=3)

trainItemPairs$title13gramtitle2 <- ifelse(is.na(trainItemPairs$title13gramtitle2) | trainItemPairs$title13gramtitle2 == Inf, -9999, trainItemPairs$title13gramtitle2)
testItemPairs$title13gramtitle2 <- ifelse(is.na(testItemPairs$title13gramtitle2) | testItemPairs$title13gramtitle2 == Inf, -9999, testItemPairs$title13gramtitle2)

trainItemPairs$title14gramtitle2  <- stringdist(trainItemPairs$title1, trainItemPairs$title2, method='jaccard', q=4)
testItemPairs$title14gramtitle2  <- stringdist(testItemPairs$title1, testItemPairs$title2, method='jaccard', q=4)

trainItemPairs$title14gramtitle2 <- ifelse(is.na(trainItemPairs$title14gramtitle2) | trainItemPairs$title14gramtitle2 == Inf, -9999, trainItemPairs$title14gramtitle2)
testItemPairs$title14gramtitle2 <- ifelse(is.na(testItemPairs$title14gramtitle2) | testItemPairs$title14gramtitle2 == Inf, -9999, testItemPairs$title14gramtitle2)

trainItemPairs$title15gramtitle2  <- stringdist(trainItemPairs$title1, trainItemPairs$title2, method='jaccard', q=5)
testItemPairs$title15gramtitle2  <- stringdist(testItemPairs$title1, testItemPairs$title2, method='jaccard', q=5)

trainItemPairs$title15gramtitle2 <- ifelse(is.na(trainItemPairs$title15gramtitle2) | trainItemPairs$title15gramtitle2 == Inf, -9999, trainItemPairs$title15gramtitle2)
testItemPairs$title15gramtitle2 <- ifelse(is.na(testItemPairs$title15gramtitle2) | testItemPairs$title15gramtitle2 == Inf, -9999, testItemPairs$title15gramtitle2)


trainItemPairs$title12gramdescription2  <- stringdist(trainItemPairs$title1, trainItemPairs$description2, method='jaccard', q=2)
testItemPairs$title12gramdescription2  <- stringdist(testItemPairs$title1, testItemPairs$description2, method='jaccard', q=2)

trainItemPairs$title12gramdescription2 <- ifelse(is.na(trainItemPairs$title12gramdescription2) | trainItemPairs$title12gramdescription2 == Inf, -9999, trainItemPairs$title12gramdescription2)
testItemPairs$title12gramdescription2 <- ifelse(is.na(testItemPairs$title12gramdescription2) | testItemPairs$title12gramdescription2 == Inf, -9999, testItemPairs$title12gramdescription2)

trainItemPairs$title13gramdescription2  <- stringdist(trainItemPairs$title1, trainItemPairs$description2, method='jaccard', q=3)
testItemPairs$title13gramdescription2  <- stringdist(testItemPairs$title1, testItemPairs$description2, method='jaccard', q=3)

trainItemPairs$title13gramdescription2 <- ifelse(is.na(trainItemPairs$title13gramdescription2) | trainItemPairs$title13gramdescription2 == Inf, -9999, trainItemPairs$title13gramdescription2)
testItemPairs$title13gramdescription2 <- ifelse(is.na(testItemPairs$title13gramdescription2) | testItemPairs$title13gramdescription2 == Inf, -9999, testItemPairs$title13gramdescription2)

trainItemPairs$title14gramdescription2  <- stringdist(trainItemPairs$title1, trainItemPairs$description2, method='jaccard', q=4)
testItemPairs$title14gramdescription2  <- stringdist(testItemPairs$title1, testItemPairs$description2, method='jaccard', q=4)

trainItemPairs$title14gramdescription2 <- ifelse(is.na(trainItemPairs$title14gramdescription2) | trainItemPairs$title14gramdescription2 == Inf, -9999, trainItemPairs$title14gramdescription2)
testItemPairs$title14gramdescription2 <- ifelse(is.na(testItemPairs$title14gramdescription2) | testItemPairs$title14gramdescription2 == Inf, -9999, testItemPairs$title14gramdescription2)

trainItemPairs$title15gramdescription2  <- stringdist(trainItemPairs$title1, trainItemPairs$description2, method='jaccard', q=5)
testItemPairs$title15gramdescription2  <- stringdist(testItemPairs$title1, testItemPairs$description2, method='jaccard', q=5)

trainItemPairs$title15gramdescription2 <- ifelse(is.na(trainItemPairs$title15gramdescription2) | trainItemPairs$title15gramdescription2 == Inf, -9999, trainItemPairs$title15gramdescription2)
testItemPairs$title15gramdescription2 <- ifelse(is.na(testItemPairs$title15gramdescription2) | testItemPairs$title15gramdescription2 == Inf, -9999, testItemPairs$title15gramdescription2)


trainItemPairs$description12gramdescription2  <- stringdist(trainItemPairs$description1, trainItemPairs$description2, method='jaccard', q=2)
testItemPairs$description12gramdescription2  <- stringdist(testItemPairs$description1, testItemPairs$description2, method='jaccard', q=2)
trainItemPairs$description12gramdescription2 <- ifelse(is.na(trainItemPairs$description12gramdescription2) | trainItemPairs$description12gramdescription2 == Inf, -9999, trainItemPairs$description12gramdescription2)
testItemPairs$description12gramdescription2 <- ifelse(is.na(testItemPairs$description12gramdescription2) | testItemPairs$description12gramdescription2 == Inf, -9999, testItemPairs$description12gramdescription2)
####################################################################################################################

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

train_allTwoword_match <- as.data.frame( t(mapply(allTwoword_match,trainItemPairs$title1,trainItemPairs$title2)))

trainItemPairs$Title1TwoWordsMatchedinTitle2   <- train_allTwoword_match[,2]
trainItemPairs$Title1TwoWordsMatchedinTitle2 <- ifelse((trainItemPairs$Title1Length -1) == 0, 0 , trainItemPairs$Title1TwoWordsMatchedinTitle2/(trainItemPairs$Title1Length -1))


train_allTwoword_match <- as.data.frame( t(mapply(allTwoword_match,trainItemPairs$title1,trainItemPairs$description2)))

trainItemPairs$Title1TwoWordsMatchedinDesc2   <- train_allTwoword_match[,2]
trainItemPairs$Title1TwoWordsMatchedinDesc2   <- ifelse((trainItemPairs$Title1Length -1) == 0, 0 , trainItemPairs$Title1TwoWordsMatchedinDesc2/(trainItemPairs$Title1Length -1))

test_allTwoword_match <- as.data.frame( t(mapply(allTwoword_match,testItemPairs$title1,testItemPairs$title2)))

testItemPairs$Title1TwoWordsMatchedinTitle2   <- test_allTwoword_match[,2]
testItemPairs$Title1TwoWordsMatchedinTitle2 <- ifelse((testItemPairs$Title1Length -1) == 0, 0 , testItemPairs$Title1TwoWordsMatchedinTitle2/(testItemPairs$Title1Length -1))


test_allTwoword_match <- as.data.frame( t(mapply(allTwoword_match,testItemPairs$title1,testItemPairs$description2)))

testItemPairs$Title1TwoWordsMatchedinDesc2   <- test_allTwoword_match[,2]
testItemPairs$Title1TwoWordsMatchedinDesc2   <- ifelse((testItemPairs$Title1Length -1) == 0, 0 , testItemPairs$Title1TwoWordsMatchedinDesc2/(testItemPairs$Title1Length -1))


trainItemPairs$titlesCosineDist <-  stringdist(trainItemPairs$title1,trainItemPairs$title2, method = 'lv') #'cosine'
testItemPairs$titlesCosineDist <-  stringdist(testItemPairs$title1,testItemPairs$title2, method = 'lv')
trainItemPairs$titlesCosinesDist <-  stringdist(trainItemPairs$title1,trainItemPairs$title2, method = 'cosine') #'cosine'
testItemPairs$titlesCosinesDist <-  stringdist(testItemPairs$title1,testItemPairs$title2, method = 'cosine')


##########################################################################################################

WordPunctCount <- function(firsttitle1,secondtitle2){
  SpecialCharCount1  <- 0
  SpecialCharCount2  <- 0
  {
  SpecialCharCount1  <- str_count(firsttitle1) - str_count(str_replace_all(firsttitle1, "[[:punct:]]", ""))
  SpecialCharCount2  <- str_count(secondtitle2) - str_count(str_replace_all(secondtitle2, "[[:punct:]]", ""))
  }
  return(c(SpecialCharCount1,SpecialCharCount2 ))
}

train_WordPunct <- as.data.frame( t(mapply(WordPunctCount,trainItemPairs$title1,trainItemPairs$title2)))
trainItemPairs$title1PunctCount <-  train_WordPunct[,1]
trainItemPairs$title2PunctCount <-  train_WordPunct[,2]


test_WordPunct <- as.data.frame( t(mapply(WordPunctCount,testItemPairs$title1,testItemPairs$title2)))
testItemPairs$title1PunctCount <-  test_WordPunct[,1]
testItemPairs$title2PunctCount <-  test_WordPunct[,2]

trainItemPairs$Title1PunctsMatchedinTitle2ratio   <- ifelse(trainItemPairs$title1PunctCount == 0, 0, trainItemPairs$title2PunctCount/trainItemPairs$title1PunctCount)
trainItemPairs$Title2PunctsMatchedinTitle1ratio   <- ifelse(trainItemPairs$title2PunctCount == 0, 0, trainItemPairs$title1PunctCount/trainItemPairs$title2PunctCount)

trainItemPairs$Title1PunctsMatchedinTitle2ratioMin    <- pmin(trainItemPairs$Title1PunctsMatchedinTitle2ratio, trainItemPairs$Title2PunctsMatchedinTitle1ratio, na.rm=TRUE)
trainItemPairs$Title1PunctsMatchedinTitle2ratioMax    <- pmax(trainItemPairs$Title1PunctsMatchedinTitle2ratio, trainItemPairs$Title2PunctsMatchedinTitle1ratio, na.rm=TRUE)

trainItemPairs$Title1PunctsMatchedinTitle2ratioMin    <- ifelse(is.na(trainItemPairs$Title1PunctsMatchedinTitle2ratioMin) | trainItemPairs$Title1PunctsMatchedinTitle2ratioMin == Inf, -9999, trainItemPairs$Title1PunctsMatchedinTitle2ratioMin)
trainItemPairs$Title1PunctsMatchedinTitle2ratioMax    <- ifelse(is.na(trainItemPairs$Title1PunctsMatchedinTitle2ratioMax) | trainItemPairs$Title1PunctsMatchedinTitle2ratioMax == Inf, -9999, trainItemPairs$Title1PunctsMatchedinTitle2ratioMax)

testItemPairs$Title1PunctsMatchedinTitle2ratio   <- ifelse(testItemPairs$title1PunctCount == 0, 0, testItemPairs$title2PunctCount/testItemPairs$title1PunctCount)
testItemPairs$Title2PunctsMatchedinTitle1ratio   <- ifelse(testItemPairs$title2PunctCount == 0, 0, testItemPairs$title1PunctCount/testItemPairs$title2PunctCount)

testItemPairs$Title1PunctsMatchedinTitle2ratioMin    <- pmin(testItemPairs$Title1PunctsMatchedinTitle2ratio, testItemPairs$Title2PunctsMatchedinTitle1ratio, na.rm=TRUE)
testItemPairs$Title1PunctsMatchedinTitle2ratioMax    <- pmax(testItemPairs$Title1PunctsMatchedinTitle2ratio, testItemPairs$Title2PunctsMatchedinTitle1ratio, na.rm=TRUE)

testItemPairs$Title1PunctsMatchedinTitle2ratioMin    <- ifelse(is.na(testItemPairs$Title1PunctsMatchedinTitle2ratioMin) | testItemPairs$Title1PunctsMatchedinTitle2ratioMin == Inf, -9999, testItemPairs$Title1PunctsMatchedinTitle2ratioMin)
testItemPairs$Title1PunctsMatchedinTitle2ratioMax    <- ifelse(is.na(testItemPairs$Title1PunctsMatchedinTitle2ratioMax) | testItemPairs$Title1PunctsMatchedinTitle2ratioMax == Inf, -9999, testItemPairs$Title1PunctsMatchedinTitle2ratioMax)

trainItemPairs$Title1PunctsMatchedinTitle2ratioDiff    <- trainItemPairs$Title1PunctsMatchedinTitle2ratio -  trainItemPairs$Title2PunctsMatchedinTitle1ratio
trainItemPairs$Title1PunctsMatchedinTitle2ratioDiff    <- ifelse(is.na(trainItemPairs$Title1PunctsMatchedinTitle2ratioDiff) | trainItemPairs$Title1PunctsMatchedinTitle2ratioDiff == Inf, -9999, trainItemPairs$Title1PunctsMatchedinTitle2ratioDiff)

testItemPairs$Title1PunctsMatchedinTitle2ratioDiff    <- testItemPairs$Title1PunctsMatchedinTitle2ratio -  testItemPairs$Title2PunctsMatchedinTitle1ratio
testItemPairs$Title1PunctsMatchedinTitle2ratioDiff    <- ifelse(is.na(testItemPairs$Title1PunctsMatchedinTitle2ratioDiff) | testItemPairs$Title1PunctsMatchedinTitle2ratioDiff == Inf, -9999, testItemPairs$Title1PunctsMatchedinTitle2ratioDiff)

trainRowHashs$AvgWidthtoHeight_array2RatioDiff  <- trainRowHashs$AvgWidthtoHeight_array2Ratio  - trainRowHashs$AvgWidthtoHeight_array1Ratio
trainRowHashs$SumWidthtoHeight_array2RatioDiff  <- trainRowHashs$SumWidthtoHeight_array2Ratio  - trainRowHashs$SumWidthtoHeight_array1Ratio
trainRowHashs$MaxWidth_array1_array2RatioDiff   <- trainRowHashs$MaxWidth_array1_array2Ratio   - trainRowHashs$MaxWidth_array2_array1Ratio
trainRowHashs$MaxHeight_array1_array2RatioDiff  <- trainRowHashs$MaxHeight_array1_array2Ratio  - trainRowHashs$MaxHeight_array2_array1Ratio

testRowHashs$AvgWidthtoHeight_array2RatioDiff  <- testRowHashs$AvgWidthtoHeight_array2Ratio  - testRowHashs$AvgWidthtoHeight_array1Ratio
testRowHashs$SumWidthtoHeight_array2RatioDiff  <- testRowHashs$SumWidthtoHeight_array2Ratio  - testRowHashs$SumWidthtoHeight_array1Ratio
testRowHashs$MaxWidth_array1_array2RatioDiff   <- testRowHashs$MaxWidth_array1_array2Ratio   - testRowHashs$MaxWidth_array2_array1Ratio
testRowHashs$MaxHeight_array1_array2RatioDiff  <- testRowHashs$MaxHeight_array1_array2Ratio  - testRowHashs$MaxHeight_array2_array1Ratio


trainRowHashs$AvgEntropy_array2ImagesDiff    <- trainRowHashs$AvgEntropy_array2Images -  trainRowHashs$AvgEntropy_array1Images
trainRowHashs$AvgEntropy_array2ImagesDiff    <- ifelse(is.na(trainRowHashs$AvgEntropy_array2ImagesDiff) | trainRowHashs$AvgEntropy_array2ImagesDiff == Inf, -9999, trainRowHashs$AvgEntropy_array2ImagesDiff)

testRowHashs$AvgEntropy_array2ImagesDiff    <- testRowHashs$AvgEntropy_array2Images -  testRowHashs$AvgEntropy_array1Images
testRowHashs$AvgEntropy_array2ImagesDiff    <- ifelse(is.na(testRowHashs$AvgEntropy_array2ImagesDiff) | testRowHashs$AvgEntropy_array2ImagesDiff == Inf, -9999, testRowHashs$AvgEntropy_array2ImagesDiff)

###########################################################################################################


ExtractNumbersFromString <- function(firsttitle1,secondtitle2){
  
  { temp1 <- ifelse(length(firsttitle1) >0 ,  gregexpr("[0-9]+", firsttitle1), "")
  NumbersFromString1 <- ifelse(length(temp1) == 0, "", as.numeric(unique(unlist(regmatches(firsttitle1, temp1)))))
  temp2 <- ifelse(length(secondtitle2) >0 ,  gregexpr("[0-9]+", secondtitle2), "")
  NumbersFromString2 <- ifelse(length(temp2) == 0, "",as.numeric(unique(unlist(regmatches(secondtitle2, temp2)))))
  NumbersFromString1 <- ifelse(is.na(NumbersFromString1),0,NumbersFromString1)
  NumbersFromString2 <- ifelse(is.na(NumbersFromString2),0,NumbersFromString2)
  }
  return(c(NumbersFromString1,NumbersFromString2 ))
}

train_ExtractNumbersFromString  <- as.data.frame( t(mapply(ExtractNumbersFromString,trainItemPairs$title1,trainItemPairs$title2)))

trainItemPairs$title1Numbers <-  train_ExtractNumbersFromString[,1]
trainItemPairs$title2NUmbers <-  train_ExtractNumbersFromString[,2] 

test_ExtractNumbersFromString  <- as.data.frame( t(mapply(ExtractNumbersFromString,testItemPairs$title1,testItemPairs$title2)))

testItemPairs$title1Numbers <-  test_ExtractNumbersFromString[,1]
testItemPairs$title2NUmbers <-  test_ExtractNumbersFromString[,2] 

trainItemPairs$titleNumbersMatch <- ifelse((trainItemPairs$title1Numbers == trainItemPairs$title2NUmbers) & trainItemPairs$title1Numbers != 0 , 1, 0)

testItemPairs$titleNumbersMatch <- ifelse((testItemPairs$title1Numbers == testItemPairs$title2NUmbers) & testItemPairs$title1Numbers != 0 , 1, 0)

##########################################################################################################
## IMHO
## New features -- 2016-07-06
## three new features added almost at the end and used in only one Prav model so need to generate 
## features files to avoid any changes in all previous combined models and Prav models

##########################################################################################################

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
train_CountEnglishWords  <- as.data.frame( t(mapply(CountEnglishWords,trainItemPairs$title1,trainItemPairs$title2)))

trainItemPairs$EWtitle1WordsEnglish <-  train_CountEnglishWords[,1] 
trainItemPairs$EWtitle2WordsEnglish <-  train_CountEnglishWords[,2]

trainItemPairs$EWtitle1totitle2Ratios <- ifelse(trainItemPairs$EWtitle2WordsEnglish == 0 , 0 , trainItemPairs$EWtitle1WordsEnglish /trainItemPairs$EWtitle2WordsEnglish)

trainItemPairs$EWtitle1Towords <- ifelse(trainItemPairs$Title1Length == 0, 0 , trainItemPairs$EWtitle1WordsEnglish / trainItemPairs$Title1Length)
trainItemPairs$EWtitle2Towords <- ifelse(trainItemPairs$Title2Length == 0, 0 , trainItemPairs$EWtitle2WordsEnglish / trainItemPairs$Title2Length)

trainItemPairs$EWtitle1totitle2Diff  <- trainItemPairs$EWtitle1Towords - trainItemPairs$EWtitle2Towords
trainItemPairs$EWTitlesToWordsRatios <- ifelse(trainItemPairs$EWtitle2Towords == 0 , 0 , trainItemPairs$EWtitle1Towords /trainItemPairs$EWtitle2Towords)


# validate function results with mix of both languages
test_CountEnglishWords  <- as.data.frame( t(mapply(CountEnglishWords,testItemPairs$title1,testItemPairs$title2)))

testItemPairs$EWtitle1WordsEnglish <-  test_CountEnglishWords[,1] 
testItemPairs$EWtitle2WordsEnglish <-  test_CountEnglishWords[,2]

testItemPairs$EWtitle1totitle2Ratios <- ifelse(testItemPairs$EWtitle2WordsEnglish == 0 , 0 , testItemPairs$EWtitle1WordsEnglish /testItemPairs$EWtitle2WordsEnglish)

testItemPairs$EWtitle1Towords <- ifelse(testItemPairs$Title1Length == 0, 0 , testItemPairs$EWtitle1WordsEnglish / testItemPairs$Title1Length)
testItemPairs$EWtitle2Towords <- ifelse(testItemPairs$Title2Length == 0, 0 , testItemPairs$EWtitle2WordsEnglish / testItemPairs$Title2Length)

testItemPairs$EWtitle1totitle2Diff  <- testItemPairs$EWtitle1Towords - testItemPairs$EWtitle2Towords
testItemPairs$EWTitlesToWordsRatios <- ifelse(testItemPairs$EWtitle2Towords == 0 , 0 , testItemPairs$EWtitle1Towords /testItemPairs$EWtitle2Towords)





##################################################################################################

trainItemPairsFull<- join(trainItemPairs, trainRowHashs, type="left")
testItemPairsFull <- join(testItemPairs , testRowHashs , type="left")

###################################################################################################

###################################################################################################

trainItemPairsFull[is.na(trainItemPairsFull) | trainItemPairsFull==-9999] <- 0
testItemPairsFull[is.na(testItemPairsFull)| testItemPairsFull==-9999]     <- 0


####################################################################################################

####################################################################################################
## Generating feature files

###################################################################################################

features  <- c(
  "itemID_1"
  ,"itemID_2"
  ,"isDuplicate"
  , "sameLat"
  ,"sameLon"
  ,"sameLocation"
  ,"sameregion"
  ,"priceDifference"
  ,"sameTitle"
  ,"sameDescription"
  ,"imageArrayDiff"                         
  ,"SimilarityTitle"
  ,"SimilarityDescription"
  ,"samemetro"
  ,"SimilarityJSON"                        
  ,"distance"
  ,"nchardescription1"   
  ,"nchardescription2"   
  ,"priceDiff"          
  ,"priceMin"            
  ,"priceMax"            
  ,"titleStringDistance2"                  
  ,"title1StartsWithTitle2"
  ,"title2StartsWithTitle1"
  ,"titleCharDiff"
  ,"titleCharMin"
  ,"titleCharMax"
  ,"descriptionCharDiff"
  ,"descriptionCharMin" 
  ,"descriptionCharMax"
  ,"Title1WordsMatchedinTitle2ratio"
  ,"Title2WordsMatchedinTitle1ratio"         
  ,"ImageCombinations"
  ,"MinHammingDistance"
  ,"MaxHammingDistance"
  ,"MaxMinDifference"                       
  ,"AvgHammingDistance"
  ,"CountBelow10distance"
  ,"CountAbove50distance"
  ,"Title1WordsMatchedinDesc2ratio"
  ,"Title2WordsMatchedinDesc1ratio"
  ,"description1WordsMatchedinDesc2ratio"    
  ,"CountBetween11and20distance"
  ,"CountBetween21and30distance"
  ,"CountBetween31and40distance"
  ,"CountBetween41and50distance"
  ,"attrsJSON1WordsMatchedinattrsJSON2ratio"
  ,"Title1WordsMatchedinTitle2ratioDiff"
  ,"Title1WordsMatchedinTitle2ratioMin" 
  ,"Title1WordsMatchedinTitle2ratioMax"
  ,"Title1WordsMatchedinDesc2ratioDiff" 
  ,"Title1WordsMatchedinDesc2ratioMin"  
  ,"Title1WordsMatchedinDesc2ratioMax"
  ,"Title1WordsMatchedinTitle2Diff"
  ,"Title1WordsMatchedinDesc2Diff"
  ,"title1StartingWithTitle2" 
  ,"title1EndsWithTitle2" 
  ,"title1SoundexWithTitle2"
  ,"title1EndSoundexWithTitle2"  
  ,"title1StartSoundexWithTitle2"
  ,"Title1WordsStrengthinTitle2"
  ,"Title1WordsStrengthinTitle2ratio"
  ,"title12gramtitle2"
  ,"Title1TwoWordsMatchedinTitle2"     
  ,"Title1TwoWordsMatchedinDesc2"      
  ,"SameWidthCount"
  ,"MaxWidthDiff"                           
  ,"MinWidthDiff"                            
  ,"AvgWidthDiff"                            
  ,"SameHeightCount"                        
  ,"MaxHeightDiff"                           
  ,"MinHeightDiff"                           
  ,"AvgHeightDiff"                          
  ,"SameEntropyCount"                        
  ,"MaxEntropyDiff"                          
  ,"MinEntropyDiff"                         
  ,"AvgEntropyDiff"                          
  ,"SamecompressedsizeCount"                 
  ,"MaxcompressedsizeDiff"                  
  ,"MincompressedsizeDiff"                   
  ,"AvgcompressedsizeDiff"
  ,"AvgEntropyarray1and2Ratio"
  ,"AvgWidthyarray1and2Ratio" 
  ,"AvgHeightarray1and2Ratio" 
  ,"Avgcompressed_sizearray1and2Ratio"
  ,"Avgsizearray1and2Ratio"
  ,"AvgEntropy_array2Images"
  ,"AvgEntropy_array1Images"
  ,"AvgWidthtoHeight_array2Ratio"
  ,"AvgWidthtoHeight_array1Ratio"
  ,"SumWidthtoHeight_array2Ratio"
  ,"SumWidthtoHeight_array1Ratio"
  ,"MaxWidth_array1_array2Ratio" 
  ,"MaxWidth_array2_array1Ratio" 
  ,"MaxHeight_array1_array2Ratio"
  ,"MaxHeight_array2_array1Ratio"
  ,"AvgEntropy_array1_array2Images"
  ,"AvgEntropy_array2_array1Images"
  ,"Title1PunctsMatchedinTitle2ratioDiff"
  ,"AvgWidthtoHeight_array2RatioDiff"
  ,"SumWidthtoHeight_array2RatioDiff"
  ,"MaxWidth_array1_array2RatioDiff" 
  ,"MaxHeight_array1_array2RatioDiff"
  ,"AvgEntropy_array2ImagesDiff"
  ,"titleNumbersMatch"

)

training <- trainItemPairsFull[,features,with=FALSE]
write.csv(training, './TrainingFeatures.csv', row.names=FALSE, quote = FALSE)

features  <- c(
  "itemID_1"
  ,"itemID_2"
  ,"id"
  , "sameLat"
  ,"sameLon"
  ,"sameLocation"
  ,"sameregion"
  ,"priceDifference"
  ,"sameTitle"
  ,"sameDescription"
  ,"imageArrayDiff"                         
  ,"SimilarityTitle"
  ,"SimilarityDescription"
  ,"samemetro"
  ,"SimilarityJSON"                        
  ,"distance"
  ,"nchardescription1"   
  ,"nchardescription2"   
  ,"priceDiff"          
  ,"priceMin"            
  ,"priceMax"            
  ,"titleStringDistance2"                  
  ,"title1StartsWithTitle2"
  ,"title2StartsWithTitle1"
  ,"titleCharDiff"
  ,"titleCharMin"
  ,"titleCharMax"
  ,"descriptionCharDiff"
  ,"descriptionCharMin" 
  ,"descriptionCharMax"
  ,"Title1WordsMatchedinTitle2ratio"
  ,"Title2WordsMatchedinTitle1ratio"         
  ,"ImageCombinations"
  ,"MinHammingDistance"
  ,"MaxHammingDistance"
  ,"MaxMinDifference"                       
  ,"AvgHammingDistance"
  ,"CountBelow10distance"
  ,"CountAbove50distance"
  ,"Title1WordsMatchedinDesc2ratio"
  ,"Title2WordsMatchedinDesc1ratio"
  ,"description1WordsMatchedinDesc2ratio"    
  ,"CountBetween11and20distance"
  ,"CountBetween21and30distance"
  ,"CountBetween31and40distance"
  ,"CountBetween41and50distance"
  ,"attrsJSON1WordsMatchedinattrsJSON2ratio"
  ,"Title1WordsMatchedinTitle2ratioDiff"
  ,"Title1WordsMatchedinTitle2ratioMin" 
  ,"Title1WordsMatchedinTitle2ratioMax"
  ,"Title1WordsMatchedinDesc2ratioDiff" 
  ,"Title1WordsMatchedinDesc2ratioMin"  
  ,"Title1WordsMatchedinDesc2ratioMax"
  ,"Title1WordsMatchedinTitle2Diff"
  ,"Title1WordsMatchedinDesc2Diff"
  ,"title1StartingWithTitle2" 
  ,"title1EndsWithTitle2" 
  ,"title1SoundexWithTitle2"
  ,"title1EndSoundexWithTitle2"  
  ,"title1StartSoundexWithTitle2"
  ,"Title1WordsStrengthinTitle2"
  ,"Title1WordsStrengthinTitle2ratio"
  ,"title12gramtitle2"
  ,"Title1TwoWordsMatchedinTitle2"     
  ,"Title1TwoWordsMatchedinDesc2"      
  ,"SameWidthCount"
  ,"MaxWidthDiff"                           
  ,"MinWidthDiff"                            
  ,"AvgWidthDiff"                            
  ,"SameHeightCount"                        
  ,"MaxHeightDiff"                           
  ,"MinHeightDiff"                           
  ,"AvgHeightDiff"                          
  ,"SameEntropyCount"                        
  ,"MaxEntropyDiff"                          
  ,"MinEntropyDiff"                         
  ,"AvgEntropyDiff"                          
  ,"SamecompressedsizeCount"                 
  ,"MaxcompressedsizeDiff"                  
  ,"MincompressedsizeDiff"                   
  ,"AvgcompressedsizeDiff"
  ,"AvgEntropyarray1and2Ratio"
  ,"AvgWidthyarray1and2Ratio" 
  ,"AvgHeightarray1and2Ratio" 
  ,"Avgcompressed_sizearray1and2Ratio"
  ,"Avgsizearray1and2Ratio"
  ,"AvgEntropy_array2Images"
  ,"AvgEntropy_array1Images"
  ,"AvgWidthtoHeight_array2Ratio"
  ,"AvgWidthtoHeight_array1Ratio"
  ,"SumWidthtoHeight_array2Ratio"
  ,"SumWidthtoHeight_array1Ratio"
  ,"MaxWidth_array1_array2Ratio" 
  ,"MaxWidth_array2_array1Ratio" 
  ,"MaxHeight_array1_array2Ratio"
  ,"MaxHeight_array2_array1Ratio"
  ,"AvgEntropy_array1_array2Images"
  ,"AvgEntropy_array2_array1Images"
  ,"Title1PunctsMatchedinTitle2ratioDiff"
  ,"AvgWidthtoHeight_array2RatioDiff"
  ,"SumWidthtoHeight_array2RatioDiff"
  ,"MaxWidth_array1_array2RatioDiff" 
  ,"MaxHeight_array1_array2RatioDiff"
  ,"AvgEntropy_array2ImagesDiff"
  ,"titleNumbersMatch"

)

testing  <- testItemPairsFull[,features,with=FALSE]
write.csv(testing , './TestingFeatures.csv', row.names=FALSE, quote = FALSE)


features  <- c(
  "itemID_1"
  ,"itemID_2"
  ,"isDuplicate"
  , "sameLat"
  ,"sameLon"
  ,"sameLocation"
  ,"sameregion"
  ,"priceDifference"
  ,"sameTitle"
  ,"sameDescription"
  ,"imageArrayDiff"                         
  ,"SimilarityTitle"
  ,"SimilarityDescription"
  ,"samemetro"
  ,"SimilarityJSON"                        
  ,"distance"
  ,"nchardescription1"   
  ,"nchardescription2"   
  ,"priceDiff"          
  ,"priceMin"            
  ,"priceMax"            
  ,"titleStringDistance2"                  
  ,"title1StartsWithTitle2"
  ,"title2StartsWithTitle1"
  ,"titleCharDiff"
  ,"titleCharMin"
  ,"titleCharMax"
  ,"descriptionCharDiff"
  ,"descriptionCharMin" 
  ,"descriptionCharMax"
  ,"Title1WordsMatchedinTitle2ratio"
  ,"Title2WordsMatchedinTitle1ratio"         
  ,"ImageCombinations"
  ,"MinHammingDistance"
  ,"MaxHammingDistance"
  ,"MaxMinDifference"                       
  ,"AvgHammingDistance"
  ,"CountBelow10distance"
  ,"CountAbove50distance"
  ,"Title1WordsMatchedinDesc2ratio"
  ,"Title2WordsMatchedinDesc1ratio"
  ,"description1WordsMatchedinDesc2ratio"    
  ,"CountBetween11and20distance"
  ,"CountBetween21and30distance"
  ,"CountBetween31and40distance"
  ,"CountBetween41and50distance"
  ,"attrsJSON1WordsMatchedinattrsJSON2ratio"
  ,"Title1WordsMatchedinTitle2ratioDiff"
  ,"Title1WordsMatchedinTitle2ratioMin" 
  ,"Title1WordsMatchedinTitle2ratioMax"
  ,"Title1WordsMatchedinDesc2ratioDiff" 
  ,"Title1WordsMatchedinDesc2ratioMin"  
  ,"Title1WordsMatchedinDesc2ratioMax"
  ,"Title1WordsMatchedinTitle2Diff"
  ,"Title1WordsMatchedinDesc2Diff"
  ,"title1StartingWithTitle2" 
  ,"title1EndsWithTitle2" 
  ,"title1SoundexWithTitle2"
  ,"title1EndSoundexWithTitle2"  
  ,"title1StartSoundexWithTitle2"
  ,"Title1WordsStrengthinTitle2"
  ,"Title1WordsStrengthinTitle2ratio"
  ,"title12gramtitle2"
  ,"Title1TwoWordsMatchedinTitle2"     
  ,"Title1TwoWordsMatchedinDesc2"      
  ,"SameWidthCount"
  ,"MaxWidthDiff"                           
  ,"MinWidthDiff"                            
  ,"AvgWidthDiff"                            
  ,"SameHeightCount"                        
  ,"MaxHeightDiff"                           
  ,"MinHeightDiff"                           
  ,"AvgHeightDiff"                          
  ,"SameEntropyCount"                        
  ,"MaxEntropyDiff"                          
  ,"MinEntropyDiff"                         
  ,"AvgEntropyDiff"                          
  ,"SamecompressedsizeCount"                 
  ,"MaxcompressedsizeDiff"                  
  ,"MincompressedsizeDiff"                   
  ,"AvgcompressedsizeDiff"
  ,"AvgEntropyarray1and2Ratio"
  ,"AvgWidthyarray1and2Ratio" 
  ,"AvgHeightarray1and2Ratio" 
  ,"Avgcompressed_sizearray1and2Ratio"
  ,"Avgsizearray1and2Ratio"
  ,"AvgEntropy_array2Images"
  ,"AvgEntropy_array1Images"
  ,"AvgWidthtoHeight_array2Ratio"
  ,"AvgWidthtoHeight_array1Ratio"
  ,"SumWidthtoHeight_array2Ratio"
  ,"SumWidthtoHeight_array1Ratio"
  ,"MaxWidth_array1_array2Ratio" 
  ,"MaxWidth_array2_array1Ratio" 
  ,"MaxHeight_array1_array2Ratio"
  ,"MaxHeight_array2_array1Ratio"
  ,"AvgEntropy_array1_array2Images"
  ,"AvgEntropy_array2_array1Images"
  ,"Title1PunctsMatchedinTitle2ratioDiff"
  ,"AvgWidthtoHeight_array2RatioDiff"
  ,"SumWidthtoHeight_array2RatioDiff"
  ,"MaxWidth_array1_array2RatioDiff" 
  ,"MaxHeight_array1_array2RatioDiff"
  ,"AvgEntropy_array2ImagesDiff"
  ,"titleNumbersMatch"
  ,"EWtitle1totitle2Ratios"
  ,"EWtitle1totitle2Diff" 
  ,"EWTitlesToWordsRatios"
)

training <- trainItemPairsFull[,features,with=FALSE]
write.csv(training, './TrainingFeatures_20160706.csv', row.names=FALSE, quote = FALSE)

features  <- c(
  "itemID_1"
  ,"itemID_2"
  ,"id"
  , "sameLat"
  ,"sameLon"
  ,"sameLocation"
  ,"sameregion"
  ,"priceDifference"
  ,"sameTitle"
  ,"sameDescription"
  ,"imageArrayDiff"                         
  ,"SimilarityTitle"
  ,"SimilarityDescription"
  ,"samemetro"
  ,"SimilarityJSON"                        
  ,"distance"
  ,"nchardescription1"   
  ,"nchardescription2"   
  ,"priceDiff"          
  ,"priceMin"            
  ,"priceMax"            
  ,"titleStringDistance2"                  
  ,"title1StartsWithTitle2"
  ,"title2StartsWithTitle1"
  ,"titleCharDiff"
  ,"titleCharMin"
  ,"titleCharMax"
  ,"descriptionCharDiff"
  ,"descriptionCharMin" 
  ,"descriptionCharMax"
  ,"Title1WordsMatchedinTitle2ratio"
  ,"Title2WordsMatchedinTitle1ratio"         
  ,"ImageCombinations"
  ,"MinHammingDistance"
  ,"MaxHammingDistance"
  ,"MaxMinDifference"                       
  ,"AvgHammingDistance"
  ,"CountBelow10distance"
  ,"CountAbove50distance"
  ,"Title1WordsMatchedinDesc2ratio"
  ,"Title2WordsMatchedinDesc1ratio"
  ,"description1WordsMatchedinDesc2ratio"    
  ,"CountBetween11and20distance"
  ,"CountBetween21and30distance"
  ,"CountBetween31and40distance"
  ,"CountBetween41and50distance"
  ,"attrsJSON1WordsMatchedinattrsJSON2ratio"
  ,"Title1WordsMatchedinTitle2ratioDiff"
  ,"Title1WordsMatchedinTitle2ratioMin" 
  ,"Title1WordsMatchedinTitle2ratioMax"
  ,"Title1WordsMatchedinDesc2ratioDiff" 
  ,"Title1WordsMatchedinDesc2ratioMin"  
  ,"Title1WordsMatchedinDesc2ratioMax"
  ,"Title1WordsMatchedinTitle2Diff"
  ,"Title1WordsMatchedinDesc2Diff"
  ,"title1StartingWithTitle2" 
  ,"title1EndsWithTitle2" 
  ,"title1SoundexWithTitle2"
  ,"title1EndSoundexWithTitle2"  
  ,"title1StartSoundexWithTitle2"
  ,"Title1WordsStrengthinTitle2"
  ,"Title1WordsStrengthinTitle2ratio"
  ,"title12gramtitle2"
  ,"Title1TwoWordsMatchedinTitle2"     
  ,"Title1TwoWordsMatchedinDesc2"      
  ,"SameWidthCount"
  ,"MaxWidthDiff"                           
  ,"MinWidthDiff"                            
  ,"AvgWidthDiff"                            
  ,"SameHeightCount"                        
  ,"MaxHeightDiff"                           
  ,"MinHeightDiff"                           
  ,"AvgHeightDiff"                          
  ,"SameEntropyCount"                        
  ,"MaxEntropyDiff"                          
  ,"MinEntropyDiff"                         
  ,"AvgEntropyDiff"                          
  ,"SamecompressedsizeCount"                 
  ,"MaxcompressedsizeDiff"                  
  ,"MincompressedsizeDiff"                   
  ,"AvgcompressedsizeDiff"
  ,"AvgEntropyarray1and2Ratio"
  ,"AvgWidthyarray1and2Ratio" 
  ,"AvgHeightarray1and2Ratio" 
  ,"Avgcompressed_sizearray1and2Ratio"
  ,"Avgsizearray1and2Ratio"
  ,"AvgEntropy_array2Images"
  ,"AvgEntropy_array1Images"
  ,"AvgWidthtoHeight_array2Ratio"
  ,"AvgWidthtoHeight_array1Ratio"
  ,"SumWidthtoHeight_array2Ratio"
  ,"SumWidthtoHeight_array1Ratio"
  ,"MaxWidth_array1_array2Ratio" 
  ,"MaxWidth_array2_array1Ratio" 
  ,"MaxHeight_array1_array2Ratio"
  ,"MaxHeight_array2_array1Ratio"
  ,"AvgEntropy_array1_array2Images"
  ,"AvgEntropy_array2_array1Images"
  ,"Title1PunctsMatchedinTitle2ratioDiff"
  ,"AvgWidthtoHeight_array2RatioDiff"
  ,"SumWidthtoHeight_array2RatioDiff"
  ,"MaxWidth_array1_array2RatioDiff" 
  ,"MaxHeight_array1_array2RatioDiff"
  ,"AvgEntropy_array2ImagesDiff"
  ,"titleNumbersMatch"
  ,"EWtitle1totitle2Ratios"
  ,"EWtitle1totitle2Diff" 
  ,"EWTitlesToWordsRatios"
)

testing  <- testItemPairsFull[,features,with=FALSE]
write.csv(testing , './TestingFeatures_20160706.csv', row.names=FALSE, quote = FALSE)


#############################################################################################################

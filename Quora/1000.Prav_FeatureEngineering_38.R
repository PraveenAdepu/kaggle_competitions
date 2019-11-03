train_df = read_csv("./input/train_features_01.csv")
test_df = read_csv("./input/test_features_01.csv")

names(test_df)[1] <- "test_id"

train_01 = read_csv("./input/train_features_02.csv")
test_01 = read_csv("./input/test_features_02.csv")

train_02 = read_csv("./input/train_features_03.csv")
test_02 = read_csv("./input/test_features_03.csv")

train_03 = read_csv("./input/train_question_freq_features.csv")
test_03 = read_csv("./input/test_question_freq_features.csv")
names(test_03)[1] <- "test_id"
train_df <- left_join(train_df, train_01, , by = 'id')
train_df <- left_join(train_df, train_02, , by = 'id')
train_df <- left_join(train_df, train_03, , by = 'id')

test_df <- left_join(test_df, test_01, by = 'test_id')
test_df <- left_join(test_df, test_02, by = 'test_id')
test_df <- left_join(test_df, test_03, by = 'test_id')


 train_df$word_q1_2w        <- pmax(train_df$word_match/ train_df$q1_freq, train_df$q1_freq/train_df$word_match)
 train_df$q1_q2_freq_2w     <- pmax(train_df$q2_freq/ train_df$q1_freq, train_df$q1_freq/train_df$q2_freq)
 train_df$word_3gram_2w     <- pmax(train_df$word_match/ train_df$q_3gram_jaccard, train_df$q_3gram_jaccard/train_df$word_match)
 train_df$tfidf_q1_2w       <- pmax(train_df$tfidf_word_match/ train_df$q1_freq, train_df$q1_freq/train_df$tfidf_word_match)
 train_df$qpmin_q2_freq_2w  <- pmax(train_df$q2_freq/ train_df$q_MatchedWords_ratio_to_q_pmin, train_df$q_MatchedWords_ratio_to_q_pmin/train_df$q2_freq)
 train_df$word_q2_2w        <- pmax(train_df$word_match/ train_df$q2_freq, train_df$q2_freq/train_df$word_match)
 train_df$tfidf_q2_2w       <- pmax(train_df$tfidf_word_match/ train_df$q2_freq, train_df$q2_freq/train_df$tfidf_word_match)
 train_df$q1_3gram_2w       <- pmax(train_df$q1_freq/ train_df$q_3gram_jaccard, train_df$q_3gram_jaccard/train_df$q1_freq)
 train_df$qpmin_q1_freq_2w  <- pmax(train_df$q1_freq/ train_df$q_MatchedWords_ratio_to_q_pmin, train_df$q_MatchedWords_ratio_to_q_pmin/train_df$q1_freq)
 train_df$tfidf_4gram_2w    <- pmax(train_df$tfidf_word_match/ train_df$q_4gram_jaccard, train_df$q_4gram_jaccard/train_df$tfidf_word_match)


 test_df$word_q1_2w        <- pmax(test_df$word_match/ test_df$q1_freq, test_df$q1_freq/test_df$word_match)
 test_df$q1_q2_freq_2w     <- pmax(test_df$q2_freq/ test_df$q1_freq, test_df$q1_freq/test_df$q2_freq)
 test_df$word_3gram_2w     <- pmax(test_df$word_match/ test_df$q_3gram_jaccard, test_df$q_3gram_jaccard/test_df$word_match)
 test_df$tfidf_q1_2w       <- pmax(test_df$tfidf_word_match/ test_df$q1_freq, test_df$q1_freq/test_df$tfidf_word_match)
 test_df$qpmin_q2_freq_2w  <- pmax(test_df$q2_freq/ test_df$q_MatchedWords_ratio_to_q_pmin, test_df$q_MatchedWords_ratio_to_q_pmin/test_df$q2_freq)
 test_df$word_q2_2w        <- pmax(test_df$word_match/ test_df$q2_freq, test_df$q2_freq/test_df$word_match)
 test_df$tfidf_q2_2w       <- pmax(test_df$tfidf_word_match/ test_df$q2_freq, test_df$q2_freq/test_df$tfidf_word_match)
 test_df$q1_3gram_2w       <- pmax(test_df$q1_freq/ test_df$q_3gram_jaccard, test_df$q_3gram_jaccard/test_df$q1_freq)
 test_df$qpmin_q1_freq_2w  <- pmax(test_df$q1_freq/ test_df$q_MatchedWords_ratio_to_q_pmin, test_df$q_MatchedWords_ratio_to_q_pmin/test_df$q1_freq)
 test_df$tfidf_4gram_2w    <- pmax(test_df$tfidf_word_match/ test_df$q_4gram_jaccard, test_df$q_4gram_jaccard/test_df$tfidf_word_match)


train_cols <- c("id","word_q1_2w",      
                "q1_q2_freq_2w",   
                "word_3gram_2w",   
                "tfidf_q1_2w",     
                "qpmin_q2_freq_2w",
                "word_q2_2w",      
                "tfidf_q2_2w",     
                "q1_3gram_2w",     
                "qpmin_q1_freq_2w",
                "tfidf_4gram_2w"  )

test_cols <- c("test_id","word_q1_2w",      
                "q1_q2_freq_2w",   
                "word_3gram_2w",   
                "tfidf_q1_2w",     
                "qpmin_q2_freq_2w",
                "word_q2_2w",      
                "tfidf_q2_2w",     
                "q1_3gram_2w",     
                "qpmin_q1_freq_2w",
                "tfidf_4gram_2w"  )

write.csv(train_df[, train_cols], './input/train_interaction_features.csv', row.names=FALSE, quote = FALSE)
write.csv(test_df[, test_cols], './input/test_interaction_features.csv', row.names=FALSE, quote = FALSE)


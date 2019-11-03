library(syuzhet)
library(DT)

data <- fromJSON("./input/train.json")

vars <- setdiff(names(data), c("photos", "features"))
train_df <- map_at(data, vars, unlist) %>% tibble::as_tibble(.)

#train_df$id<-seq(1:length(train_df$building_id)) #numerical ids!


sentiment <- get_nrc_sentiment(train_df$description)
head(sentiment)

testdata <- fromJSON("./input/test.json")

testvars <- setdiff(names(testdata), c("photos", "features"))
test_df <- map_at(testdata, testvars, unlist) %>% tibble::as_tibble(.)

#test_df$id<-seq(1:length(test_df$building_id)) #numerical ids!


testsentiment <- get_nrc_sentiment(test_df$description)
head(testsentiment)

train_features <- as.data.frame( train_df$listing_id)
names(train_features) = "listing_id"
train_features <- cbind(train_features, sentiment)

test_features <- as.data.frame(test_df$listing_id)
names(test_features) = "listing_id"
test_features <- cbind(test_features, testsentiment)

head(train_features)
names(test_features)

write.csv(train_features,  './input/Prav_train_DescriptionSentimentFeatures.csv', row.names=FALSE, quote = FALSE)
write.csv(test_features,   './input/Prav_test_DescriptionSentimentFeatures.csv' , row.names=FALSE, quote = FALSE)

# ,"anger"
# ,"anticipation"
# ,"disgust"
# ,"fear"
# ,"joy"
# ,"sadness"
# ,"surprise"    
# ,"trust"
# ,"negative"
# ,"positive" 

cor(sentiment)



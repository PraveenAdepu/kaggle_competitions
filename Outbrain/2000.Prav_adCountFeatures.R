training <- read_csv("./input/trainingSet12.csv")
testing <- read_csv("./input/testingSet12.csv")

training <- as.data.table(training)
testing  <- as.data.table(testing)
names(training)
head(training,20)


training[ , adCount := .N, by = list(display_id)]
testing[ , adCount := .N, by = list(display_id)]
head(testing,20)

feature.columns <- c("display_id","ad_id","adCount")

training <- as.data.frame(training)

train.features <-subset(training,,feature.columns)
test.features <-subset(testing,,feature.columns)

head(train.features)
head(test.features)

write_csv(train.features, "./input/trainadCountFeatures.csv")
write_csv(test.features, "./input/testadCountFeatures.csv")


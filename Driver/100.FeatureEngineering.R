
train <- read_csv("./input/train.csv")

test <- read_csv("./input/test.csv")

submission <- read_csv("./input/sample_submission.csv")

head(submission, 10)

train <- as.data.table(train)
head(train,5)

test <- as.data.table(test)
head(test,5)

train <- as.data.frame(train)

require(ggvis)

train %>%
  ggvis(~target) %>%
  layer_histograms()

train %>%
  group_by(target) %>%
  summarise(targetDistribution = n()/595212)

train %>%
  ggvis(~ps_car_12, fill = ~target) %>%
  layer_points()




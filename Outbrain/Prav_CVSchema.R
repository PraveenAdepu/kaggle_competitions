# using Ash split file for CV
Prav_CVindices_5folds  <- fread("./CVSchema/splits.csv") 

head(Prav_CVindices_5folds)

unique(Prav_CVindices_5folds$is_train)

table(Prav_CVindices_5folds$is_train)

training <- read_csv("./input/trainingSet12.csv")

training <- left_join(training, Prav_CVindices_5folds, by = "display_id")

#63,502,376
X_build  <- training[training$is_train == 1,]
#23,639,355
X_valid  <- training[training$is_train == 0,]

X_build$is_train <- NULL
X_valid$is_train <- NULL

write_csv(X_build, "./input/trainingSet20_train.csv")
write_csv(X_valid, "./input/trainingSet20_valid.csv")



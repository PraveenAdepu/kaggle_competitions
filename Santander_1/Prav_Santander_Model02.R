rm(list=ls())

setwd("C:/Users/SriPrav/Documents/R/53Santander")
root_directory = "C:/Users/SriPrav/Documents/R/53Santander"


library(data.table)
library(dplyr)
library(xgboost)
library(Metrics)

train <- fread("./input/train_ver2.csv", colClasses=c(indrel_1mes="character", conyuemp="character"))
head(train)
# convert to dates, fast at data.table
train[, ':='(fecha_dato = as.Date(fecha_dato), fecha_alta = as.Date(fecha_alta))]

train <- as.data.frame(train)

# 24 target columns
target_cols = names(train %>% select(contains("ult1")))

train$MonthTotalProducts <- rowSums(train[,target_cols], na.rm=TRUE)

# Customers getting more than one product in a given month
summary(train$MonthTotalProducts)
table(train$MonthTotalProducts)

train_sub <- train %>% filter(MonthTotalProducts == 15)
head(train_sub)

target <- names(train[,target_cols])[max.col(train[,target_cols] == 1)]
train$target <- target

train_sub <- train %>% filter(MonthTotalProducts == 15)
head(train_sub)
# Here is one problem, we are getting random column as target

# To fix the issue, select the ties.method = first
# Limitation of this simple quick method, 

target <- names(train[,target_cols])[max.col(train[,target_cols] == 1,ties.method="first")]
train$targetLabel <- target

train_sub <- train %>% filter(MonthTotalProducts == 15)
head(train_sub)

# validate the target, all seems in correct order

test <- fread("./input/test_ver2.csv", colClasses=c(indrel_1mes="character", conyuemp="character"))
head(test)
# convert to dates, fast at data.table
test[, ':='(fecha_dato = as.Date(fecha_dato), fecha_alta = as.Date(fecha_alta))]

str(test)



CharacterType.columns <- sapply(test, is.character, value=TRUE)
character.features <- setdiff(names(test[, CharacterType.columns]),non.features)

char.feature.names <- c("ind_empleado","pais_residencia","sexo","ult_fec_cli_1t","indrel_1mes"
                        ,"tiprel_1mes","indresi","indext","conyuemp","canal_entrada","indfall"
                        ,"nomprov","segmento")

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in char.feature.names) {
  if (class(train[[f]])=="character") {
    cat("char feature : ", f ,"\n")
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}



test <- as.data.frame(test)

train[,feature.names][is.na(train[,feature.names])] <- 0
test[,feature.names][is.na(test[,feature.names])]   <- 0

target.levels <- factor(train$targetLabel, levels=target_cols, ordered=TRUE)

train$target <- as.integer(target.levels)-1

head(train)

non.features <- c("fecha_dato","ncodpers","fecha_alta","targetLabel","target")
feature.names <- setdiff(names(test),non.features)

# Cross validation Strategy
# Time based split to mimic real time forecast in production environment
# Last month data as validation set

X_build = train %>% filter(fecha_dato <= "2016-05-28")
X_valid = train %>% filter(fecha_dato == "2016-05-28")

X_build$target[is.na(X_build$target)] <- 0
X_valid$target[is.na(X_valid$target)] <- 0

dtrain <- xgb.DMatrix(data = as.matrix(X_build[, feature.names]), label = X_build$target)
dval   <- xgb.DMatrix(data = as.matrix(X_valid[, feature.names]), label = X_valid$target)
watchlist <- list( val = dval,train = dtrain)


n_class <- length(target_cols)

sort(unique(X_build$target))

cv          = 1
bags        = 1
nround.cv   = 5
printeveryn = 1
seed        = 201808


map7 <- function(preds, dtrain) {
  labels <- as.list(getinfo(dtrain,"label"))
  num.class = n_class
  pred <- matrix(preds, nrow = num.class)
  top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-6)]-1))
  top <- split(top, 1:NROW(top))
  
  map <- mapk(7, labels, top)
  return(list(metric = "map7", value = map))
}

## for all remaining models, use same parameters 

param <- list(  "objective"        = "multi:softprob",
                "booster"          = "gbtree",
                #"eval_metric"      = "mlogloss",
                "tree_method"      = "approx",
                "max_depth"        = 5,     
                "eta"              = 0.05, 
                "subsample"        = 0.7,  
                "colsample_bytree" = 0.7,
                "min_child_weight" = 3     
                
)


cat("X_build training Processing\n")
XGModel <- xgb.train(   params              = param,
                        feval               = map7, # custom MAP@k metric
                        num_class           = n_class,
                        data                = dtrain,
                        watchlist           = watchlist,
                        nrounds             = nround.cv ,
                        print_every_n       = printeveryn,
                        verbose             = TRUE,
                        nthread             = 12,
                        set.seed            = seed
)

#########################################################################################################################################
#########################################################################################################################################
# Model Run time - 10 hrs, 12 threads
# [23:58:54] Tree method is automatically selected to be 'approx' for faster speed. to use old behavior(exact greedy algorithm on single machine), set tree_method to 'exact'

# [1]	val-map7:0.825302	train-map7:0.834661 

# Prav - check if there is a bug in map7 metric function
#########################################################################################################################################
#########################################################################################################################################


cat("X_val prediction Processing\n")

pred_cv        <- predict(XGModel, as.matrix(X_valid[, feature.names]))
pred_cv_matrix <- data.table(matrix(pred_cv, ncol = n_class, byrow = T))
pred_cv_matrix <- as.data.frame(pred_cv_matrix)

names(pred_cv_matrix) <- target_cols

pred_cv_forecasts <- cbind(X_valid[,c("fecha_dato","ncodpers","target","targetLabel")],pred_cv_matrix)

head(pred_cv_forecasts)

cat("CV TestingSet prediction Processing\n")
pred_test  <- predict(XGModel, as.matrix(test[,feature.names]))
pred_test_matrix <- data.table(matrix(pred_test, ncol = n_class, byrow = T))
pred_test_matrix <- as.data.frame(pred_test_matrix)
names(pred_test_matrix) <- target_cols

head(test)

pred_test_forecasts <- cbind(test[,c("fecha_dato","ncodpers")],pred_test_matrix)

head(pred_test_forecasts)

#########################################################################################################
model = xgb.dump(XGModel, with_stats=TRUE)
names = dimnames(X_build[,feature.names])[[2]]
importance_matrix = xgb.importance( names , model = XGModel) 
xgb.plot.importance(importance_matrix)
impMatrix <- as.data.frame(importance_matrix)
# Top 5 features from importance matrix
head(impMatrix,5)

# #########################################################################################################






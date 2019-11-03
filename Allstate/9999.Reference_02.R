# For all those R users that want a competitive starter
# a shameless port of Faron's super python script to R
# https://www.kaggle.com/mmueller/allstate-claims-severity/yet-another-xgb-starter/code
# scores 1128 on public leaderboard but produced 1126 on my local run

library(data.table)
library(Matrix)
library(xgboost)
library(Metrics)

ID = 'id'
TARGET = 'loss'
SEED = 0

LETTERS_AY <- LETTERS[-length(LETTERS)]
LETTERS702 <- c(LETTERS_AY, sapply(LETTERS_AY, function(x) paste0(x, LETTERS_AY)), "ZZ")

TRAIN_FILE = "./input/train.csv"
TEST_FILE = "./input/test.csv"

train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)
train_ids <- train[,ID, with = FALSE][[ID]] # gotta love this style. 
test_ids <- test[,ID, with = FALSE][[ID]]
y_train = log(train[,TARGET, with = FALSE])[[TARGET]]

train[, c(ID, TARGET) := NULL]
test[, c(ID) := NULL]

ntrain = nrow(train)
train_test = rbind(train, test)

features = setdiff(names(train), c("id", "loss")) # just in case

for (f in features) {
  if (class(train_test[[f]])=="character") {
    levels <- intersect(LETTERS702, unique(train_test[[f]])) # get'em ordered!
    labels <- match(levels, LETTERS702)
    #train_test[[f]] <- factor(train_test[[f]], levels=levels) # uncomment this for one-hot
    train_test[[f]] <- as.integer(as.character(factor(train_test[[f]], levels=levels, labels = labels))) # comment this one away for one-hot
  }
}


x_train = train_test[1:ntrain,]
x_test = train_test[(ntrain+1):nrow(train_test),]

rm(train, test, train_test); gc()

dtrain.sparse <- sparse.model.matrix( ~ .-1, data = x_train)
dtest.sparse <- sparse.model.matrix( ~ .-1, data = x_test)

dim(dtrain.sparse); dim(dtest.sparse)
dtrain <- xgb.DMatrix(dtrain.sparse, label=y_train)
dtest <-  xgb.DMatrix(dtest.sparse)


xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

logcoshobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  grad <- tanh(preds-labels)
  hess <- 1-grad*grad
  return(list(grad = grad, hess = hess))
}

cauchyobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  c <- 3  #the lower the "slower/smoother" the loss is. Cross-Validate.
  x <-  preds-labels
  grad <- x / (x^2/c^2+1)
  hess <- -c^2*(x^2-c^2)/(x^2+c^2)^2
  return(list(grad = grad, hess = hess))
}


fairobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  c <- 2 #the lower the "slower/smoother" the loss is. Cross-Validate.
  x <-  preds-labels
  grad <- c*x / (abs(x)+c)
  hess <- c^2 / (abs(x)+c)^2
  return(list(grad = grad, hess = hess))
}

xgb_params = list(
  seed = 0,
  colsample_bytree = 0.6,#0.7
  #subsample = 0.7,
  eta = 0.075,
  objective = logcoshobj, #fairobj #logcoshobj, #cauchyobj #,#'reg:linear',
  eval_metric = xg_eval_mae, # "mae"
  max_depth = 6, #6
  num_parallel_tree = 1,
  min_child_weight = 1,
  alpha=10, #8,9,10
  base_score = 7.65
)

res = xgb.cv(xgb_params,
             dtrain,
             prediction =T,
             nrounds=100, #2000 for local run
             nfold=5,
             early_stopping_rounds=15,
             print_every_n = 10,
             verbose= 1,
             #obj = fairobj, #logcoshobj, #cauchyobj #
             #feval = xg_eval_mae, #"mae" 
             maximize=FALSE)

best_nrounds = res$best_iteration
cv_mean = res$evaluation_log$test_error_mean[best_nrounds]
cv_std = res$evaluation_log$test_error_std[best_nrounds]
cat(paste0('CV-Mean: ',cv_mean,' ', cv_std))
cv_mean_char <- as.character(round(cv_mean,4))

gbdt = xgb.train(xgb_params, dtrain, nrounds=best_nrounds, maximize = F)
out <- exp(predict(gbdt,dtest))
stacking <- rbindlist(list(data.frame(train_ids, res$pred),
                           data.frame(test_ids, out) ))
names(stacking) <- c("id", cv_mean_char)

submission = data.frame(id=test_ids, loss=out)

filename_stack <- paste("stack_xgbCV_dmi3kno", cv_mean_char, format(Sys.time(), "%Y%m%d%H%M%S"), sep = "_")
filename_res <- paste("res_xgbCV_dmi3kno", cv_mean_char, format(Sys.time(), "%Y%m%d%H%M%S"), sep = "_")
write.csv(stacking, paste0(filename_stack,'.csv',collapse = ""), row.names=FALSE, quote = FALSE)
write.csv(submission, paste0(filename_res,'.csv',collapse = ""), row.names=FALSE, quote = FALSE)  
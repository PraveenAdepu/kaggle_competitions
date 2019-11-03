rm(list=ls())

setwd("C:/Users/SriPrav/Documents/R/53Santander")
root_directory = "C:/Users/SriPrav/Documents/R/53Santander"


## setting file paths and seed (edit the paths before running)
path_train_file <- "./input/train_ver2.csv"
path_test_file  <- "./input/test_ver2.csv"
path_preds      <- "./input/preds.csv"

## put your favourite number as seed
seed <- 123
set.seed(seed)

## loading libraries
library(data.table)
library(dplyr)
library(xgboost)

## loading raw data
train <- fread(path_train_file, showProgress = T) # 13647309 * 48
test <- fread(path_test_file, showProgress = T)   #   929615 * 24

## removing five products
train[, ind_ahor_fin_ult1 := NULL]
train[, ind_aval_fin_ult1 := NULL]
train[, ind_deco_fin_ult1 := NULL]
train[, ind_deme_fin_ult1 := NULL]
train[, ind_viv_fin_ult1 := NULL]

## extracting train data of each product and rbinding them together with single multiclass label
i <- 0
target_cols <- names(train)[which(regexpr("ult1", names(train)) > 0)]

for (target_col in target_cols)
{
  i <- i + 1
  
  S <- paste0("train", i, " <- train[", target_col, " > 0]")
  eval(parse(text = S))
}

# rm(train)
gc()

i = 1
for (i in 1:19)
{
  S1 <- paste0("train", i, " <- train", i, "[, !target_cols, with = F]")
  eval(parse(text = S1))
  
  S2 <- paste0("train", i, "[, target := ", i-1, "]")
  eval(parse(text = S2))
}

train_full <- rbind(train1, train2, train3, train4, train5, train6, train7, train8, train9, train10,
                    train11, train12, train13, train14, train15, train16, train17, train18, train19)   # 19851490 * 25

rm(train1, train2, train3, train4, train5, train6, train7, train8, train9, train10,
   train11, train12, train13, train14, train15, train16, train17, train18, train19)
gc()


## rbinding train and test data
X_panel <- rbind(train_full, test, use.names = T, fill = T) # 20781105 * 25

tail(X_panel)

## adding corresponding numeric months (1-18) to fecha_dato
X_panel[, month := as.numeric(as.factor(fecha_dato))]

## creating user-product matrix
X_user_target <- dcast(X_panel[!is.na(target)], ncodpers + month ~ target, length, value.var = "target", fill = 0)

head(X_user_target) # 11077019 * 21

## creating product lag-variables of order-12 and merging with data
X_user_target[, month := month + 1]

setnames(X_user_target,
         c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
           "16", "17", "18"),
         c("prev_0", "prev_1", "prev_2", "prev_3", "prev_4", "prev_5", "prev_6", "prev_7",
           "prev_8", "prev_9", "prev_10", "prev_11", "prev_12", "prev_13", "prev_14", "prev_15",
           "prev_16", "prev_17", "prev_18"))

X_panel <- merge(X_panel, X_user_target, all.x = T, by = c("ncodpers", "month")) # 20781105 * 45

tail(X_panel)


X_panel[is.na(X_panel)] <- 0

## cleaning raw features
X_panel[, ":="(ind_empleado = as.numeric(as.factor(ind_empleado)),
               pais_residencia = as.numeric(as.factor(pais_residencia)),
               sexo = as.numeric(as.factor(sexo)),
               year_joining = year(as.Date(fecha_alta)),
               month_joining = month(as.Date(fecha_alta)),
               fecha_alta = as.numeric(as.Date(fecha_alta) - as.Date("2016-05-31")),
               ult_fec_cli_1t = ifelse(ult_fec_cli_1t == "", 0, 1),
               indrel_1mes = as.numeric(as.factor(indrel_1mes)),
               tiprel_1mes = as.numeric(as.factor(tiprel_1mes)),
               indresi = as.numeric(as.factor(indresi)),
               indext = as.numeric(as.factor(indext)),
               conyuemp = as.numeric(as.factor(conyuemp)),
               canal_entrada = as.numeric(as.factor(canal_entrada)),
               indfall = as.numeric(as.factor(indfall)),
               tipodom = NULL,
               cod_prov = as.numeric(as.factor(cod_prov)),
               nomprov = NULL,
               segmento = as.numeric(as.factor(segmento)))]



## creating train and test data for June-15 (seasonality) and May-16 (trend) models
#X_train_1 <- X_panel[fecha_dato %in% c("2015-06-28")]
X_train_2 <- X_panel[fecha_dato %in% c("2016-01-28","2016-02-28","2016-03-28","2016-04-28","2016-05-28")]

#X_test_1 <- X_panel[fecha_dato %in% c("2016-06-28")]
X_test_2 <- X_panel[fecha_dato %in% c("2016-06-28")]

X_test_order <- X_test_2$ncodpers


## creating binary flag for new products, test data will always have 1 since we need to predict new products
#X_train_1$flag_new <- 0
X_train_2$flag_new <- 0

#X_test_1$flag_new <- 1
X_test_2$flag_new <- 1

for (i in 0:18)
{
  
  S2 <- paste0("X_train_2$flag_new[X_train_2$prev_", i, " == 0 & X_train_2$target == ", i, "] <- 1")
  eval(parse(text = S2))
}




names(X_train_2)
head(X_train_2)

X_train_2 <- as.data.frame(X_train_2)
X_test_2  <- as.data.frame(X_test_2)

feature.names     <- names(X_train_2[,-which(names(X_train_2) %in% c("ncodpers","month", "fecha_dato","target" ))])
## extracting labels


X_build = X_train_2 %>% filter(fecha_dato <= "2016-05-28")
X_valid = X_train_2 %>% filter(fecha_dato == "2016-05-28")



dtrain <- xgb.DMatrix(data = as.matrix(X_build[, feature.names]), label = X_build$target)
dval   <- xgb.DMatrix(data = as.matrix(X_valid[, feature.names]), label = X_valid$target)
watchlist <- list( val = dval,train = dtrain)


n_class <- length(target_cols)

sort(unique(X_build$target))


cv          = 1
bags        = 1
nround.cv   = 50
printeveryn = 10
seed        = 201808

## for all remaining models, use same parameters 

param <- list(  "objective"        = "multi:softprob",
                "booster"          = "gbtree",
                "eval_metric"      = "mlogloss",
                "tree_method"      = "approx",
                "max_depth"        = 5,     
                "eta"              = 0.05, 
                "subsample"        = 0.7,  
                "colsample_bytree" = 0.7,
                "min_child_weight" = 3     
                
)


cat("X_build training Processing\n")
XGModel <- xgb.train(   params              = param,
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

# [14:52:23] Tree method is selected to be 'approx'
# [1]	val-mlogloss:2.730801	train-mlogloss:2.729804 
# [11]	val-mlogloss:1.849005	train-mlogloss:1.844938 
# [21]	val-mlogloss:1.493262	train-mlogloss:1.488605 
# [31]	val-mlogloss:1.283373	train-mlogloss:1.278857 
# [41]	val-mlogloss:1.148632	train-mlogloss:1.144127 
# [50]	val-mlogloss:1.066131	train-mlogloss:1.061729 

#########################################################################################################################################
#########################################################################################################################################


cat("X_val prediction Processing\n")

pred_cv        <- predict(XGModel, as.matrix(X_valid[, feature.names]))
pred_cv_matrix <- data.table(matrix(pred_cv, ncol = n_class, byrow = T))
pred_cv_matrix <- as.data.frame(pred_cv_matrix)

names(pred_cv_matrix) <- target_cols

pred_cv_forecasts <- cbind(X_valid[,c("fecha_dato","ncodpers","target")],pred_cv_matrix)

head(pred_cv_forecasts)

cat("CV TestingSet prediction Processing\n")
pred_test  <- predict(XGModel, as.matrix(X_test_2[,feature.names]))
pred_test_matrix <- data.table(matrix(pred_test, ncol = n_class, byrow = T))
pred_test_matrix <- as.data.frame(pred_test_matrix)
names(pred_test_matrix) <- target_cols

head(test)

pred_test_forecasts <- cbind(X_test_2[,c("fecha_dato","ncodpers")],pred_test_matrix)

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

write.csv(pred_cv_forecasts,  './input/Prav.xgb.Model04-cv.csv', row.names=FALSE, quote = FALSE)
write.csv(pred_test_forecasts, './input/Prav.xgb.Model04-test.csv', row.names=FALSE, quote = FALSE)


##########################################################################################################


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
library(Metrics)

## loading raw data
train <- fread(path_train_file, showProgress = T) # 13647309 * 48
test  <- fread(path_test_file, showProgress = T)   #   929615 * 24

# train[, ':='(fecha_dato = as.Date(fecha_dato), fecha_alta = as.Date(fecha_alta))]
# test[, ':='(fecha_dato = as.Date(fecha_dato), fecha_alta = as.Date(fecha_alta))]

##############################################################################################################################
##############################################################################################################################

age.median <- median(train$age,na.rm=TRUE)
ind_nuevo.median <- median(train$ind_nuevo,na.rm=TRUE)

train$antiguedad[train$antiguedad<0]      <- 0
antiguedad.min               <- min(train$antiguedad,na.rm=TRUE)
fecha_alta.median            <- median(train$fecha_alta,na.rm=TRUE)
indrel.median                <- median(train$indrel,na.rm=TRUE)
cod_prov.median              <- median(train$cod_prov,na.rm=TRUE)
ind_actividad_cliente.median <- median(train$ind_actividad_cliente,na.rm=TRUE)
renta.median                 <- median(train$renta,na.rm=TRUE)

train$age[is.na(train$age)]               <- age.median
train$ind_nuevo[is.na(train$ind_nuevo)]   <- ind_nuevo.median
train$antiguedad[is.na(train$antiguedad)] <- antiguedad.min
train$fecha_alta[is.na(train$fecha_alta)] <- fecha_alta.median
train$indrel[is.na(train$indrel)]         <- indrel.median
train$cod_prov[is.na(train$cod_prov)]     <- cod_prov.median
train$ind_actividad_cliente[is.na(train$ind_actividad_cliente)] <- ind_actividad_cliente.median
train$renta[is.na(train$renta)]          <- renta.median



train$ind_nomina_ult1[is.na(train$ind_nomina_ult1)]     <- 0
train$ind_nom_pens_ult1[is.na(train$ind_nom_pens_ult1)] <- 0
##############################################################################################################################
# Apply missing values from train to train
##############################################################################################################################

train$indfall[train$indfall==""]                 <- "N"
train$tiprel_1mes[train$tiprel_1mes==""]         <- "U"
train$indrel_1mes[train$indrel_1mes==""]         <- "1"
train$indrel_1mes[train$indrel_1mes=="P"]        <- "5" 
train$pais_residencia[train$pais_residencia==""] <- "U"
train$sexo[train$sexo==""]                       <- "U"
train$ult_fec_cli_1t[train$ult_fec_cli_1t==""]   <- "U"
train$ind_empleado[train$ind_empleado==""]       <- "U"
train$indext[train$indext==""]                   <- "U"
train$indresi[train$indresi==""]                 <- "U"
train$conyuemp[train$conyuemp==""]               <- "U"
train$segmento[train$segmento==""]               <- "U"
################################################################################################################################
# Apply missing values from train to test, In production phase, we need to save these missing values and use to test set fill
# to maintain the same consistence
################################################################################################################################


test$age[is.na(test$age)]               <- age.median
test$ind_nuevo[is.na(test$ind_nuevo)]   <- ind_nuevo.median

test$antiguedad[test$antiguedad<0]      <- 0
test$antiguedad[is.na(test$antiguedad)] <- antiguedad.min
test$fecha_alta[is.na(test$fecha_alta)] <- fecha_alta.median

test$indrel[is.na(test$indrel)]         <- indrel.median
test$cod_prov[is.na(test$cod_prov)]     <- cod_prov.median
test$ind_actividad_cliente[is.na(test$ind_actividad_cliente)] <- ind_actividad_cliente.median
test$renta[is.na(test$renta)]           <- renta.median

test$indfall[test$indfall==""]                 <- "N"
test$tiprel_1mes[test$tiprel_1mes==""]         <- "U"
test$indrel_1mes[test$indrel_1mes==""]         <- "1"
test$indrel_1mes[test$indrel_1mes=="P"]        <- "5" 
test$pais_residencia[test$pais_residencia==""] <- "U"
test$sexo[test$sexo==""]                       <- "U"
test$ult_fec_cli_1t[test$ult_fec_cli_1t==""]   <- "U"
test$ind_empleado[test$ind_empleado==""]       <- "U"
test$indext[test$indext==""]                   <- "U"
test$indresi[test$indresi==""]                 <- "U"
test$conyuemp[test$conyuemp==""]               <- "U"
test$segmento[test$segmento==""]               <- "U"

################################################################################################################################
################################################################################################################################

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
  
  S2 <- paste0("train", i, "[, targetLabel := '", target_col, "']")
  eval(parse(text = S2))
  
}

# rm(train)
gc()


for (i in 1:19)
{
  S1 <- paste0("train", i, " <- train", i, "[, !target_cols, with = F]")
  eval(parse(text = S1))
  
  S2 <- paste0("train", i, "[, target := ", i-1, "]")
  eval(parse(text = S2))
}

train_full <- rbind(train1, train2, train3, train4, train5, train6, train7, train8, train9, train10,
                    train11, train12, train13, train14, train15, train16, train17, train18, train19)   # 19851490 * 26

rm(train1, train2, train3, train4, train5, train6, train7, train8, train9, train10,
   train11, train12, train13, train14, train15, train16, train17, train18, train19)
rm(train)
gc()


## rbinding train and test data
X_panel <- rbind(train_full, test, use.names = T, fill = T) # 20781105 * 25

rm(test,train_full)
gc()
tail(X_panel)

## adding corresponding numeric months (1-18) to fecha_dato
X_panel[, month := as.numeric(as.factor(fecha_dato))]

## creating user-product matrix
X_user_target <- dcast(X_panel[!is.na(target)], ncodpers + month ~ target, length, value.var = "target", fill = 0)

head(X_user_target)
## creating product lag-variables of order-12 and merging with data


#####################################################################################################################
# Feature Engineering - Lag Features ################################################################################
#####################################################################################################################

# lag - 1, hence adding one month to join to dataset
X_user_target_lag1 <- copy(X_user_target)
head(X_user_target_lag1) # 11077019 * 21
X_user_target_lag1[, month := month + 1]
head(X_user_target_lag1)

setnames(X_user_target_lag1,
         c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
           "16", "17", "18"),
         c("prev_0", "prev_1", "prev_2", "prev_3", "prev_4", "prev_5", "prev_6", "prev_7",
           "prev_8", "prev_9", "prev_10", "prev_11", "prev_12", "prev_13", "prev_14", "prev_15",
           "prev_16", "prev_17", "prev_18"))

X_panel <- merge(X_panel, X_user_target_lag1, all.x = T, by = c("ncodpers", "month")) # 20781105 * 45
rm(X_user_target_lag1)
gc()
tail(X_panel)

# lag - 2, hence adding one more month previously added month to join to dataset

X_user_target_lag2 <- copy(X_user_target)
head(X_user_target_lag2)
X_user_target_lag2[, month := month + 2]
head(X_user_target_lag2)

setnames(X_user_target_lag2,
         c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
           "16", "17", "18"),
         c("prev_lag2_0", "prev_lag2_1", "prev_lag2_2", "prev_lag2_3", "prev_lag2_4", "prev_lag2_5", "prev_lag2_6", "prev_lag2_7",
           "prev_lag2_8", "prev_lag2_9", "prev_lag2_10", "prev_lag2_11", "prev_lag2_12", "prev_lag2_13", "prev_lag2_14", "prev_lag2_15",
           "prev_lag2_16", "prev_lag2_17", "prev_lag2_18"))

X_panel <- merge(X_panel, X_user_target_lag2, all.x = T, by = c("ncodpers", "month")) # 20781105 * 45
rm(X_user_target_lag2)
gc()
tail(X_panel)


# lag3, hence adding one more month previously added month to join to dataset

X_user_target_lag3 <- copy(X_user_target)
head(X_user_target_lag3)
X_user_target_lag3[, month := month + 3]
head(X_user_target_lag3)

setnames(X_user_target_lag3,
         c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
           "16", "17", "18"),
         c("prev_lag3_0", "prev_lag3_1", "prev_lag3_2", "prev_lag3_3", "prev_lag3_4", "prev_lag3_5", "prev_lag3_6", "prev_lag3_7",
           "prev_lag3_8", "prev_lag3_9", "prev_lag3_10", "prev_lag3_11", "prev_lag3_12", "prev_lag3_13", "prev_lag3_14", "prev_lag3_15",
           "prev_lag3_16", "prev_lag3_17", "prev_lag3_18"))

X_panel <- merge(X_panel, X_user_target_lag3, all.x = T, by = c("ncodpers", "month")) # 20781105 * 45
rm(X_user_target_lag3)
gc()
tail(X_panel)


# lag4, hence adding one more month previously added month to join to dataset

X_user_target_lag4 <- copy(X_user_target)
head(X_user_target_lag4)
X_user_target_lag4[, month := month + 3]
head(X_user_target_lag4)

setnames(X_user_target_lag4,
         c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
           "16", "17", "18"),
         c("prev_lag4_0", "prev_lag4_1", "prev_lag4_2", "prev_lag4_3", "prev_lag4_4", "prev_lag4_5", "prev_lag4_6", "prev_lag4_7",
           "prev_lag4_8", "prev_lag4_9", "prev_lag4_10", "prev_lag4_11", "prev_lag4_12", "prev_lag4_13", "prev_lag4_14", "prev_lag4_15",
           "prev_lag4_16", "prev_lag4_17", "prev_lag4_18"))

X_panel <- merge(X_panel, X_user_target_lag4, all.x = T, by = c("ncodpers", "month")) # 20781105 * 45
rm(X_user_target_lag4)
gc()
tail(X_panel)


# lag5, hence adding one more month previously added month to join to dataset

X_user_target_lag5 <- copy(X_user_target)
head(X_user_target_lag5)
X_user_target_lag5[, month := month + 3]
head(X_user_target_lag5)

setnames(X_user_target_lag5,
         c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
           "16", "17", "18"),
         c("prev_lag5_0", "prev_lag5_1", "prev_lag5_2", "prev_lag5_3", "prev_lag5_4", "prev_lag5_5", "prev_lag5_6", "prev_lag5_7",
           "prev_lag5_8", "prev_lag5_9", "prev_lag5_10", "prev_lag5_11", "prev_lag5_12", "prev_lag5_13", "prev_lag5_14", "prev_lag5_15",
           "prev_lag5_16", "prev_lag5_17", "prev_lag5_18"))

X_panel <- merge(X_panel, X_user_target_lag5, all.x = T, by = c("ncodpers", "month")) # 20781105 * 45
rm(X_user_target_lag5)
gc()
tail(X_panel)


# lag6, hence adding one more month previously added month to join to dataset

X_user_target_lag6 <- copy(X_user_target)
head(X_user_target_lag6)
X_user_target_lag6[, month := month + 3]
head(X_user_target_lag6)

setnames(X_user_target_lag6,
         c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
           "16", "17", "18"),
         c("prev_lag6_0", "prev_lag6_1", "prev_lag6_2", "prev_lag6_3", "prev_lag6_4", "prev_lag6_5", "prev_lag6_6", "prev_lag6_7",
           "prev_lag6_8", "prev_lag6_9", "prev_lag6_10", "prev_lag6_11", "prev_lag6_12", "prev_lag6_13", "prev_lag6_14", "prev_lag6_15",
           "prev_lag6_16", "prev_lag6_17", "prev_lag6_18"))

X_panel <- merge(X_panel, X_user_target_lag6, all.x = T, by = c("ncodpers", "month")) # 20781105 * 45
rm(X_user_target_lag6)
rm(X_user_target)
gc()
tail(X_panel)

#####################################################################################################################
# Feature Engineering - Lag Features ################################################################################
#####################################################################################################################

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



## Because lag6 features, we don't require to take first 6 months dataset for training

X_train <- X_panel[fecha_dato %in% c("2015-07-28","2015-08-28","2015-09-28","2015-10-28","2015-11-28","2015-12-28","2016-01-28","2016-02-28","2016-03-28","2016-04-28","2016-05-28")]

X_test <- X_panel[fecha_dato %in% c("2016-06-28")]

sort(unique(X_train$fecha_dato))

## creating binary flag for new products, test data will always have 1 since we need to predict new products
#X_train_1$flag_new <- 0
X_train$flag_new <- 0

#X_test_1$flag_new <- 1
X_test$flag_new <- 1

for (i in 0:18)
{
  
  S2 <- paste0("X_train$flag_new[X_train$prev_", i, " == 0 & X_train$target == ", i, "] <- 1")
  eval(parse(text = S2))
}


X_train <- as.data.frame(X_train)
X_test  <- as.data.frame(X_test)

# 137 features
feature.names     <- names(X_train[,-which(names(X_train) %in% c("ncodpers","month", "fecha_dato","target","targetLabel" ))])

rm(X_panel)
gc()

####################################################################################################################
# Cross validation using last month of train dataset, time based split

X_build = X_train %>% filter(fecha_dato <  "2016-05-28")
X_valid = X_train %>% filter(fecha_dato == "2016-05-28")


dtrain <- xgb.DMatrix(data = as.matrix(X_build[, feature.names]), label = X_build$target)
dval   <- xgb.DMatrix(data = as.matrix(X_valid[, feature.names]), label = X_valid$target)
watchlist <- list( val = dval,train = dtrain)


n_class <- length(target_cols)
sort(unique(X_build$target))


cv          = 1
bags        = 1
nround.cv   = 75
printeveryn = 10
seed        = 201808

## for all remaining models, use same parameters 

param <- list(  "objective"        = "multi:softprob",
                "booster"          = "gbtree",
                "eval_metric"      = "mlogloss",
                "tree_method"      = "approx",
                "max_depth"        = 5,     
                "eta"              = 0.1, 
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
                        nthread             = 25,
                        set.seed            = seed
)

#########################################################################################################################################
#########################################################################################################################################
# Model Run time - 10 hrs, 12 threads

# [23:42:58] Tree method is selected to be 'approx'
# [1]	  val-mlogloss:2.480139	train-mlogloss:2.482277 
# [11]	val-mlogloss:1.383103	train-mlogloss:1.390343 
# [21]	val-mlogloss:1.075702	train-mlogloss:1.084568 
# [31]	val-mlogloss:0.945477	train-mlogloss:0.954678 
# [41]	val-mlogloss:0.884074	train-mlogloss:0.893551 
# [51]	val-mlogloss:0.853189	train-mlogloss:0.862726 
# [61]	val-mlogloss:0.835774	train-mlogloss:0.845323 
# [71]	val-mlogloss:0.825461	train-mlogloss:0.834947 
# [75]	val-mlogloss:0.822316	train-mlogloss:0.831787 

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
pred_test  <- predict(XGModel, as.matrix(X_test[,feature.names]))
pred_test_matrix <- data.table(matrix(pred_test, ncol = n_class, byrow = T))
pred_test_matrix <- as.data.frame(pred_test_matrix)
names(pred_test_matrix) <- target_cols


pred_test_forecasts <- cbind(X_test[,c("fecha_dato","ncodpers")],pred_test_matrix)

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

pred_cv_forecasts_long <- melt(pred_cv_forecasts, id.vars=c("fecha_dato", "ncodpers", "target", "targetLabel"))
head(pred_cv_forecasts_long)
pred_cv_forecasts_long <- as.data.table(pred_cv_forecasts_long)
setorderv( pred_cv_forecasts_long, c("fecha_dato","ncodpers","targetLabel","value"), c(1,1,1,-1)  ) #Sort by -prob

pred_cv_forecasts_long_top7 <- pred_cv_forecasts_long[, head(.SD, 7), by = c("fecha_dato","ncodpers","targetLabel")]
rm(pred_cv_forecasts_long);
gc()

# in submission format
pred_cv_submission <- pred_cv_forecasts_long_top7[,.(added_products=paste0(variable,collapse=" ")), keyby=c("ncodpers","targetLabel") ]; gc() #Build submission
rm(pred_cv_forecasts_long_top7);
gc()
head(pred_cv_submission)

# MAP@7 - validation score
validation_mapk7_acore <- mapk(7,pred_cv_submission$targetLabel,strsplit(pred_cv_submission$added_products," "))
cat("validation mapk7 score - ", validation_mapk7_acore)

# #########################################################################################################


write.csv(pred_cv_forecasts,  './input/Prav.xgb.Model06-cv.csv', row.names=FALSE, quote = FALSE)
write.csv(pred_test_forecasts, './input/Prav.xgb.Model06-test.csv', row.names=FALSE, quote = FALSE)
write.csv(impMatrix, './input/Prav.xgb.Model06-VariableImportance.csv', row.names=FALSE, quote = FALSE)

##########################################################################################################


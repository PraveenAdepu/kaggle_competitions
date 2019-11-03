
# rm(list=ls())
# 
# setwd("C:/Users/SriPrav/Documents/R/53Santander")
# root_directory = "C:/Users/SriPrav/Documents/R/53Santander"


## setting file paths and seed (edit the paths before running)
path_train_file <- paste0(root_directory,"/input/train_ver2.csv")
path_test_file  <- paste0(root_directory,"/input/test_ver2.csv")
path_preds      <- paste0(root_directory,"/input/preds.csv")

## put your favourite number as seed
seed <- 123
set.seed(seed)

## loading libraries

library(data.table)
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(scales)


my_theme <- theme_bw() +
  theme(axis.title=element_text(size=12),
        plot.title=element_text(size=14,hjust = 0.5),
        axis.text =element_text(size=12,angle = 90, hjust = 1))


train <- fread(path_train_file, showProgress = T) # 13647309 * 48
test  <- fread(path_test_file, showProgress = T)   #   929615 * 24


# DQ check - both date formats are in Date formats
str(train)

# Check dates from train dataset, we need to 17 months of data from "2015-01-28" to "2016-05-28"
sort(unique(train$fecha_dato))

# get Month number from dates to check on Time Series trends

train$month <- month(train$fecha_dato)

# DQ check - check missing data - one the most common in data pipeline development process

sapply(train,function(x)any(is.na(x)))

# Missing information columns - 11 columns

# Age
# fecha_alta
# ind_nuevo
# antiguedad
# indrel
# tipodom
# cod_prov
# ind_actividad_cliente
# renta
# ind_nomina_ult1
# ind_nom_pens_ult1

na.percents <- train %>% 
                  summarise_all(funs(100*mean(is.na(.))))
na.percents <- melt(na.percents)

viz <- na.percents %>%
        filter(value > 0) %>%
        ggplot(aes(x=variable,y=(value/100))) + 
        geom_bar(stat="identity",position = "dodge",fill="#FF6666") + 
        labs(title = paste0("Missing Records by Feature")
             , x = "Feature"
             , y = "Missing Records")+
        scale_y_continuous(labels = scales::percent)+
        my_theme

print(viz)


##############################################################################################################################
# EDA - Age
##############################################################################################################################

viz <- ggplot(data=train,aes(x=age)) + 
        geom_bar(alpha=0.75,fill="#FF6666") +
        ggtitle("Age Distribution") + 
        my_theme
print(viz)

summary(train$age)

# Missing percentage
percent(round(sum(is.na(train$age))/nrow(train), digits=4))
# 0.2%, is actually not bad

# we got Age from 2 to 164, I am NOT sure of these low and very high values
# Option01 - Adjust these low and very high values or leave those as we don't know wether this is data quality or domain specific info

# Opiton02- use median to fill missing values
age.median <- median(train$age,na.rm=TRUE)
train$age[is.na(train$age)]  <- age.median


# DQ Check - missing value fill has worked, need to get 0%
percent(round(sum(is.na(train$age))/nrow(train), digits=4))

##############################################################################################################################
# EDA - Age
##############################################################################################################################


##############################################################################################################################
# EDA - ind_nuevo
##############################################################################################################################

viz <- ggplot(data=train,aes(x=ind_nuevo)) + 
  geom_bar(alpha=0.75,fill="#FF6666") +
  ggtitle("ind_nuevo Distribution") + 
  my_theme
print(viz)
summary(train$ind_nuevo)

# Missing percentage
percent(round(sum(is.na(train$ind_nuevo))/nrow(train), digits=4))
# 0.2%, is actually not bad

# we got Age from 2 to 164, I am NOT sure of these low and very high values
# Option01 - Adjust these low and very high values or leave those as we don't know wether this is data quality or domain specific info

# Opiton02- use median to fill missing values
ind_nuevo.median <- median(train$ind_nuevo,na.rm=TRUE)
train$ind_nuevo[is.na(train$ind_nuevo)]  <- ind_nuevo.median

# DQ Check - missing value fill has worked, need to get 0%
percent(round(sum(is.na(train$ind_nuevo))/nrow(train), digits=4))

##############################################################################################################################
# EDA - ind_nuevo
##############################################################################################################################


##############################################################################################################################
# EDA - antiguedad
##############################################################################################################################

viz <- ggplot(data=train,aes(x=antiguedad)) + 
  geom_density(alpha=0.75,fill="#FF6666") +
  ggtitle("antiguedad Distribution") + 
  my_theme
print(viz)
summary(train$antiguedad)

# Missing percentage
percent(round(sum(is.na(train$antiguedad))/nrow(train), digits=4))
# 0.2%, is actually not bad

# we got Age from 2 to 164, I am NOT sure of these low and very high values
# Option01 - Adjust these low and very high values or leave those as we don't know wether this is data quality or domain specific info

# Opiton02- use median to fill missing values

train$antiguedad[train$antiguedad<0]      <- 0
antiguedad.min <- min(train$antiguedad,na.rm=TRUE)
train$antiguedad[is.na(train$antiguedad)] <- antiguedad.min

# DQ Check - missing value fill has worked, need to get 0%
percent(round(sum(is.na(train$antiguedad))/nrow(train), digits=4))

##############################################################################################################################
# EDA - antiguedad
##############################################################################################################################

##############################################################################################################################
# EDA - fecha_alta
##############################################################################################################################


# Missing percentage
percent(round(sum(is.na(train$fecha_alta))/nrow(train), digits=4))
# 0.2%, is actually not bad

# we got Age from 2 to 164, I am NOT sure of these low and very high values
# Option01 - Adjust these low and very high values or leave those as we don't know wether this is data quality or domain specific info

# Opiton02- use median to fill missing values
fecha_alta.median <- median(train$fecha_alta,na.rm=TRUE)
train$fecha_alta[is.na(train$fecha_alta)] <- fecha_alta.median

# DQ Check - missing value fill has worked, need to get 0%
percent(round(sum(is.na(train$fecha_alta))/nrow(train), digits=4))

##############################################################################################################################
# EDA - fecha_alta
##############################################################################################################################

##############################################################################################################################
# EDA - indrel
##############################################################################################################################

summary(train$indrel)

# Missing percentage
percent(round(sum(is.na(train$indrel))/nrow(train), digits=4))
# 0.2%, is actually not bad

# we got Age from 2 to 164, I am NOT sure of these low and very high values
# Option01 - Adjust these low and very high values or leave those as we don't know wether this is data quality or domain specific info

# Opiton02- use median to fill missing values
indrel.median <- median(train$indrel,na.rm=TRUE)
train$indrel[is.na(train$indrel)] <- indrel.median

# DQ Check - missing value fill has worked, need to get 0%
percent(round(sum(is.na(train$indrel))/nrow(train), digits=4))

##############################################################################################################################
# EDA - indrel
##############################################################################################################################

##############################################################################################################################
# EDA - tipodom
##############################################################################################################################

summary(train$tipodom)

# Missing percentage
percent(round(sum(is.na(train$tipodom))/nrow(train), digits=4))
# 0.2%, is actually not bad

# Interesting, all values are 1 and NA, Zero Variance so we can exlude this column

##############################################################################################################################
# EDA - tipodom
##############################################################################################################################

##############################################################################################################################
# EDA - cod_prov
##############################################################################################################################

viz <- ggplot(data=train,aes(x=cod_prov)) + 
  geom_bar(alpha=0.75,fill="#FF6666") +
  ggtitle("cod_prov Distribution") + 
  my_theme
print(viz)
summary(train$cod_prov)

# Missing percentage
percent(round(sum(is.na(train$cod_prov))/nrow(train), digits=4))
# 0.69%, is actually not bad

cod_prov.median <- median(train$cod_prov,na.rm=TRUE)
train$cod_prov[is.na(train$cod_prov)] <- cod_prov.median

# DQ Check - missing value fill has worked, need to get 0%
percent(round(sum(is.na(train$cod_prov))/nrow(train), digits=4))

##############################################################################################################################
# EDA - cod_prov
##############################################################################################################################

##############################################################################################################################
# EDA - ind_actividad_cliente
##############################################################################################################################

viz <- ggplot(data=train,aes(x=ind_actividad_cliente)) + 
  geom_bar(alpha=0.75,fill="#FF6666") +
  ggtitle("ind_actividad_cliente Distribution") + 
  my_theme
print(viz)
summary(train$ind_actividad_cliente)

# Missing percentage
percent(round(sum(is.na(train$ind_actividad_cliente))/nrow(train), digits=4))
# 0.69%, is actually not bad

ind_actividad_cliente.median <- median(train$ind_actividad_cliente,na.rm=TRUE)
train$ind_actividad_cliente[is.na(train$ind_actividad_cliente)] <- ind_actividad_cliente.median

# DQ Check - missing value fill has worked, need to get 0%
percent(round(sum(is.na(train$ind_actividad_cliente))/nrow(train), digits=4))

##############################################################################################################################
# EDA - ind_actividad_cliente
##############################################################################################################################

##############################################################################################################################
# EDA - renta
##############################################################################################################################

viz <- ggplot(data=train,aes(x=renta)) + 
  geom_density(alpha=0.75,fill="#FF6666") +
  ggtitle("renta Distribution") + 
  my_theme
print(viz)
summary(train$renta)

train$nomprov[train$nomprov==""] <- "missing"

viz <- train %>%
  filter(!is.na(renta)) %>%
  group_by(nomprov) %>%
  summarise(med.income = median(renta)) %>%
  arrange(med.income) %>%
  mutate(city=factor(nomprov,levels=nomprov)) %>% # the factor() call prevents reordering the names
  ggplot(aes(x=city,y=med.income)) + 
  geom_point(color="#FF6666") + 
  xlab("City") +
  ylab("Median Income") +  
  my_theme
print(viz)
# Missing percentage
percent(round(sum(is.na(train$renta))/nrow(train), digits=4))
# 20.50%, is actually very bad

renta.median <- median(train$renta,na.rm=TRUE)
train$renta[is.na(train$renta)] <- renta.median

# DQ Check - missing value fill has worked, need to get 0%
percent(round(sum(is.na(train$renta))/nrow(train), digits=4))

##############################################################################################################################
# EDA - renta
##############################################################################################################################

##############################################################################################################################
# EDA - ind_nomina_ult1, ind_nom_pens_ult1
##############################################################################################################################

train$ind_nomina_ult1[is.na(train$ind_nomina_ult1)] <- 0
train$ind_nom_pens_ult1[is.na(train$ind_nom_pens_ult1)] <- 0
##############################################################################################################################
# EDA - ind_nomina_ult1, ind_nom_pens_ult1
##############################################################################################################################

sapply(train,function(x)any(is.na(x)))


viz <- ggplot(data=train,aes(x=indfall)) + 
        geom_bar(alpha=0.75,fill="#FF6666") +
        ggtitle("indfall Distribution") + 
        my_theme
print(viz)
train$indfall[train$indfall==""]           <- "N"

viz <- ggplot(data=train,aes(x=tiprel_1mes)) + 
        geom_bar(alpha=0.75,fill="#FF6666") +
        ggtitle("tiprel_1mes Distribution") + 
        my_theme
print(viz)
train$tiprel_1mes[train$tiprel_1mes==""]   <- "U"

viz <- ggplot(data=train,aes(x=indrel_1mes)) + 
        geom_bar(alpha=0.75,fill="#FF6666") +
        ggtitle("indrel_1mes Distribution") + 
        my_theme
print(viz)
train$indrel_1mes[train$indrel_1mes==""]         <- "1"
train$indrel_1mes[train$indrel_1mes=="P"]        <- "5" 

viz <- ggplot(data=train,aes(x=pais_residencia)) + 
        geom_bar(alpha=0.75,fill="#FF6666") +
        ggtitle("pais_residencia Distribution") + 
        my_theme
print(viz)
train$pais_residencia[train$pais_residencia==""] <- "U"

viz <- ggplot(data=train,aes(x=sexo)) + 
        geom_bar(alpha=0.75,fill="#FF6666") +
        ggtitle("sexo Distribution") + 
        my_theme
print(viz)
train$sexo[train$sexo==""] <- "U"

viz <- ggplot(data=train,aes(x=ult_fec_cli_1t)) + 
        geom_bar(alpha=0.75,fill="#FF6666") +
        ggtitle("ult_fec_cli_1t Distribution") + 
        my_theme
print(viz)
train$ult_fec_cli_1t[train$ult_fec_cli_1t==""] <- "U"

viz <- ggplot(data=train,aes(x=ind_empleado)) + 
        geom_bar(alpha=0.75,fill="#FF6666") +
        ggtitle("ind_empleado Distribution") + 
        my_theme
print(viz)
train$ind_empleado[train$ind_empleado==""] <- "U"

viz <- ggplot(data=train,aes(x=indext)) + 
        geom_bar(alpha=0.75,fill="#FF6666") +
        ggtitle("indext Distribution") + 
        my_theme
print(viz)
train$indext[train$indext==""] <- "U"

viz <- ggplot(data=train,aes(x=indresi)) + 
        geom_bar(alpha=0.75,fill="#FF6666") +
        ggtitle("indresi Distribution") + 
        my_theme
print(viz)
train$indresi[train$indresi==""] <- "U"

viz <- ggplot(data=train,aes(x=conyuemp)) + 
        geom_bar(alpha=0.75,fill="#FF6666") +
        ggtitle("conyuemp Distribution") + 
        my_theme
print(viz)
train$conyuemp[train$conyuemp==""] <- "U"

viz <- ggplot(data=train,aes(x=segmento)) + 
        geom_bar(alpha=0.75,fill="#FF6666") +
        ggtitle("segmento Distribution") + 
        my_theme
print(viz)
train$segmento[train$segmento==""] <- "U"

########################################################################################################################

#test[, ':='(fecha_dato = as.Date(fecha_dato), fecha_alta = as.Date(fecha_alta))]

# Check dates from test dataset, "2016-06-28"
sort(unique(test$fecha_dato))

# get Month number from dates to check on Time Series trends
test$month <- month(test$fecha_dato)

test$age[is.na(test$age)]              <- age.median
test$ind_nuevo[is.na(test$ind_nuevo)]  <- ind_nuevo.median

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


viz <- X_train %>%
        group_by(fecha_dato,flag_new) %>%
        summarise(rowCount = n()) %>%
        ggplot(aes(x=factor(fecha_dato),y=rowCount, fill = factor(flag_new))) + 
        geom_bar(stat="identity",position = "dodge") + 
        labs(title = paste0("Records by Month by Type")
             , x = "fecha_dato"
             , y = "#Records")+
        scale_fill_discrete(name = "New Products")+
        my_theme
print(viz)

viz <- X_train %>%
        group_by(fecha_dato,flag_new) %>%
        summarise(rowCount = n()) %>%
        ggplot(aes(x=factor(fecha_dato),y=rowCount, fill = factor(flag_new))) + 
        geom_bar(stat="identity",position = "dodge") + 
        labs(title = paste0("Records by Month by Type")
             , x = "fecha_dato"
             , y = "#Records")+
        scale_fill_discrete(name = "New Products")+
        my_theme
print(viz)


viz <-  X_train %>%
          filter(targetLabel == "ind_ahor_fin_ult1") %>%
          group_by(fecha_dato,flag_new) %>%
          summarise(rowCount = n()) %>%
          ggplot(aes(x=factor(fecha_dato),y=rowCount, fill = factor(flag_new))) + 
          geom_bar(stat="identity",position = "dodge") + 
          labs(title = paste0("ind_ahor_fin_ult1 Records by Month by Type")
               , x = "fecha_dato"
               , y = "#Records")+
          scale_fill_discrete(name = "New Products")+
          my_theme
print(viz)

viz <- X_train %>%
        filter(targetLabel == "ind_aval_fin_ult1") %>%
        group_by(fecha_dato,flag_new) %>%
        summarise(rowCount = n()) %>%
        ggplot(aes(x=factor(fecha_dato),y=rowCount, fill = factor(flag_new))) + 
        geom_bar(stat="identity",position = "dodge") + 
        labs(title = paste0("ind_aval_fin_ult1 Records by Month by Type")
             , x = "fecha_dato"
             , y = "#Records")+
        scale_fill_discrete(name = "New Products")+
        my_theme
print(viz)

viz <- X_train %>%
        filter(targetLabel == "ind_deco_fin_ult1") %>%
        group_by(fecha_dato,flag_new) %>%
        summarise(rowCount = n()) %>%
        ggplot(aes(x=factor(fecha_dato),y=rowCount, fill = factor(flag_new))) + 
        geom_bar(stat="identity",position = "dodge") + 
        labs(title = paste0("ind_deco_fin_ult1 Records by Month by Type")
             , x = "fecha_dato"
             , y = "#Records")+
        scale_fill_discrete(name = "New Products")+
        my_theme
print(viz)

viz <- X_train %>%
        filter(targetLabel == "ind_deme_fin_ult1") %>%
        group_by(fecha_dato,flag_new) %>%
        summarise(rowCount = n()) %>%
        ggplot(aes(x=factor(fecha_dato),y=rowCount, fill = factor(flag_new))) + 
        geom_bar(stat="identity",position = "dodge") + 
        labs(title = paste0("ind_deme_fin_ult1 Records by Month by Type")
             , x = "fecha_dato"
             , y = "#Records")+
        scale_fill_discrete(name = "New Products")+
        my_theme
print(viz)

viz <- X_train %>%
        filter(targetLabel == "ind_viv_fin_ult1") %>%
        group_by(fecha_dato,flag_new) %>%
        summarise(rowCount = n()) %>%
        ggplot(aes(x=factor(fecha_dato),y=rowCount, fill = factor(flag_new))) + 
        geom_bar(stat="identity",position = "dodge") + 
        labs(title = paste0("ind_viv_fin_ult1 Records by Month by Type")
             , x = "fecha_dato"
             , y = "#Records")+
        scale_fill_discrete(name = "New Products")+
        my_theme
print(viz)

generic.data.visualisations <- function(dataset, feature)
{
  
  viz <-    dataset %>%
                  filter(targetLabel == feature) %>%
                  group_by(fecha_dato,flag_new) %>%
                  summarise(rowCount = n()) %>%
                  ggplot(aes(x=factor(fecha_dato),y=rowCount, fill = factor(flag_new))) + 
                  geom_bar(stat="identity",position = "dodge") + 
                  labs(title = paste0(feature, " Records by Month by Type")
                       , x = "fecha_dato"
                       , y = "#Records")+
                  scale_fill_discrete(name = "New Products")+
                  my_theme
    
    print(viz)
}



for (target_col in target_cols)
{
  generic.data.visualisations(X_train,target_col)
  
}



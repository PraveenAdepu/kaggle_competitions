
rm(list=ls())
require(readr)
require(dplyr)

setwd("C:/Users/SriPrav/Documents/R/42Toxic")
root_directory = "C:/Users/SriPrav/Documents/R/42Toxic"


LSTM102_fold1 <- read_csv("./submissions/Prav.102_Prav_LSTM.fold1-test.csv")
LSTM102_fold2 <- read_csv("./submissions/Prav.102_Prav_LSTM.fold2-test.csv")
LSTM102_fold3 <- read_csv("./submissions/Prav.102_Prav_LSTM.fold3-test.csv")
LSTM102_fold4 <- read_csv("./submissions/Prav.102_Prav_LSTM.fold4-test.csv")
LSTM102_fold5 <- read_csv("./submissions/Prav.102_Prav_LSTM.fold5-test.csv")

LSTM102_Model <- rbind(LSTM102_fold1,LSTM102_fold2,LSTM102_fold3,LSTM102_fold4,LSTM102_fold5)

head(LSTM102_Model)

LSTM102_Model <- LSTM102_Model %>% 
                      group_by(id) %>%
                      summarise_each(funs(mean), toxic,severe_toxic,obscene,threat,insult,identity_hate)


                    
head(LSTM102_Model)

write.csv(LSTM102_Model,"./submissions/Prav.102_LSTM.folds-test.csv", row.names = FALSE, quote = FALSE)


LSTM103_fold1 <- read_csv("./submissions/Prav.103_Prav_LSTM.fold1-test.csv")
LSTM103_fold2 <- read_csv("./submissions/Prav.103_Prav_LSTM.fold2-test.csv")
LSTM103_fold3 <- read_csv("./submissions/Prav.103_Prav_LSTM.fold3-test.csv")
LSTM103_fold4 <- read_csv("./submissions/Prav.103_Prav_LSTM.fold4-test.csv")
LSTM103_fold5 <- read_csv("./submissions/Prav.103_Prav_LSTM.fold5-test.csv")

LSTM103_Model <- rbind(LSTM103_fold1,LSTM103_fold2,LSTM103_fold3,LSTM103_fold4,LSTM103_fold5)

head(LSTM103_Model)

LSTM103_Model <- LSTM103_Model %>% 
  group_by(id) %>%
  summarise_each(funs(mean), toxic,severe_toxic,obscene,threat,insult,identity_hate)



head(LSTM103_Model)

write.csv(LSTM103_Model,"./submissions/Prav.103_LSTM.folds-test.csv", row.names = FALSE, quote = FALSE)

###################################################################################################################
CNN100_fold1 <- read_csv("./submissions/Prav.100_Prav_CNN.fold1-test.csv")
CNN100_fold2 <- read_csv("./submissions/Prav.100_Prav_CNN.fold2-test.csv")
CNN100_fold3 <- read_csv("./submissions/Prav.100_Prav_CNN.fold3-test.csv")
CNN100_fold4 <- read_csv("./submissions/Prav.100_Prav_CNN.fold4-test.csv")
CNN100_fold5 <- read_csv("./submissions/Prav.100_Prav_CNN.fold5-test.csv")

CNN100_Model <- rbind(CNN100_fold1,CNN100_fold2,CNN100_fold3,CNN100_fold4,CNN100_fold5)

head(CNN100_Model)

CNN100_Model <- CNN100_Model %>% 
  group_by(id) %>%
  summarise_each(funs(mean), toxic,severe_toxic,obscene,threat,insult,identity_hate)

head(CNN100_Model)

write.csv(CNN100_Model,"./submissions/Prav.100_CNN.folds-test.csv", row.names = FALSE, quote = FALSE)
#####################################################################################################################

###################################################################################################################
GRU100_fold1 <- read_csv("./submissions/Prav.100_Prav_GRU.fold1-test.csv")
GRU100_fold2 <- read_csv("./submissions/Prav.100_Prav_GRU.fold2-test.csv")
GRU100_fold3 <- read_csv("./submissions/Prav.100_Prav_GRU.fold3-test.csv")
GRU100_fold4 <- read_csv("./submissions/Prav.100_Prav_GRU.fold4-test.csv")
GRU100_fold5 <- read_csv("./submissions/Prav.100_Prav_GRU.fold5-test.csv")

GRU100_Model <- rbind(GRU100_fold1,GRU100_fold2,GRU100_fold3,GRU100_fold4,GRU100_fold5)

head(GRU100_Model)

GRU100_Model <- GRU100_Model %>% 
  group_by(id) %>%
  summarise_each(funs(mean), toxic,severe_toxic,obscene,threat,insult,identity_hate)

head(GRU100_Model)

write.csv(GRU100_Model,"./submissions/Prav.100_GRU.folds-test.csv", row.names = FALSE, quote = FALSE)
#####################################################################################################################

train <- read_csv("./input/train.csv")
train$comment_text <- NULL

LSTM103_fold1 <- read_csv("./submissions/Prav.100_Prav_CNN.fold1.csv")
LSTM103_fold2 <- read_csv("./submissions/Prav.100_Prav_CNN.fold2.csv")
LSTM103_fold3 <- read_csv("./submissions/Prav.100_Prav_CNN.fold3.csv")
LSTM103_fold4 <- read_csv("./submissions/Prav.100_Prav_CNN.fold4.csv")
LSTM103_fold5 <- read_csv("./submissions/Prav.100_Prav_CNN.fold5.csv")

LSTM103_Model <- rbind(LSTM103_fold1,LSTM103_fold2,LSTM103_fold3,LSTM103_fold4,LSTM103_fold5)


Prav_5fold_CVindices <- read_csv("./input/Prav_5folds_CVindices.csv")

train <- left_join(train, Prav_5fold_CVindices,  by= "id")

train <- left_join(train, LSTM103_Model,  by= "id")

head(train)

score <- function(a,b,metric)
  
{    
  switch(metric,           
         accuracy = sum(abs(a-b)<=0.5)/length(a),           
         auc = auc(a,b),           
         logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),           
         mae = sum(abs(a-b))/length(a),           
         precision = length(a[a==b])/length(a),           
         rmse = sqrt(sum((a-b)^2)/length(a)),           
         rmspe = sqrt(sum(((a-b)/a)^2)/length(a)))    
}

metric = "logloss"


toxic_logloss         <- score(train$toxic.x, ifelse(train$toxic.y==1,0.9999,train$toxic.y), metric)
severe_toxic_logloss  <- score(train$severe_toxic.x, train$severe_toxic.y, metric)
obscene_logloss       <- score(train$obscene.x, train$obscene.y, metric)
threat_logloss        <- score(train$threat.x, train$threat.y, metric)
insult_logloss        <- score(train$insult.x, train$insult.y, metric)
identity_hate_logloss <- score(train$identity_hate.x, train$identity_hate.y, metric)

model_logloss <- (toxic_logloss+severe_toxic_logloss+obscene_logloss+threat_logloss+insult_logloss+identity_hate_logloss)/6
cat(toxic_logloss,severe_toxic_logloss,obscene_logloss,threat_logloss,insult_logloss,identity_hate_logloss)
model_logloss

# > cat(toxic_logloss,severe_toxic_logloss,obscene_logloss,threat_logloss,insult_logloss,identity_hate_logloss)
# 0.1055055 0.02317443 0.05974623 0.0130876 0.07108514 0.02459346
# > model_logloss
# [1] 0.04953206

#####################################################################################################################

train <- read_csv("./input/train.csv")
train$comment_text <- NULL

LSTM103_fold1 <- read_csv("./submissions/Prav.100_Prav_LSTM_CNN.fold1.csv")
LSTM103_fold2 <- read_csv("./submissions/Prav.100_Prav_LSTM_CNN.fold2.csv")
LSTM103_fold3 <- read_csv("./submissions/Prav.100_Prav_LSTM_CNN.fold3.csv")
LSTM103_fold4 <- read_csv("./submissions/Prav.100_Prav_LSTM_CNN.fold4.csv")
LSTM103_fold5 <- read_csv("./submissions/Prav.100_Prav_LSTM_CNN.fold5.csv")

LSTM103_Model <- rbind(LSTM103_fold1,LSTM103_fold2,LSTM103_fold3,LSTM103_fold4,LSTM103_fold5)


Prav_5fold_CVindices <- read_csv("./input/Prav_5folds_CVindices.csv")

train <- left_join(train, Prav_5fold_CVindices,  by= "id")

train <- left_join(train, LSTM103_Model,  by= "id")

head(train)

score <- function(a,b,metric)
  
{    
  switch(metric,           
         accuracy = sum(abs(a-b)<=0.5)/length(a),           
         auc = auc(a,b),           
         logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),           
         mae = sum(abs(a-b))/length(a),           
         precision = length(a[a==b])/length(a),           
         rmse = sqrt(sum((a-b)^2)/length(a)),           
         rmspe = sqrt(sum(((a-b)/a)^2)/length(a)))    
}

metric = "logloss"


toxic_logloss         <- score(train$toxic.x, ifelse(train$toxic.y==1,0.9999,train$toxic.y), metric)
severe_toxic_logloss  <- score(train$severe_toxic.x, train$severe_toxic.y, metric)
obscene_logloss       <- score(train$obscene.x, train$obscene.y, metric)
threat_logloss        <- score(train$threat.x, train$threat.y, metric)
insult_logloss        <- score(train$insult.x, train$insult.y, metric)
identity_hate_logloss <- score(train$identity_hate.x, train$identity_hate.y, metric)

model_logloss <- (toxic_logloss+severe_toxic_logloss+obscene_logloss+threat_logloss+insult_logloss+identity_hate_logloss)/6
cat(toxic_logloss,severe_toxic_logloss,obscene_logloss,threat_logloss,insult_logloss,identity_hate_logloss)
model_logloss

# LSTM103
# 0.10227 0.02323378 0.05494101 0.009785115 0.06765308 0.02307663
# model_logloss
# 0.04682661

#####################################################################################################################

train <- read_csv("./input/train.csv")
train$comment_text <- NULL

LSTM103_fold1 <- read_csv("./submissions/Prav.105_Prav_LSTM.fold1.csv")
LSTM103_fold2 <- read_csv("./submissions/Prav.105_Prav_LSTM.fold2.csv")
LSTM103_fold3 <- read_csv("./submissions/Prav.105_Prav_LSTM.fold3.csv")
LSTM103_fold4 <- read_csv("./submissions/Prav.105_Prav_LSTM.fold4.csv")
LSTM103_fold5 <- read_csv("./submissions/Prav.105_Prav_LSTM.fold5.csv")

LSTM103_Model <- rbind(LSTM103_fold1,LSTM103_fold2,LSTM103_fold3,LSTM103_fold4,LSTM103_fold5)


Prav_5fold_CVindices <- read_csv("./input/Prav_5folds_CVindices.csv")

train <- left_join(train, Prav_5fold_CVindices,  by= "id")

train <- left_join(train, LSTM103_Model,  by= "id")

head(train)

score <- function(a,b,metric)
  
{    
  switch(metric,           
         accuracy = sum(abs(a-b)<=0.5)/length(a),           
         auc = auc(a,b),           
         logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),           
         mae = sum(abs(a-b))/length(a),           
         precision = length(a[a==b])/length(a),           
         rmse = sqrt(sum((a-b)^2)/length(a)),           
         rmspe = sqrt(sum(((a-b)/a)^2)/length(a)))    
}

metric = "logloss"


toxic_logloss         <- score(train$toxic.x, ifelse(train$toxic.y==1,0.9999,train$toxic.y), metric)
severe_toxic_logloss  <- score(train$severe_toxic.x, ifelse(train$severe_toxic.y==1,0.9999,train$severe_toxic.y), metric)
obscene_logloss       <- score(train$obscene.x, ifelse(train$obscene.y==1,0.9999,train$obscene.y), metric)
threat_logloss        <- score(train$threat.x, ifelse(train$threat.y==1,0.9999,train$threat.y), metric)
insult_logloss        <- score(train$insult.x, ifelse(train$insult.y==1,0.9999,train$insult.y), metric)
identity_hate_logloss <- score(train$identity_hate.x, ifelse(train$identity_hate.y==1,0.9999,train$identity_hate.y), metric)

model_logloss <- (toxic_logloss+severe_toxic_logloss+obscene_logloss+threat_logloss+insult_logloss+identity_hate_logloss)/6
cat(toxic_logloss,severe_toxic_logloss,obscene_logloss,threat_logloss,insult_logloss,identity_hate_logloss)
model_logloss

# LSTM103
# 0.10227 0.02323378 0.05494101 0.009785115 0.06765308 0.02307663
# model_logloss
# 0.04682661

LSTM104_fold1 <- read_csv("./submissions/Prav.105_Prav_LSTM.fold1-test.csv")
LSTM104_fold2 <- read_csv("./submissions/Prav.105_Prav_LSTM.fold2-test.csv")
LSTM104_fold3 <- read_csv("./submissions/Prav.105_Prav_LSTM.fold3-test.csv")
LSTM104_fold4 <- read_csv("./submissions/Prav.105_Prav_LSTM.fold4-test.csv")
LSTM104_fold5 <- read_csv("./submissions/Prav.105_Prav_LSTM.fold5-test.csv")

LSTM104_Model <- rbind(LSTM104_fold1,LSTM104_fold2,LSTM104_fold3,LSTM104_fold4,LSTM104_fold5)

LSTM104_Model$toxic <- ifelse(LSTM104_Model$toxic==1,0.9999,LSTM104_Model$toxic)
LSTM104_Model$severe_toxic <- ifelse(LSTM104_Model$severe_toxic==1,0.9999,LSTM104_Model$severe_toxic)
LSTM104_Model$obscene <- ifelse(LSTM104_Model$obscene==1,0.9999,LSTM104_Model$obscene)
LSTM104_Model$threat <- ifelse(LSTM104_Model$threat==1,0.9999,LSTM104_Model$threat)
LSTM104_Model$insult <- ifelse(LSTM104_Model$insult==1,0.9999,LSTM104_Model$insult)
LSTM104_Model$identity_hate <- ifelse(LSTM104_Model$identity_hate==1,0.9999,LSTM104_Model$identity_hate)


head(LSTM104_Model)

LSTM104_Model %>% filter(obscene == 1)

LSTM104_Model <- LSTM104_Model %>% 
  group_by(id) %>%
  summarise_each(funs(mean), toxic,severe_toxic,obscene,threat,insult,identity_hate)


head(LSTM104_Model)


write.csv(LSTM104_Model,"./submissions/Prav.105_LSTM.folds-test.csv", row.names = FALSE, quote = FALSE)


#######################################################################################################################

#####################################################################################################################

train <- read_csv("./input/train.csv")
train$comment_text <- NULL

LSTM103_fold1 <- read_csv("./submissions/Prav.107_Prav_LSTM.fold1.csv")
LSTM103_fold2 <- read_csv("./submissions/Prav.107_Prav_LSTM.fold2.csv")
LSTM103_fold3 <- read_csv("./submissions/Prav.107_Prav_LSTM.fold3.csv")
LSTM103_fold4 <- read_csv("./submissions/Prav.107_Prav_LSTM.fold4.csv")
LSTM103_fold5 <- read_csv("./submissions/Prav.107_Prav_LSTM.fold5.csv")

LSTM103_Model <- rbind(LSTM103_fold1,LSTM103_fold2,LSTM103_fold3,LSTM103_fold4,LSTM103_fold5)


Prav_5fold_CVindices <- read_csv("./input/Prav_5folds_CVindices.csv")

train <- left_join(train, Prav_5fold_CVindices,  by= "id")

train <- left_join(train, LSTM103_Model,  by= "id")

head(train)

score <- function(a,b,metric)
  
{    
  switch(metric,           
         accuracy = sum(abs(a-b)<=0.5)/length(a),           
         auc = auc(a,b),           
         logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),           
         mae = sum(abs(a-b))/length(a),           
         precision = length(a[a==b])/length(a),           
         rmse = sqrt(sum((a-b)^2)/length(a)),           
         rmspe = sqrt(sum(((a-b)/a)^2)/length(a)))    
}

metric = "logloss"



train <- train %>%
           rowwise() %>% 
             mutate(  toxic.y = min(max(toxic.y, 1E-15), 1-1E-15)
                    , severe_toxic.y = min(max(severe_toxic.y, 1E-15), 1-1E-15)
                    , obscene.y = min(max(obscene.y, 1E-15), 1-1E-15)
                    , threat.y = min(max(threat.y, 1E-15), 1-1E-15)
                    , insult.y = min(max(insult.y, 1E-15), 1-1E-15)
                    , identity_hate.y = min(max(identity_hate.y, 1E-15), 1-1E-15)
                    
                    )


toxic_logloss         <- score(train$toxic.x, train$toxic.y, metric)
severe_toxic_logloss  <- score(train$severe_toxic.x, train$severe_toxic.y, metric)
obscene_logloss       <- score(train$obscene.x, train$obscene.y, metric)
threat_logloss        <- score(train$threat.x, train$threat.y, metric)
insult_logloss        <- score(train$insult.x, train$insult.y, metric)
identity_hate_logloss <- score(train$identity_hate.x, train$identity_hate.y, metric)

model_logloss <- (toxic_logloss+severe_toxic_logloss+obscene_logloss+threat_logloss+insult_logloss+identity_hate_logloss)/6
cat(toxic_logloss,severe_toxic_logloss,obscene_logloss,threat_logloss,insult_logloss,identity_hate_logloss)
model_logloss

# LSTM103
# 0.10227 0.02323378 0.05494101 0.009785115 0.06765308 0.02307663
# model_logloss
# 0.04682661

LSTM104_fold1 <- read_csv("./submissions/Prav.107_Prav_LSTM.fold1-test.csv")
LSTM104_fold2 <- read_csv("./submissions/Prav.107_Prav_LSTM.fold2-test.csv")
LSTM104_fold3 <- read_csv("./submissions/Prav.107_Prav_LSTM.fold3-test.csv")
LSTM104_fold4 <- read_csv("./submissions/Prav.107_Prav_LSTM.fold4-test.csv")
LSTM104_fold5 <- read_csv("./submissions/Prav.107_Prav_LSTM.fold5-test.csv")

LSTM104_Model <- rbind(LSTM104_fold1,LSTM104_fold2,LSTM104_fold3,LSTM104_fold4,LSTM104_fold5)

LSTM104_Model <- LSTM104_Model %>%
  rowwise() %>% 
  mutate(  toxic = min(max(toxic, 1E-15), 1-1E-15)
           , severe_toxic = min(max(severe_toxic, 1E-15), 1-1E-15)
           , obscene = min(max(obscene, 1E-15), 1-1E-15)
           , threat = min(max(threat, 1E-15), 1-1E-15)
           , insult = min(max(insult, 1E-15), 1-1E-15)
           , identity_hate = min(max(identity_hate, 1E-15), 1-1E-15)
           
  )


head(LSTM104_Model)


LSTM104_Model <- LSTM104_Model %>% 
  group_by(id) %>%
  summarise_each(funs(mean), toxic,severe_toxic,obscene,threat,insult,identity_hate)


head(LSTM104_Model)


write.csv(LSTM104_Model,"./submissions/Prav.107_LSTM.folds-test.csv", row.names = FALSE, quote = FALSE)


#######################################################################################################################


# Ensembling ##########################################################################################################
#######################################################################################################################
train <- read_csv("./input/train.csv")
train$comment_text <- NULL

LSTM103_fold1 <- read_csv("./submissions/Prav.103_Prav_LSTM.fold1.csv")
LSTM103_fold2 <- read_csv("./submissions/Prav.103_Prav_LSTM.fold2.csv")
LSTM103_fold3 <- read_csv("./submissions/Prav.103_Prav_LSTM.fold3.csv")
LSTM103_fold4 <- read_csv("./submissions/Prav.103_Prav_LSTM.fold4.csv")
LSTM103_fold5 <- read_csv("./submissions/Prav.103_Prav_LSTM.fold5.csv")

LSTM103_Model <- rbind(LSTM103_fold1,LSTM103_fold2,LSTM103_fold3,LSTM103_fold4,LSTM103_fold5)

CNN100_fold1 <- read_csv("./submissions/Prav.100_Prav_CNN.fold1.csv")
CNN100_fold2 <- read_csv("./submissions/Prav.100_Prav_CNN.fold2.csv")
CNN100_fold3 <- read_csv("./submissions/Prav.100_Prav_CNN.fold3.csv")
CNN100_fold4 <- read_csv("./submissions/Prav.100_Prav_CNN.fold4.csv")
CNN100_fold5 <- read_csv("./submissions/Prav.100_Prav_CNN.fold5.csv")

CNN100_Model <- rbind(CNN100_fold1,CNN100_fold2,CNN100_fold3,CNN100_fold4,CNN100_fold5)

GRU100_fold1 <- read_csv("./submissions/Prav.100_Prav_GRU.fold1.csv")
GRU100_fold2 <- read_csv("./submissions/Prav.100_Prav_GRU.fold2.csv")
GRU100_fold3 <- read_csv("./submissions/Prav.100_Prav_GRU.fold3.csv")
GRU100_fold4 <- read_csv("./submissions/Prav.100_Prav_GRU.fold4.csv")
GRU100_fold5 <- read_csv("./submissions/Prav.100_Prav_GRU.fold5.csv")

GRU100_Model <- rbind(GRU100_fold1,GRU100_fold2,GRU100_fold3,GRU100_fold4,GRU100_fold5)

LSTMCNN100_fold1 <- read_csv("./submissions/Prav.100_Prav_LSTM_CNN.fold1.csv")
LSTMCNN100_fold2 <- read_csv("./submissions/Prav.100_Prav_LSTM_CNN.fold2.csv")
LSTMCNN100_fold3 <- read_csv("./submissions/Prav.100_Prav_LSTM_CNN.fold3.csv")
LSTMCNN100_fold4 <- read_csv("./submissions/Prav.100_Prav_LSTM_CNN.fold4.csv")
LSTMCNN100_fold5 <- read_csv("./submissions/Prav.100_Prav_LSTM_CNN.fold5.csv")

LSTMCNN100_Model <- rbind(LSTMCNN100_fold1,LSTMCNN100_fold2,LSTMCNN100_fold3,LSTMCNN100_fold4,LSTMCNN100_fold5)

Models <- rbind(LSTM103_Model, CNN100_Model,GRU100_Model,LSTMCNN100_Model)

Models <- Models %>% 
  group_by(id) %>%
  summarise_each(funs(mean), toxic,severe_toxic,obscene,threat,insult,identity_hate)

Prav_5fold_CVindices <- read_csv("./input/Prav_5folds_CVindices.csv")

train <- left_join(train, Prav_5fold_CVindices,  by= "id")

train <- left_join(train, Models,  by= "id")

head(train)

score <- function(a,b,metric)
  
{    
  switch(metric,           
         accuracy = sum(abs(a-b)<=0.5)/length(a),           
         auc = auc(a,b),           
         logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),           
         mae = sum(abs(a-b))/length(a),           
         precision = length(a[a==b])/length(a),           
         rmse = sqrt(sum((a-b)^2)/length(a)),           
         rmspe = sqrt(sum(((a-b)/a)^2)/length(a)))    
}

metric = "logloss"


toxic_logloss         <- score(train$toxic.x, ifelse(train$toxic.y==1,0.9999,train$toxic.y), metric)
severe_toxic_logloss  <- score(train$severe_toxic.x, train$severe_toxic.y, metric)
obscene_logloss       <- score(train$obscene.x, train$obscene.y, metric)
threat_logloss        <- score(train$threat.x, train$threat.y, metric)
insult_logloss        <- score(train$insult.x, train$insult.y, metric)
identity_hate_logloss <- score(train$identity_hate.x, train$identity_hate.y, metric)

model_logloss <- (toxic_logloss+severe_toxic_logloss+obscene_logloss+threat_logloss+insult_logloss+identity_hate_logloss)/6
cat(toxic_logloss,severe_toxic_logloss,obscene_logloss,threat_logloss,insult_logloss,identity_hate_logloss)
model_logloss


# > cat(toxic_logloss,severe_toxic_logloss,obscene_logloss,threat_logloss,insult_logloss,identity_hate_logloss)
# 0.09699534 0.02235245 0.05254489 0.009906022 0.06490545 0.02148846
# > model_logloss
# [1] 0.04469877


#######################################################################################################################

LSTM103_Model <- read_csv("./submissions/Prav.103_LSTM.folds-test.csv")
CNN100_Model  <- read_csv("./submissions/Prav.100_CNN.folds-test.csv")
GRU100_Model  <- read_csv("./submissions/Prav.100_GRU.folds-test.csv")

Models <- rbind(LSTM103_Model, CNN100_Model,GRU100_Model)

Models <- Models %>% 
  group_by(id) %>%
  summarise_each(funs(mean), toxic,severe_toxic,obscene,threat,insult,identity_hate)
head(Models)
write.csv(Models,"./submissions/Prav.103LSTM100CNN100GRU.folds-test.csv", row.names = FALSE, quote = FALSE)

# cor.features <- setdiff(names(models),"id")
# cor(models[,cor.features])
######################################################################################################################

Model001  <- read_csv("./submissions/Prav.104_LSTM.folds-test.csv")
Model002  <- read_csv("./submissions/Prav.105_LSTM.folds-test.csv")

Models <- left_join(Model001, Model002, by="id")

cor.features <- setdiff(names(Models),"id")
cor(Models[,cor.features])

######################################################################################################################

train <- read_csv("./input/train.csv")
train$comment_text <- NULL

LSTM103_fold1 <- read_csv("./submissions/Prav.104_Prav_LSTM.fold1.csv")
LSTM103_fold2 <- read_csv("./submissions/Prav.104_Prav_LSTM.fold2.csv")
LSTM103_fold3 <- read_csv("./submissions/Prav.104_Prav_LSTM.fold3.csv")
LSTM103_fold4 <- read_csv("./submissions/Prav.104_Prav_LSTM.fold4.csv")
LSTM103_fold5 <- read_csv("./submissions/Prav.104_Prav_LSTM.fold5.csv")

LSTM103_Model <- rbind(LSTM103_fold1,LSTM103_fold2,LSTM103_fold3,LSTM103_fold4,LSTM103_fold5)

CNN100_fold1 <- read_csv("./submissions/Prav.105_Prav_LSTM.fold1.csv")
CNN100_fold2 <- read_csv("./submissions/Prav.105_Prav_LSTM.fold2.csv")
CNN100_fold3 <- read_csv("./submissions/Prav.105_Prav_LSTM.fold3.csv")
CNN100_fold4 <- read_csv("./submissions/Prav.105_Prav_LSTM.fold4.csv")
CNN100_fold5 <- read_csv("./submissions/Prav.105_Prav_LSTM.fold5.csv")

CNN100_Model <- rbind(CNN100_fold1,CNN100_fold2,CNN100_fold3,CNN100_fold4,CNN100_fold5)

GRU100_fold1 <- read_csv("./submissions/Prav.106_Prav_CNN.fold1.csv")
GRU100_fold2 <- read_csv("./submissions/Prav.106_Prav_CNN.fold2.csv")
GRU100_fold3 <- read_csv("./submissions/Prav.106_Prav_CNN.fold3.csv")
GRU100_fold4 <- read_csv("./submissions/Prav.106_Prav_CNN.fold4.csv")
GRU100_fold5 <- read_csv("./submissions/Prav.106_Prav_CNN.fold5.csv")

GRU100_Model <- rbind(GRU100_fold1,GRU100_fold2,GRU100_fold3,GRU100_fold4,GRU100_fold5)

LSTMCNN100_fold1 <- read_csv("./submissions/Prav.107_Prav_LSTM.fold1.csv")
LSTMCNN100_fold2 <- read_csv("./submissions/Prav.107_Prav_LSTM.fold2.csv")
LSTMCNN100_fold3 <- read_csv("./submissions/Prav.107_Prav_LSTM.fold3.csv")
LSTMCNN100_fold4 <- read_csv("./submissions/Prav.107_Prav_LSTM.fold4.csv")
LSTMCNN100_fold5 <- read_csv("./submissions/Prav.107_Prav_LSTM.fold5.csv")

LSTMCNN100_Model <- rbind(LSTMCNN100_fold1,LSTMCNN100_fold2,LSTMCNN100_fold3,LSTMCNN100_fold4,LSTMCNN100_fold5)

Models <- rbind(LSTM103_Model, CNN100_Model,GRU100_Model,LSTMCNN100_Model)

Models <- Models %>% 
  group_by(id) %>%
  summarise_each(funs(mean), toxic,severe_toxic,obscene,threat,insult,identity_hate)

Prav_5fold_CVindices <- read_csv("./input/Prav_5folds_CVindices.csv")

train <- left_join(train, Prav_5fold_CVindices,  by= "id")

train <- left_join(train, Models,  by= "id")

head(train)

score <- function(a,b,metric)
  
{    
  switch(metric,           
         accuracy = sum(abs(a-b)<=0.5)/length(a),           
         auc = auc(a,b),           
         logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),           
         mae = sum(abs(a-b))/length(a),           
         precision = length(a[a==b])/length(a),           
         rmse = sqrt(sum((a-b)^2)/length(a)),           
         rmspe = sqrt(sum(((a-b)/a)^2)/length(a)))    
}

metric = "logloss"


toxic_logloss         <- score(train$toxic.x, ifelse(train$toxic.y==1,0.9999,train$toxic.y), metric)
severe_toxic_logloss  <- score(train$severe_toxic.x, train$severe_toxic.y, metric)
obscene_logloss       <- score(train$obscene.x, train$obscene.y, metric)
threat_logloss        <- score(train$threat.x, train$threat.y, metric)
insult_logloss        <- score(train$insult.x, train$insult.y, metric)
identity_hate_logloss <- score(train$identity_hate.x, train$identity_hate.y, metric)

model_logloss <- (toxic_logloss+severe_toxic_logloss+obscene_logloss+threat_logloss+insult_logloss+identity_hate_logloss)/6
cat(toxic_logloss,severe_toxic_logloss,obscene_logloss,threat_logloss,insult_logloss,identity_hate_logloss)
model_logloss


# > cat(toxic_logloss,severe_toxic_logloss,obscene_logloss,threat_logloss,insult_logloss,identity_hate_logloss)
# 0.09699534 0.02235245 0.05254489 0.009906022 0.06490545 0.02148846
# > model_logloss
# [1] 0.04469877


#######################################################################################################################


Model001  <- read_csv("./submissions/Prav.104_LSTM.folds-test.csv")
Model002  <- read_csv("./submissions/Prav.105_LSTM.folds-test.csv")
Model003  <- read_csv("./submissions/Prav.106_CNN.folds-test.csv")
Model004  <- read_csv("./submissions/Prav.107_LSTM.folds-test.csv")

Models <- rbind(Model001, Model002,Model003, Model004)

Models <- Models %>% 
  group_by(id) %>%
  summarise_each(funs(mean), toxic,severe_toxic,obscene,threat,insult,identity_hate)
head(Models)
write.csv(Models,"./submissions/Prav.10456_LSTM_107CNN.folds-test.csv", row.names = FALSE, quote = FALSE)

# cor.features <- setdiff(names(models),"id")
# cor(models[,cor.features])
######################################################################################################################

Model001  <- read_csv("./submissions/Prav.10456_LSTM_107CNN.folds-test.csv")
head(Model001)
Model002  <- read_csv("./submissions/Prav_reference_submission.csv")
head(Model002)
ensemble <- left_join(Model001, Model002, by="id")

cor.features <- setdiff(names(ensemble),"id")
cor(ensemble[,cor.features])

Models <- rbind(Model001, Model002)

Models <- Models %>% 
  group_by(id) %>%
  summarise_each(funs(mean), toxic,severe_toxic,obscene,threat,insult,identity_hate)
head(Models)
Models$id <- as.integer(Models$id)
write.csv(Models,"./submissions/Prav.10456_LSTM_107CNN_Ref.folds-test.csv", row.names = FALSE, quote = FALSE)

length(unique(Models$id))



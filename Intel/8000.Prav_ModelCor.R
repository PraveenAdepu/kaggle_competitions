setwd("C:/Users/SriPrav/Documents/R/22Intel")
root_directory = "C:/Users/SriPrav/Documents/R/22Intel"

# paste(root_directory, "/input/events.csv", sep='')

# rm(list=ls())

require(data.table)
require(Matrix)
require(sqldf)
require(plyr)
require(dplyr)
require(ROCR)
require(Metrics)
require(pROC)
require(caret)
require(readr)
require(MLmetrics)
########################################################################################################
model01 <- read_csv("./submissions/Prav_dense161_aug_ensemble.csv") 
model02 <- read_csv("./input/solution_stg1_release/solution_stg1_release.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")
lables <- c("Type_1.y", "Type_2.y", "Type_3.y")
probs <- c("Type_1.x", "Type_2.x", "Type_3.x")
head(all_ensemble)
MultiLogLoss(y_true = data.matrix(all_ensemble[, lables]), y_pred = data.matrix(all_ensemble[, probs]))

#######################################################################################################

model01 <- read_csv("./submissions/Prav.resnet152.CNN01.csv") 
model02 <- read_csv("./submissions/Prav.densenet121.CNN01.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$Type_1    <- (all_ensemble$Type_1.x+all_ensemble$Type_1.y)/2
all_ensemble$Type_2    <- (all_ensemble$Type_2.x+all_ensemble$Type_2.y)/2
all_ensemble$Type_3    <- (all_ensemble$Type_3.x+all_ensemble$Type_3.y)/2
head(all_ensemble)

cols <- c("image_name","Type_1","Type_2","Type_3")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_res152_dense121_ensemble.csv', row.names=FALSE, quote = FALSE)

################################################################################################

model01 <- read_csv("./submissions/Prav.densenet161.CNN01.csv") 
model02 <- read_csv("./submissions/Prav_res152_dense121_ensemble.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$Type_1    <- (all_ensemble$Type_1.x+all_ensemble$Type_1.y)/2
all_ensemble$Type_2    <- (all_ensemble$Type_2.x+all_ensemble$Type_2.y)/2
all_ensemble$Type_3    <- (all_ensemble$Type_3.x+all_ensemble$Type_3.y)/2
head(all_ensemble)

cols <- c("image_name","Type_1","Type_2","Type_3")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_res152dense121_dense161ensemble.csv', row.names=FALSE, quote = FALSE)

################################################################################################
model01 <- read_csv("./submissions/Prav.densenet161.CNN01.csv") 
model02 <- read_csv("./submissions/Prav.densenet169.CNN01.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$Type_1    <- (all_ensemble$Type_1.x+all_ensemble$Type_1.y)/2
all_ensemble$Type_2    <- (all_ensemble$Type_2.x+all_ensemble$Type_2.y)/2
all_ensemble$Type_3    <- (all_ensemble$Type_3.x+all_ensemble$Type_3.y)/2
head(all_ensemble)

cols <- c("image_name","Type_1","Type_2","Type_3")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_dense161_169_ensemble.csv', row.names=FALSE, quote = FALSE)
################################################################################################
model01 <- read_csv("./submissions/Prav_dense161_169_ensemble.csv") 
model02 <- read_csv("./submissions/Prav_res152_dense121_ensemble.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$Type_1    <- (all_ensemble$Type_1.x+all_ensemble$Type_1.y)/2
all_ensemble$Type_2    <- (all_ensemble$Type_2.x+all_ensemble$Type_2.y)/2
all_ensemble$Type_3    <- (all_ensemble$Type_3.x+all_ensemble$Type_3.y)/2
head(all_ensemble)

cols <- c("image_name","Type_1","Type_2","Type_3")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_dense161169_dense121res156_ensemble.csv', row.names=FALSE, quote = FALSE)
#####################################################################################################################

model01 <- read_csv("./submissions/Prav.densenet161.CNN01.csv") 
model02 <- read_csv("./submissions/Prav.densenet161.CNN02_Aug.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$Type_1    <- (all_ensemble$Type_1.x+all_ensemble$Type_1.y)/2
all_ensemble$Type_2    <- (all_ensemble$Type_2.x+all_ensemble$Type_2.y)/2
all_ensemble$Type_3    <- (all_ensemble$Type_3.x+all_ensemble$Type_3.y)/2
head(all_ensemble)

cols <- c("image_name","Type_1","Type_2","Type_3")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav_dense161_aug_ensemble.csv', row.names=FALSE, quote = FALSE)

################################################################################################


###############################################################################################
# stage 2 ensemble 
###############################################################################################

#################################################################################################
model01 <- read_csv("./submissions/Prav.resnet152.stg2.CNN01.csv") 
model02 <- read_csv("./submissions/Prav.densenet121.stg2.CNN01.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$Type_1    <- (all_ensemble$Type_1.x+all_ensemble$Type_1.y)/2
all_ensemble$Type_2    <- (all_ensemble$Type_2.x+all_ensemble$Type_2.y)/2
all_ensemble$Type_3    <- (all_ensemble$Type_3.x+all_ensemble$Type_3.y)/2
head(all_ensemble)

cols <- c("image_name","Type_1","Type_2","Type_3")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.stg2_res152_dense121_ensemble.csv', row.names=FALSE, quote = FALSE)

################################################################################################

################################################################################################
model01 <- read_csv("./submissions/Prav.densenet161.stg2.CNN01.csv") 
model02 <- read_csv("./submissions/Prav.densenet169.stg2.CNN01.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$Type_1    <- (all_ensemble$Type_1.x+all_ensemble$Type_1.y)/2
all_ensemble$Type_2    <- (all_ensemble$Type_2.x+all_ensemble$Type_2.y)/2
all_ensemble$Type_3    <- (all_ensemble$Type_3.x+all_ensemble$Type_3.y)/2
head(all_ensemble)

cols <- c("image_name","Type_1","Type_2","Type_3")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.stg2_dense161_169_ensemble.csv', row.names=FALSE, quote = FALSE)
################################################################################################

################################################################################################
model01 <- read_csv("./submissions/Prav.stg2_dense161_169_ensemble.csv") 
model02 <- read_csv("./submissions/Prav.stg2_res152_dense121_ensemble.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$Type_1    <- (all_ensemble$Type_1.x+all_ensemble$Type_1.y)/2
all_ensemble$Type_2    <- (all_ensemble$Type_2.x+all_ensemble$Type_2.y)/2
all_ensemble$Type_3    <- (all_ensemble$Type_3.x+all_ensemble$Type_3.y)/2
head(all_ensemble)

cols <- c("image_name","Type_1","Type_2","Type_3")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.stg2_dense161169_dense121res156_ensemble.csv', row.names=FALSE, quote = FALSE)
#####################################################################################################################
# combine stage1 and stage2 submit files

model01 <- read_csv("./submissions/Prav_dense161169_dense121res156_ensemble.csv") 
model02 <- read_csv("./submissions/Prav.stg2_dense161169_dense121res156_ensemble.csv")

head(model01)
head(model02)

finalsub <- rbind(model01, model02)
write.csv(finalsub, './submissions/Prav.stg12_dense161169_dense121res156_ensemble.csv', row.names=FALSE, quote = FALSE)

#####################################################################################################################

model01 <- read_csv("./submissions/Prav.densenet161.stg2.CNN01.csv") 
model02 <- read_csv("./submissions/Prav.densenet161.stg2.CNN02_Aug.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$Type_1    <- (all_ensemble$Type_1.x+all_ensemble$Type_1.y)/2
all_ensemble$Type_2    <- (all_ensemble$Type_2.x+all_ensemble$Type_2.y)/2
all_ensemble$Type_3    <- (all_ensemble$Type_3.x+all_ensemble$Type_3.y)/2
head(all_ensemble)

cols <- c("image_name","Type_1","Type_2","Type_3")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.stg2_dense161_aug_ensemble.csv', row.names=FALSE, quote = FALSE)

################################################################################################

#####################################################################################################################
# combine stage1 and stage2 submit files

model01 <- read_csv("./submissions/Prav_dense161_aug_ensemble.csv") 
model02 <- read_csv("./submissions/Prav.stg2_dense161_aug_ensemble.csv")

head(model01)
head(model02)

finalsub <- rbind(model01, model02)
write.csv(finalsub, './submissions/Prav.stg12_dense161_aug_ensemble.csv', row.names=FALSE, quote = FALSE)

#####################################################################################################################

model01 <- read_csv("./input/rectangles_train.csv") 
model02 <- read_csv("./input/rectangles_test.csv")
model03 <- read_csv("./input/solution_stg1_release.csv")

head(model01)
unique(model01$clss)
head(model02)
head(model03,10)

model03$class <- model03$Type_1 + model03$Type_2 * 2 + model03$Type_3 * 3
model03$class <- model03$class - 1

model02 <- left_join(model02, model03, by = "image_name")
model02$clss <- model02$class
model02$class <- NULL
model02$Type_1 <- NULL
model02$Type_2 <- NULL
model02$Type_3 <- NULL

head(model01)
head(model02)

train2 <- rbind(model01, model02)

write.csv(train2, './input/rectangles_train2.csv', row.names=FALSE, quote = FALSE)
####################################################################################################################

model01 <- read_csv("./submissions/Prav.densenet161.stg1and2.CNN01.csv") 
model02 <- read_csv("./submissions/Prav.densenet161.stg1and2.CNN02_Aug.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$Type_1    <- (all_ensemble$Type_1.x+all_ensemble$Type_1.y)/2
all_ensemble$Type_2    <- (all_ensemble$Type_2.x+all_ensemble$Type_2.y)/2
all_ensemble$Type_3    <- (all_ensemble$Type_3.x+all_ensemble$Type_3.y)/2
head(all_ensemble)

cols <- c("image_name","Type_1","Type_2","Type_3")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.stg1and2_dense161_aug_ensemble.csv', row.names=FALSE, quote = FALSE)
####################################################################################################################

#####################################################################################################################
# combine stage1 and stage2 submit files

model01 <- read_csv("./submissions/Prav_dense161_aug_ensemble.csv") 
model02 <- read_csv("./submissions/Prav.stg1and2_dense161_aug_ensemble.csv")

head(model01)
head(model02)

finalsub <- rbind(model01, model02)
write.csv(finalsub, './submissions/Prav.stg1and2_dense161_stg1and2_aug_ensemble.csv', row.names=FALSE, quote = FALSE)

#####################################################################################################################


model01 <- read_csv("./submissions/Prav.stg2_dense161_aug_ensemble.csv") 
model02 <- read_csv("./submissions/Prav.stg1and2_dense161_aug_ensemble.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])

train <- read_csv('./input/train.csv')
test  <- read_csv('./input/test.csv')
test$loss <- -100

trainFeatures <- read_csv('./input/trainFeatures.csv')
testFeatures  <- read_csv('./input/testFeatures.csv')

train <- left_join(train, trainFeatures, by = "id")
test  <- left_join(test, testFeatures, by = "id")


CVindices5folds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")
names(train)
summary(train$loss)
names(test)

summary(test$loss)

#feature.names     <- grep("cat", names(train), value = T)

# train$catUniqueLength <- apply(train[, feature.names],1, function(x) length(unique(x)))
# test$catUniqueLength  <- apply(test[, feature.names],1, function(x) length(unique(x)))

# set.seed(2016)
# trainCluster <- kmeans(train[, feature.names], 3)
# trainCluster
# 
# train$Cluster <- trainCluster$Cluster


#########################################################################################
#########################################################################################
# 
# train <- as.data.table(train)
# 
# train[,cat80countRatio:= .N/length(train$cat80),by= c("cat80")]
# train[,cat79countRatio:= .N/length(train$cat79),by= c("cat79")]
# train[,cat12countRatio:= .N/length(train$cat12),by= c("cat12")]
# train[,cat81countRatio:= .N/length(train$cat81),by= c("cat81")]
# train[,cat103countRatio:= .N/length(train$cat103),by= c("cat103")]
# train[,cat1countRatio:= .N/length(train$cat1),by= c("cat1")]
# 
# train[,cat8079countRatio:= .N/length(train$cat80),by= c("cat80","cat79")]
# train[,cat8012countRatio:= .N/length(train$cat80),by= c("cat80","cat12")]
# train[,cat8081countRatio:= .N/length(train$cat80),by= c("cat80","cat81")]
# train[,cat80103countRatio:= .N/length(train$cat80),by= c("cat80","cat103")]
# train[,cat801countRatio:= .N/length(train$cat80),by= c("cat80","cat1")]
# 
# train[,cat7912countRatio:= .N/length(train$cat80),by= c("cat79","cat12")]
# train[,cat7981countRatio:= .N/length(train$cat80),by= c("cat79","cat81")]
# train[,cat79103countRatio:= .N/length(train$cat80),by= c("cat79","cat103")]
# train[,cat791countRatio:= .N/length(train$cat80),by= c("cat79","cat1")]
# 
# train[,cat1281countRatio:= .N/length(train$cat80),by= c("cat12","cat81")]
# train[,cat12103countRatio:= .N/length(train$cat80),by= c("cat12","cat103")]
# train[,cat121countRatio:= .N/length(train$cat80),by= c("cat12","cat1")]
# 
# train[,cat81103countRatio:= .N/length(train$cat80),by= c("cat81","cat103")]
# train[,cat811countRatio:= .N/length(train$cat80),by= c("cat81","cat1")]
# 
# train[,cat1031countRatio:= .N/length(train$cat80),by= c("cat103","cat1")]
# 
# train[,cat807912countRatio:= .N/length(train$cat80),by= c("cat80","cat79","cat12")]
# train[,cat807981countRatio:= .N/length(train$cat80),by= c("cat80","cat79","cat81")]
# train[,cat8079103countRatio:= .N/length(train$cat80),by= c("cat80","cat79","cat103")]
# train[,cat80791countRatio:= .N/length(train$cat80),by= c("cat80","cat79","cat1")]
# 
# train[,cat791281countRatio:= .N/length(train$cat80),by= c("cat79","cat12","cat81")]
# train[,cat7912103countRatio:= .N/length(train$cat80),by= c("cat79","cat12","cat103")]
# train[,cat79121countRatio:= .N/length(train$cat80),by= c("cat79","cat12","cat1")]
# train[,cat1281103countRatio:= .N/length(train$cat80),by= c("cat12","cat81","cat103")]
# train[,cat12811countRatio:= .N/length(train$cat80),by= c("cat12","cat81","cat1")]
# train[,cat811031countRatio:= .N/length(train$cat80),by= c("cat81","cat103","cat1")]
# 
# train[,cat100countRatio:= .N/length(train$cat80),by= c("cat100")]
# train[,cat101countRatio:= .N/length(train$cat80),by= c("cat101")]
# train[,cat2countRatio:= .N/length(train$cat80),by= c("cat2")]
# train[,cat53countRatio:= .N/length(train$cat80),by= c("cat53")]
# train[,cat114countRatio:= .N/length(train$cat80),by= c("cat114")]
# train[,cat10countRatio:= .N/length(train$cat80),by= c("cat10")]
# train[,cat57countRatio:= .N/length(train$cat80),by= c("cat57")]
# train[,cat72countRatio:= .N/length(train$cat80),by= c("cat72")]
# train[,cat87countRatio:= .N/length(train$cat80),by= c("cat87")]
# 
# train[,cat100101countRatio:= .N/length(train$cat80),by= c("cat100","cat101")]
# train[,cat1002countRatio:= .N/length(train$cat80),by= c("cat100","cat2")]
# train[,cat10053countRatio:= .N/length(train$cat80),by= c("cat100","cat53")]
# train[,cat100114countRatio:= .N/length(train$cat80),by= c("cat100","cat114")]
# train[,cat10010countRatio:= .N/length(train$cat80),by= c("cat100","cat10")]
# train[,cat10057countRatio:= .N/length(train$cat80),by= c("cat100","cat57")]
# train[,cat10072countRatio:= .N/length(train$cat80),by= c("cat100","cat72")]
# train[,cat10087countRatio:= .N/length(train$cat80),by= c("cat100","cat87")]
# 
# train[,cat1001012countRatio:= .N/length(train$cat80),by= c("cat100","cat101","cat2")]
# train[,cat10010153countRatio:= .N/length(train$cat80),by= c("cat100","cat101","cat53")]
# train[,cat100101114countRatio:= .N/length(train$cat80),by= c("cat100","cat101","cat114")]
# train[,cat10010110countRatio:= .N/length(train$cat80),by= c("cat100","cat101","cat10")]
# train[,cat10010157countRatio:= .N/length(train$cat80),by= c("cat100","cat101","cat57")]
# train[,cat10010172countRatio:= .N/length(train$cat80),by= c("cat100","cat101","cat72")]
# train[,cat10010187countRatio:= .N/length(train$cat80),by= c("cat100","cat101","cat87")]
# 
# train[,cat101253countRatio:= .N/length(train$cat80),by= c("cat101","cat2","cat53")]
# train[,cat1012114countRatio:= .N/length(train$cat80),by= c("cat101","cat2","cat114")]
# train[,cat101210countRatio:= .N/length(train$cat80),by= c("cat101","cat2","cat10")]
# train[,cat101257countRatio:= .N/length(train$cat80),by= c("cat101","cat2","cat57")]
# train[,cat101272countRatio:= .N/length(train$cat80),by= c("cat101","cat2","cat72")]
# train[,cat101287countRatio:= .N/length(train$cat80),by= c("cat101","cat2","cat87")]
# 
# train[,cat253114countRatio:= .N/length(train$cat80),by= c("cat2","cat53","cat114")]
# train[,cat25310countRatio:= .N/length(train$cat80),by= c("cat2","cat53","cat10")]
# train[,cat25357countRatio:= .N/length(train$cat80),by= c("cat2","cat53","cat57")]
# train[,cat25372countRatio:= .N/length(train$cat80),by= c("cat2","cat53","cat72")]
# train[,cat25387countRatio:= .N/length(train$cat80),by= c("cat2","cat53","cat87")]
# 
# train[,cat5311410countRatio:= .N/length(train$cat80),by= c("cat53","cat114","cat10")]
# train[,cat5311457countRatio:= .N/length(train$cat80),by= c("cat53","cat114","cat57")]
# train[,cat5311472countRatio:= .N/length(train$cat80),by= c("cat53","cat114","cat72")]
# train[,cat5311487countRatio:= .N/length(train$cat80),by= c("cat53","cat114","cat87")]
# 
# train[,cat1141057countRatio:= .N/length(train$cat80),by= c("cat114","cat10","cat57")]
# train[,cat1141072countRatio:= .N/length(train$cat80),by= c("cat114","cat10","cat72")]
# train[,cat1141087countRatio:= .N/length(train$cat80),by= c("cat114","cat10","cat87")]
# 
# train[,cat105772countRatio:= .N/length(train$cat80),by= c("cat10","cat57","cat72")]
# train[,cat105787countRatio:= .N/length(train$cat80),by= c("cat10","cat57","cat87")]
# 
# train[,cat577287countRatio:= .N/length(train$cat80),by= c("cat57","cat72","cat87")]
# 
# 
# 
# 
# train <- as.data.frame(train)
# 
# 
# test <- as.data.table(test)
# 
# test[,cat80countRatio:= .N/length(test$cat80),by= c("cat80")]
# test[,cat79countRatio:= .N/length(test$cat79),by= c("cat79")]
# test[,cat12countRatio:= .N/length(test$cat12),by= c("cat12")]
# test[,cat81countRatio:= .N/length(test$cat81),by= c("cat81")]
# test[,cat103countRatio:= .N/length(test$cat103),by= c("cat103")]
# test[,cat1countRatio:= .N/length(test$cat1),by= c("cat1")]
# 
# test[,cat8079countRatio:= .N/length(test$cat80),by= c("cat80","cat79")]
# test[,cat8012countRatio:= .N/length(test$cat80),by= c("cat80","cat12")]
# test[,cat8081countRatio:= .N/length(test$cat80),by= c("cat80","cat81")]
# test[,cat80103countRatio:= .N/length(test$cat80),by= c("cat80","cat103")]
# test[,cat801countRatio:= .N/length(test$cat80),by= c("cat80","cat1")]
# 
# test[,cat7912countRatio:= .N/length(test$cat80),by= c("cat79","cat12")]
# test[,cat7981countRatio:= .N/length(test$cat80),by= c("cat79","cat81")]
# test[,cat79103countRatio:= .N/length(test$cat80),by= c("cat79","cat103")]
# test[,cat791countRatio:= .N/length(test$cat80),by= c("cat79","cat1")]
# 
# test[,cat1281countRatio:= .N/length(test$cat80),by= c("cat12","cat81")]
# test[,cat12103countRatio:= .N/length(test$cat80),by= c("cat12","cat103")]
# test[,cat121countRatio:= .N/length(test$cat80),by= c("cat12","cat1")]
# 
# test[,cat81103countRatio:= .N/length(test$cat80),by= c("cat81","cat103")]
# test[,cat811countRatio:= .N/length(test$cat80),by= c("cat81","cat1")]
# 
# test[,cat1031countRatio:= .N/length(test$cat80),by= c("cat103","cat1")]
# 
# test[,cat807912countRatio:= .N/length(test$cat80),by= c("cat80","cat79","cat12")]
# test[,cat807981countRatio:= .N/length(test$cat80),by= c("cat80","cat79","cat81")]
# test[,cat8079103countRatio:= .N/length(test$cat80),by= c("cat80","cat79","cat103")]
# test[,cat80791countRatio:= .N/length(test$cat80),by= c("cat80","cat79","cat1")]
# 
# test[,cat791281countRatio:= .N/length(test$cat80),by= c("cat79","cat12","cat81")]
# test[,cat7912103countRatio:= .N/length(test$cat80),by= c("cat79","cat12","cat103")]
# test[,cat79121countRatio:= .N/length(test$cat80),by= c("cat79","cat12","cat1")]
# test[,cat1281103countRatio:= .N/length(test$cat80),by= c("cat12","cat81","cat103")]
# test[,cat12811countRatio:= .N/length(test$cat80),by= c("cat12","cat81","cat1")]
# test[,cat811031countRatio:= .N/length(test$cat80),by= c("cat81","cat103","cat1")]
# 
# 
# test[,cat100countRatio:= .N/length(test$cat80),by= c("cat100")]
# test[,cat101countRatio:= .N/length(test$cat80),by= c("cat101")]
# test[,cat2countRatio:= .N/length(test$cat80),by= c("cat2")]
# test[,cat53countRatio:= .N/length(test$cat80),by= c("cat53")]
# test[,cat114countRatio:= .N/length(test$cat80),by= c("cat114")]
# test[,cat10countRatio:= .N/length(test$cat80),by= c("cat10")]
# test[,cat57countRatio:= .N/length(test$cat80),by= c("cat57")]
# test[,cat72countRatio:= .N/length(test$cat80),by= c("cat72")]
# test[,cat87countRatio:= .N/length(test$cat80),by= c("cat87")]
# 
# test[,cat100101countRatio:= .N/length(test$cat80),by= c("cat100","cat101")]
# test[,cat1002countRatio:= .N/length(test$cat80),by= c("cat100","cat2")]
# test[,cat10053countRatio:= .N/length(test$cat80),by= c("cat100","cat53")]
# test[,cat100114countRatio:= .N/length(test$cat80),by= c("cat100","cat114")]
# test[,cat10010countRatio:= .N/length(test$cat80),by= c("cat100","cat10")]
# test[,cat10057countRatio:= .N/length(test$cat80),by= c("cat100","cat57")]
# test[,cat10072countRatio:= .N/length(test$cat80),by= c("cat100","cat72")]
# test[,cat10087countRatio:= .N/length(test$cat80),by= c("cat100","cat87")]
# 
# test[,cat1001012countRatio:= .N/length(test$cat80),by= c("cat100","cat101","cat2")]
# test[,cat10010153countRatio:= .N/length(test$cat80),by= c("cat100","cat101","cat53")]
# test[,cat100101114countRatio:= .N/length(test$cat80),by= c("cat100","cat101","cat114")]
# test[,cat10010110countRatio:= .N/length(test$cat80),by= c("cat100","cat101","cat10")]
# test[,cat10010157countRatio:= .N/length(test$cat80),by= c("cat100","cat101","cat57")]
# test[,cat10010172countRatio:= .N/length(test$cat80),by= c("cat100","cat101","cat72")]
# test[,cat10010187countRatio:= .N/length(test$cat80),by= c("cat100","cat101","cat87")]
# 
# 
# test[,cat101253countRatio:= .N/length(test$cat80),by= c("cat101","cat2","cat53")]
# test[,cat1012114countRatio:= .N/length(test$cat80),by= c("cat101","cat2","cat114")]
# test[,cat101210countRatio:= .N/length(test$cat80),by= c("cat101","cat2","cat10")]
# test[,cat101257countRatio:= .N/length(test$cat80),by= c("cat101","cat2","cat57")]
# test[,cat101272countRatio:= .N/length(test$cat80),by= c("cat101","cat2","cat72")]
# test[,cat101287countRatio:= .N/length(test$cat80),by= c("cat101","cat2","cat87")]
# 
# test[,cat253114countRatio:= .N/length(test$cat80),by= c("cat2","cat53","cat114")]
# test[,cat25310countRatio:= .N/length(test$cat80),by= c("cat2","cat53","cat10")]
# test[,cat25357countRatio:= .N/length(test$cat80),by= c("cat2","cat53","cat57")]
# test[,cat25372countRatio:= .N/length(test$cat80),by= c("cat2","cat53","cat72")]
# test[,cat25387countRatio:= .N/length(test$cat80),by= c("cat2","cat53","cat87")]
# 
# test[,cat5311410countRatio:= .N/length(test$cat80),by= c("cat53","cat114","cat10")]
# test[,cat5311457countRatio:= .N/length(test$cat80),by= c("cat53","cat114","cat57")]
# test[,cat5311472countRatio:= .N/length(test$cat80),by= c("cat53","cat114","cat72")]
# test[,cat5311487countRatio:= .N/length(test$cat80),by= c("cat53","cat114","cat87")]
# 
# test[,cat1141057countRatio:= .N/length(test$cat80),by= c("cat114","cat10","cat57")]
# test[,cat1141072countRatio:= .N/length(test$cat80),by= c("cat114","cat10","cat72")]
# test[,cat1141087countRatio:= .N/length(test$cat80),by= c("cat114","cat10","cat87")]
# 
# test[,cat105772countRatio:= .N/length(test$cat80),by= c("cat10","cat57","cat72")]
# test[,cat105787countRatio:= .N/length(test$cat80),by= c("cat10","cat57","cat87")]
# 
# test[,cat577287countRatio:= .N/length(test$cat80),by= c("cat57","cat72","cat87")]

# test <- as.data.frame(test)


#########################################################################################
#########################################################################################

train_test = rbind(train, test)

#feature.names     <- names(train[,-which(names(train) %in% c("id","loss"))])
#sapply(train_test[,feature.names], class)
# 
# for (f in feature.names) {
#   if (class(train_test[[f]])=="character") {
#     cat("VARIABLE : ",f,"\n")
#     levels <- unique(train_test[[f]])
#     train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
#   }
# }

cate.variables <- grep("cat", names(train_test), value = T)
#cate.Ratio.variables <- grep("Ratio", names(train_test), value = T)

onehot.variables <- cate.variables

formula <-  as.formula(paste("~ ", paste(onehot.variables, collapse= "+"))) 
ohe_feats = onehot.variables

dummies <- dummyVars(formula, data = train_test)
train_test_ohe <- as.data.frame(predict(dummies, newdata = train_test))
train_test_combined <- cbind(train_test[,-c(which(colnames(train_test) %in% ohe_feats))],train_test_ohe)

cont.variables <- grep("cont", names(train_test_combined), value = T)

summary(train_test_combined$cont1)
train_test_combined[,cont.variables] <- apply(train_test_combined[,cont.variables], 2, normalit)


# paste(xnam, collapse= "+")
# ohe_feats = c( 'first_weekday', 'agegroup', 'gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser')
# dummies <- dummyVars(~ first_weekday + agegroup + gender + signup_method + signup_flow + language + affiliate_channel + affiliate_provider + first_affiliate_tracked + signup_app + first_device_type + first_browser, data = df_all)
# df_all_ohe <- as.data.frame(predict(dummies, newdata = df_all))
# df_all_combined <- cbind(df_all[,-c(which(colnames(df_all) %in% ohe_feats))],df_all_ohe)


# for (f in feature.names) {
#   if (class(train_test[[f]])=="numeric" & (skewness(train_test[[f]]) > 0.25 | skewness(train_test[[f]]) < -0.25)) {
#     lambda = BoxCox.lambda( train_test[[f]] )
#     skewness = skewness( train_test[[f]] )
#     kurtosis = kurtosis( train_test[[f]] )
#     cat("VARIABLE : ",f, "lambda : ",lambda, "skewness : ",skewness, "kurtosis : ",kurtosis, "\n")
#     
#   }
# }

# VARIABLE :  cont1  lambda :  0.3619269   skewness :  0.5132021  kurtosis :  -0.1087942 
# VARIABLE :  cont2  lambda :  1.520525    skewness :  -0.3111445 kurtosis :  -0.8924776 
# VARIABLE :  cont4  lambda :  -0.3112189  skewness :  0.4175573  kurtosis :  -0.9611522 
# VARIABLE :  cont5  lambda :  -0.9999242  skewness :  0.6796072  kurtosis :  -0.8815892 
# VARIABLE :  cont6  lambda :  0.01024023  skewness :  0.4584111  kurtosis :  -0.7677769 
# VARIABLE :  cont7  lambda :  -0.3589959  skewness :  0.8258848  kurtosis :  0.04901617 
# VARIABLE :  cont8  lambda :  -0.9999242  skewness :  0.6732339  kurtosis :  -0.5496037 
# VARIABLE :  cont9  lambda :  -0.1835979  skewness :  1.067242   kurtosis :  0.5524011 
# VARIABLE :  cont10 lambda :  0.2010104   skewness :  0.3521146  kurtosis :  -0.855878 
# VARIABLE :  cont11 lambda :  -0.05698178 skewness :  0.2811381  kurtosis :  -1.049778 
# VARIABLE :  cont12 lambda :  -0.06049427 skewness :  0.2919955  kurtosis :  -1.029596 
# VARIABLE :  cont13 lambda :  -0.09996735 skewness :  0.376136   kurtosis :  -1.352632 
# VARIABLE :  cont14 lambda :  -0.9275925  skewness :  0.250672   kurtosis :  -1.529682 


# dt[,.N,by=Species][,prop := N/sum(N)]
# 
# skewness(train_test$cont1)                                     #  0.5132021
# lambda = BoxCox.lambda( train_test$cont1 )
# train_test$cont1TBC = BoxCox( train_test$cont1, lambda)
# train_test$cont1TSQ = sqrt(train_test$cont1)
# skewness(train_test$cont1TBC)                                  # -0.4990672, NOT good 
# skewness(train_test$cont1TSQ)                                  # -0.1353651,     good
# 
# skewness(train_test$cont4)                                     #  0.5132021
# lambda = BoxCox.lambda( train_test$cont4 )
# train_test$cont4TBC =  BoxCox( train_test$cont4, lambda)
# train_test$cont4TSQ =  1/sqrt(train_test$cont4)
# skewness(train_test$cont4TBC)                                  # -0.4137888 ,     good
# skewness(train_test$cont4TSQ)                                  #  0.5452625 , NOT good 
# 
# skewness(train_test$cont5)                                     #  0.6796072
# lambda = BoxCox.lambda( train_test$cont5 )
# train_test$cont5TBC =  BoxCox( train_test$cont5, lambda)
# train_test$cont5TIN =  1/train_test$cont5
# skewness(train_test$cont5TBC)                                  # -0.005050599 , NOT good
# skewness(train_test$cont5TIN)                                  #  0.005071639 ,     good 
# 
# skewness(train_test$cont6)                                     #  0.4584111
# lambda = BoxCox.lambda( train_test$cont6 )
# train_test$cont6TBC =  BoxCox( train_test$cont6, lambda)
# train_test$cont6TLG =  log(train_test$cont6)
# skewness(train_test$cont6TBC)                                  # -0.699844  , NOT good
# skewness(train_test$cont6TLG)                                  # -0.7320321 , NOT good 
# 
# skewness(train_test$cont7)                                     #  0.8258848
# lambda = BoxCox.lambda( train_test$cont7 )
# train_test$cont7TBC =  BoxCox( train_test$cont7, lambda)
# train_test$cont7TSQ =  1/sqrt(train_test$cont7)
# skewness(train_test$cont7TBC)                                  # -0.6235395  ,     good
# skewness(train_test$cont7TSQ)                                  #  0.990795   , NOT good 
# 
# skewness(train_test$cont8)                                     #  0.6732339
# lambda = BoxCox.lambda( train_test$cont8 )
# train_test$cont8TBC =  BoxCox( train_test$cont8, lambda)
# train_test$cont8TIN =  1/train_test$cont8
# skewness(train_test$cont8TBC)                                  # -0.3023764  , NOT good
# skewness(train_test$cont8TIN)                                  # 0.3024085   ,     good 
# 
# skewness(train_test$cont9)                                     #  1.067242
# lambda = BoxCox.lambda( train_test$cont9 )
# train_test$cont9TBC =  BoxCox( train_test$cont9, lambda)
# skewness(train_test$cont9TBC)                                  # -22.28253  ,      good
# 
# skewness(train_test$cont10)                                     #  0.3521146
# lambda = BoxCox.lambda( train_test$cont10 )
# train_test$cont10TBC =  BoxCox( train_test$cont10, lambda)
# skewness(train_test$cont10TBC)                                  # -1.980067  , NOT good
# 
# skewness(train_test$cont14)                                     #  0.250672
# lambda = BoxCox.lambda( train_test$cont14 )
# train_test$cont14TBC =  BoxCox( train_test$cont14, lambda)
# train_test$cont14TIN =  1/train_test$cont14
# skewness(train_test$cont14TBC)                                  # -0.4938727  , NOT good
# skewness(train_test$cont14TIN)                                  #  0.528312   , NOT good 
# 
# features.bad.transformation <- c("cont1TBC","cont4TSQ","cont5TBC","cont6TBC","cont6TLG","cont7TSQ","cont8TBC","cont10TBC","cont14TBC","cont14TIN")
# features.remove.totransform <- c("cont1","cont4","cont5","cont7","cont8","cont9")

# training <- train_test[train_test$loss != -100,]
# testing  <- train_test[train_test$loss == -100,]


training <- train_test_combined[train_test_combined$loss != -100,]
testing  <- train_test_combined[train_test_combined$loss == -100,]

summary(training$loss)
summary(testing$loss)

testing$loss <- NULL

###################################################################################
# skewness tests
###################################################################################
# skewness(trainingSet$cont2) #  0.5164158
# kurtosis(trainingSet$cont1) # -0.1017717
# 
# ggplot(train_test, aes(x = cont2), binwidth = 2) + 
#   geom_histogram(aes(y = ..density..), fill = 'red', alpha = 0.5) + 
#   geom_density(colour = 'blue') + 
#   xlab(expression(bold('Simulated Samples'))) + 
#   ylab(expression(bold('Density')))
# 
# lambda = BoxCox.lambda( trainingSet$cont1 )
# trainingSet$cont1BC = BoxCox( trainingSet$cont1, lambda)

###################################################################################
###################################################################################


trainingSet <- left_join(training, CVindices5folds, by = "id")
testingSet  <- testing

rm(train,test, training, testing, train_test ,CVindices5folds); gc()

write.csv(trainingSet, paste(root_directory, "/input/train_deeplearningOhe.csv", sep=''), row.names=FALSE, quote = FALSE)
write.csv(testingSet, paste(root_directory, "/input/test_deeplearningOhe.csv", sep=''), row.names=FALSE, quote = FALSE)

################################################################################################

# Sys.time()
# save.image(file = "Allstate_Baseline01_20161014.RData" , safe = TRUE)
# Sys.time()

################################################################################################


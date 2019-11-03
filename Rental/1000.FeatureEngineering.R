


train.json <- fromJSON("./input/train.json")
# There has to be a better way to do this while getting repeated rows for the "feature" and "photos" columns
train <- data.table(bathrooms=unlist(train.json$bathrooms)
                 ,bedrooms=unlist(train.json$bedrooms)
                 ,building_id=unlist(train.json$building_id)
                 ,created=as.POSIXct(unlist(train.json$created))
                 ,description=unlist(train.json$description) 
                 ,display_address=unlist(train.json$display_address) 
                 ,latitude=unlist(train.json$latitude)
                 ,longitude=unlist(train.json$longitude)
                 ,listing_id=unlist(train.json$listing_id)
                 ,manager_id=unlist(train.json$manager_id)
                 ,price=unlist(train.json$price)
                 ,interest_level=unlist(train.json$interest_level)
                 ,street_adress=unlist(train.json$street_address)
               
)

train.featurePhotos <- data.table( listing_id=unlist(train.json$listing_id)
                       ,features=unlist(train.json$features) 
                      ,photos=unlist(train.json$photos) 
)


train.featurePhotos$features = gsub(" ","_", train.featurePhotos$features)

train.featurePhotosNormalize <- ddply(train.featurePhotos, .(listing_id), summarize, features = toString(features), photos = toString(photos))


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

test.json <- fromJSON("./input/test.json")
# There has to be a better way to do this while getting repeated rows for the "feature" and "photos" columns
test <- data.table(bathrooms=unlist(test.json$bathrooms)
                   ,bedrooms=unlist(test.json$bedrooms)
                   ,building_id=unlist(test.json$building_id)
                   ,created=as.POSIXct(unlist(test.json$created))
                   ,description=unlist(test.json$description)
                   ,display_address=unlist(test.json$display_address) 
                   ,latitude=unlist(test.json$latitude)
                   ,longitude=unlist(test.json$longitude)
                   ,listing_id=unlist(test.json$listing_id)
                   ,manager_id=unlist(test.json$manager_id)
                   ,price=unlist(test.json$price)
                   ,street_adress=unlist(test.json$street_address) 
                  
)


test.featurePhotos <- data.table( listing_id=unlist(test.json$listing_id)
                                   ,features=unlist(test.json$features)
                                   ,photos=unlist(test.json$photos) 
)

test.featurePhotos$features = gsub(" ","_", test.featurePhotos$features)

test.featurePhotosNormalize <- ddply(test.featurePhotos, .(listing_id), summarize, features = toString(features), photos = toString(photos))

rm(train.featurePhotos, test.featurePhotos); gc()

train <- left_join(train, train.featurePhotosNormalize, by= "listing_id")
test  <- left_join(test, test.featurePhotosNormalize, by= "listing_id")

rm(train.featurePhotosNormalize, test.featurePhotosNormalize); gc()

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################



train_features01 <- read_csv("./input/train_features01.csv")
test_features01  <- read_csv("./input/test_features01.csv")

train <- left_join(train, train_features01, by = "listing_id")
test  <- left_join(test, test_features01, by = "listing_id")

rm(train_features01, test_features01); gc()

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
names(train)
train <- subset(train, select=c(1:11,13:22,12))
names(test)
test$interest_level <- "none"

unique(train$interest_level)
unique(test$interest_level)

all_data <- rbind(train, test)
gc()

############################################################################################################################################
# Initial Step
# Sys.time()
# save.image(file = "Rental_01.RData" , safe = TRUE)
# Sys.time()
# load("Rental_01.RData")
# Sys.time()
#############################################################################################################################################

#############################################################################################################################################



all_data$address_similarity          <- stringdist(all_data$display_address,all_data$street_adress,method='jw')
all_data$address_similarity_cosine   <- stringdist(all_data$display_address,all_data$street_adress,method='cosine')
all_data$address_similarity_jaccard  <- stringdist(all_data$display_address,all_data$street_adress,method='jaccard')
# chance for cleaning of numbers
all_data$address_similarity_sound    <- stringdist(all_data$display_address,all_data$street_adress,method='soundex')

head(all_data$display_address)
head(all_data$street_adress)
head(all_data$address_similarity)
head(all_data$address_similarity_sound)

head(all_data[, category.features])

StreetAddressDTM = function(categories){  
  categories = Corpus(VectorSource(categories))
  categories <- tm_map(categories, content_transformer(function(x) {
    x = gsub("-"," ", tolower(x)) 
    x = gsub("boulevard","blvd",x)
    x = gsub("st","street",x)
    
    return(x)
  } ))
  categories <- tm_map(categories, removePunctuation)
  categories <- tm_map(categories, removeNumbers)
  categories <- tm_map(categories, removeWords, c(stopwords("english")
                                                  , c("th")
  ))
  categories <- tm_map(categories, stemDocument, language="english")  
  dtm = DocumentTermMatrix(categories)
  #dtm <- removeSparseTerms(dtm, 0.1)
  #dtm = DocumentTermMatrix(categories,control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))
  #dtm <- as.data.frame(as.matrix(dtm))
  return(dtm);
}

StreetAddressDTM = StreetAddressDTM(all_data$street_adress)
StreetAddressfreqterms = findFreqTerms(StreetAddressDTM, 100,Inf) # min 50 freq terms
StreetAddressDTM = as.data.frame(as.matrix(StreetAddressDTM))

# head(categoriesDTM)
rownames(StreetAddressDTM) <- all_data$listing_id
streedAddress_names = colnames(StreetAddressDTM)


StreetAddressFreqTerms <- StreetAddressDTM[, StreetAddressfreqterms]


#all_data <- cbind(all_data, categoriesDTM)
all_data <- cbind(all_data, StreetAddressFreqTerms)

category.features <- c("display_address", "manager_id", "building_id", "street_adress")

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in category.features) {
  
    levels <- unique(c(all_data[[f]]))
    all_data[[f]] <- as.integer(factor(all_data[[f]], levels=levels))
    
  
}


#############################################################################################################################################
getCategoriesDTM = function(categories){  
  categories = Corpus(VectorSource(categories))
  categories <- tm_map(categories, content_transformer(function(x) {
    x = gsub("-"," ", tolower(x)) 
    x = gsub("absolut","absolute",x) 
    x = gsub("attaché","attach",x)
    
    x = gsub("appli","application",x)
    x = gsub("applianc","application",x)
    x = gsub("applic","application",x)
    x = gsub("applicationationationc","application",x)
    x = gsub("applicationationation","application",x)
    x = gsub("appl","application",x)
    
    x = gsub("babi","bbq",x)
    x = gsub("bbqs","bbq",x)
    x = gsub("barbecu","bbq",x)
    
    x = gsub("blk","block",x)
    x = gsub("blks","block",x)
    
    x = gsub("price","prices",x)
    
    x = gsub("pictures","picture",x)
    x = gsub("pregnancy","pregnant",x)
    x = gsub("products","product",x)
    x = gsub("properties","property",x)
    x = gsub("puzzel","puzzle",x)
    x = gsub("puzzles","puzzle",x)
    x = gsub("rentals","rental",x)
    x = gsub("regional","region",x)
    x = gsub("reservations","reservation",x)
    x = gsub("tourism","tourist",x)
    x = gsub('photo.*','photo', x)
    return(x)
  } ))
  categories <- tm_map(categories, removePunctuation)
  categories <- tm_map(categories, removeNumbers)
  categories <- tm_map(categories, removeWords, c(stopwords("english")
                                                   , c("the", "third", "three", "non", "and", "app", "not",'group',  "version")
                                                  ))
  categories <- tm_map(categories, stemDocument, language="english")  
  dtm = DocumentTermMatrix(categories)
  #dtm <- removeSparseTerms(dtm, 0.1)
  #dtm = DocumentTermMatrix(categories,control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))
  #dtm <- as.data.frame(as.matrix(dtm))
  return(dtm);
}


categoriesDTM = getCategoriesDTM(all_data$features)
freqterms = findFreqTerms(categoriesDTM, 10,Inf) # min 5 freq terms
categoriesDTM = as.data.frame(as.matrix(categoriesDTM))
# descrCor <- cor(categoriesDTM)
# summary(descrCor[upper.tri(descrCor)])
# highlyCorDescr <- findCorrelation(descrCor, cutoff = 0.90)
# train1 <- train1[,-highlyCorDescr]

# head(categoriesDTM)
rownames(categoriesDTM) <- all_data$listing_id
term_names = colnames(categoriesDTM)

# dim(categoriesDTM)
# 
# head(categoriesDTM)

categoriesFreqTerms <- categoriesDTM[, freqterms]
categoriesDTM$price <- NULL

#all_data <- cbind(all_data, categoriesDTM)
all_data <- cbind(all_data, categoriesFreqTerms)



#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

# all_data$park   <- NULL
# all_data$terrac <- NULL

train <- all_data[all_data$interest_level != "none",] #49,352
test  <- all_data[all_data$interest_level == "none",] #74,659


#length(names(train))
#length(unique(names(train)))


train <- as.data.frame(train)

Prav_CV_5folds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")
Prav_CV_5folds$interest_level <- NULL

trainingSet <- left_join(train, Prav_CV_5folds, by = "listing_id")

classnames = levels(as.factor(trainingSet$interest_level))

trainingSet$target <- (as.integer(as.factor(trainingSet$interest_level))-1)
num.class = length(classnames)
summary(trainingSet$target)


testingSet <- as.data.frame(test)

feature.names     <- names(trainingSet[,-which(names(trainingSet) %in% c(
                                                                         "interest_level",
                                                                         "target",
                                                                         "created", 
                                                                         "description",
                                                                         "photos",
                                                                         "features",
                                                                         "CVindices" ))]
                          )

summary(trainingSet[, feature.names])
##################################################################################

# order of columns are matching 
##################################################################################

# Check for all columns order, all columns orders are same after removing id columns


cv          = 5
bags        = 1
nround.cv   = 840 
printeveryn = 100
seed        = 2017
earlystoprounds = 20

## for all remaining models, use same parameters 

param <- list(  "objective"        = "multi:softprob", 
                "booster"          = "gbtree",
                "eval_metric"      = "mlogloss",
                "num_class"        = num.class,
                "nthread"          = 20,     
                "max_depth"        = 6,     
                "eta"              = 0.05, 
                "subsample"        = 0.7,  
                "colsample_bytree" = 0.7,  
                "min_child_weight" = 1     
                
)


cat(cv, "-fold Cross Validation\n", sep = "")
set.seed(seed)
for (i in 1:cv)
  
{
  cat(i ,"fold Processing\n")
  
  cat("X_build fold Processing\n")
  X_build <- subset(trainingSet, CVindices != i, select = -c( CVindices))
  cat("X_val fold Processing\n")
  X_val   <- subset(trainingSet, CVindices == i, select = -c( CVindices)) 
  
  
  dtrain <-xgb.DMatrix(data=data.matrix(X_build[, feature.names]),label=X_build$target)
  dval   <-xgb.DMatrix(data=data.matrix(X_val[, feature.names])  ,label=X_val$target)
  watchlist <- list( train = dtrain,val = dval)
  
  
  pred_cv_bags   <- rep(0, nrow(X_val[, feature.names])* num.class)
  #pred_test_bags <- rep(0, nrow(testingSet[,feature.names])* num.class)
  
  for (b in 1:bags) 
  {
    # seed = seed + b
    # set.seed(seed)
    cat(b ," - bag Processing\n")
    cat("X_build training Processing\n")
    XGModel <- xgb.train(   params              = param,
                            data                = dtrain,
                            watchlist           = watchlist,
                            nrounds             = nround.cv ,
                            print_every_n       = printeveryn,
                            verbose             = TRUE, 
                            #maximize            = TRUE,
                            early_stopping_rounds = earlystoprounds,
                            set.seed            = seed
    )
    cat("X_val prediction Processing\n")
    pred_cv    <- predict(XGModel, data.matrix(X_val[,feature.names]), type = "prob")
    
    cat("CV TestingSet prediction Processing\n")
    #pred_test  <- predict(XGModel, data.matrix(testingSet[,feature.names]), type = "prob")
    
    
    pred_cv_bags   <- pred_cv_bags + pred_cv
    #pred_test_bags <- pred_test_bags + pred_test
  }
  pred_cv_bags   <- pred_cv_bags / bags
  #pred_test_bags <- pred_test_bags / bags
  
  pred_cv_bags <- as.data.frame(t(matrix(pred_cv_bags, nrow=num.class, ncol=length(pred_cv_bags)/num.class)))
  names(pred_cv_bags) <- classnames
  
  #pred_test_bags <- as.data.frame(t(matrix(pred_test_bags, nrow=num.class, ncol=length(pred_test_bags)/num.class)))
  #names(pred_test_bags) <- classnames
  
  #cat("CV Fold-", i, " ", metric, ": ", score(X_val$loss, pred_cv_bags, metric), "\n", sep = "")
  
  val_predictions <- data.frame(cbind(listing_id=X_val$listing_id,  pred_cv_bags))
  val_predictions <- val_predictions[,c(1,2,4,3)]
  #test_predictions <- data.frame(cbind(listing_id=testingSet$listing_id, pred_test_bags))
  #test_predictions <- test_predictions[,c(1,2,4,3)]


  # if(i == 1)
  # {
  #   write.csv(val_predictions,  './submissions/prav.xgb01.fold1.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.xgb01.fold1-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 2)
  # {
  #   write.csv(val_predictions,  './submissions/prav.xgb01.fold2.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.xgb01.fold2-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 3)
  # {
  #   write.csv(val_predictions,  './submissions/prav.xgb01.fold3.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.xgb01.fold3-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 4)
  # {
  #   write.csv(val_predictions,  './submissions/prav.xgb01.fold4.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.xgb01.fold4-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 5)
  # {
  #   write.csv(val_predictions,  './submissions/prav.xgb01.fold5.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.xgb01.fold5-test.csv', row.names=FALSE, quote = FALSE)
  # }
  
}

# Full training

dtrain<-xgb.DMatrix(data=data.matrix(trainingSet[, feature.names]),label=log(trainingSet$loss+constant))
watchlist <- list( train = dtrain)

fulltrainnrounds = 1.1 * nround.cv


# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

fulltest_ensemble <- rep(0, nrow(testingSet[,feature.names])*num.class)

for (b in 1:bags) {
  # seed = seed + b
  # set.seed(seed)
  cat(b ," - bag Processing\n")
  cat("Bagging Full TrainingSet training\n")
  XGModelFulltrain <- xgb.train(    params              = param,
                                    data                = dtrain,
                                    watchlist           = watchlist,
                                    nrounds             = fulltrainnrounds,
                                    print_every_n       = printeveryn,
                                    verbose             = TRUE, 
                                    #maximize            = TRUE,
                                    set.seed            = seed
  )
  cat("Bagging Full Model prediction Processing\n")
  
  predfull_test         <- predict(XGModelFulltrain, data.matrix(testingSet[,testfeature.names]))
  
  fulltest_ensemble <- fulltest_ensemble + (exp(predfull_test) - constant)
  
}

fulltest_ensemble <- fulltest_ensemble / bags
testfull_predictions  <- data.frame(id=testingSet$id, loss = fulltest_ensemble)
write.csv(testfull_predictions, './submissions/prav.xgb01.full.csv', row.names=FALSE, quote = FALSE)




# #########################################################################################################
# Bagging - Full train
# #########################################################################################################



# Sys.time()
# load("Rental_01.RData") # 22 basic features from source files
# Sys.time()
#############################################################################################################################################



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
                                   
)

train[,":="(wday=wday(created))]
train$weekend <- ifelse(train$wday == 0 | train$wday == 6,1,0)


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
                                  
)
test[,":="(wday=wday(created))]
test$weekend <- ifelse(test$wday == 0 | test$wday == 6,1,0)

###############################################################################################################


StandardingString <- function(str) {
  str <- tolower(str)
  str = gsub("hardwood floors","hardwood", str, ignore.case =  TRUE)
  str = gsub("laundry in building","laundry", str, ignore.case =  TRUE)
  str = gsub("laundry in unit","laundry", str, ignore.case =  TRUE)
  str = gsub("laundry room","laundry", str, ignore.case =  TRUE)
  str = gsub("on-site laundry","laundry", str, ignore.case =  TRUE)
  str = gsub("dryer in unit","laundry", str, ignore.case =  TRUE)
  str = gsub("washer in unit","laundry", str, ignore.case =  TRUE)
  str = gsub("washer/dryer","laundry", str, ignore.case =  TRUE)
  str = gsub("roof-deck","roof deck", str, ignore.case =  TRUE)
  str = gsub("common roof deck","roof deck", str, ignore.case =  TRUE)
  str = gsub("roofdeck","roof deck", str, ignore.case =  TRUE)
  
  str = gsub("outdoor space","outdoor", str, ignore.case =  TRUE)
  str = gsub("common outdoor space","outdoor", str, ignore.case =  TRUE)
  str = gsub("private outdoor space","outdoor", str, ignore.case =  TRUE)
  str = gsub("publicoutdoor","outdoor", str, ignore.case =  TRUE)
  str = gsub("outdoor areas","outdoor", str, ignore.case =  TRUE)
  str = gsub("private outdoor","outdoor", str, ignore.case =  TRUE)
  str = gsub("common outdoor","outdoor", str, ignore.case =  TRUE)
  
  str = gsub("garden/patio","garden", str, ignore.case =  TRUE)
  str = gsub("residents garden","garden", str, ignore.case =  TRUE)
  
  str = gsub("parking space","parking", str, ignore.case =  TRUE)
  str = gsub("common parking/garage","parking", str, ignore.case =  TRUE)
  str = gsub("on-site garage","parking", str, ignore.case =  TRUE)
  
  str = gsub("fitness center","fitness", str, ignore.case =  TRUE)
  str = gsub("gym","fitness", str, ignore.case =  TRUE)
  str = gsub("gym/fitness","fitness", str, ignore.case =  TRUE)
  str = gsub("fitness/fitness","fitness", str, ignore.case =  TRUE)
  
  str = gsub("cats allowed","pets", str, ignore.case =  TRUE)
  str = gsub("dogs allowed","pets", str, ignore.case =  TRUE)
  str = gsub("pets on approval","pets", str, ignore.case =  TRUE)
  
  str = gsub("live-in superintendent","live-in super", str, ignore.case =  TRUE)
  
  str = gsub("full-time doorman","doorman", str, ignore.case =  TRUE)
  str = gsub("newly renovated","renovated", str, ignore.case =  TRUE)
  str = gsub("pre-war","prewar", str, ignore.case =  TRUE)
  
  return (str)
}

train.featurePhotos$features <- StandardingString(train.featurePhotos$features)
train.featurePhotos$features = gsub(" ","_", train.featurePhotos$features)
setDT(train.featurePhotos)[, num_uniquefeatures := uniqueN(features), by = listing_id]
train.featurePhotosNormalize <- ddply(train.featurePhotos, .(listing_id), summarize, features = toString(features), count_uniquefeatures = mean(num_uniquefeatures))

test.featurePhotos$features <- StandardingString(test.featurePhotos$features)
test.featurePhotos$features = gsub(" ","_", test.featurePhotos$features)
setDT(test.featurePhotos)[, num_uniquefeatures := uniqueN(features), by = listing_id]
test.featurePhotosNormalize <- ddply(test.featurePhotos, .(listing_id), summarize, features = toString(features), count_uniquefeatures = mean(num_uniquefeatures))

train <- left_join(train, train.featurePhotosNormalize, by= "listing_id")
test  <- left_join(test, test.featurePhotosNormalize, by= "listing_id")

train_features01 <- read_csv("./input/train_features01.csv")
test_features01  <- read_csv("./input/test_features01.csv")

train <- left_join(train, train_features01, by = "listing_id")
test  <- left_join(test, test_features01, by = "listing_id")

rm(train_features01, test_features01); gc()


names(train)
train <- subset(train, select=c(1:11,13:24,12))
names(test)
test$interest_level <- "none"

unique(train$interest_level)
unique(test$interest_level)

all_data <- rbind(train, test)
gc()


txt <- "a patterned layer within a microelectronicpattern pattern."
txt_replaced <- gsub("\\<pattern\\>","form",txt)
txt_replaced

#############################################################################################################################################

StandardingString <- function(str) {
  
  # str <- gsub("[^[:graph:]]", " ",str, ignore.case = TRUE)
  # str <- gsub('[[:digit:]]+', '', str)
  # str <- tolower(str)
  # str = gsub("th"," ", str, ignore.case =  TRUE)
  # str = gsub("  "," ", str, ignore.case =  TRUE)
  # str = gsub("^\\s+|\\s+$", "", str, ignore.case =  TRUE)
  str <- tolower(str)
  str = gsub("\\<w\\> ","west", str, ignore.case =  TRUE)
  str = gsub("\\<st.\\>","street", str, ignore.case =  TRUE)
  str = gsub("\\<st\\>","street", str, ignore.case =  TRUE)
  str = gsub("\\<ave\\>","avenue", str, ignore.case =  TRUE)
  str = gsub("\\<e\\>","east", str, ignore.case =  TRUE)
  str = gsub("\\<n\\>","north", str, ignore.case =  TRUE)
  str = gsub("\\<s\\>","south", str, ignore.case =  TRUE)

  return (str)
}

head(all_data$display_address)
all_data$street_adress               <- StandardingString(all_data$street_adress)
all_data$display_address             <- StandardingString(all_data$display_address)

all_data$address_similarity          <- stringdist(all_data$display_address,all_data$street_adress,method='jw')
# all_data$address_similarity_cosine   <- stringdist(all_data$display_address,all_data$street_adress,method='cosine')
# all_data$address_similarity_jaccard  <- stringdist(all_data$display_address,all_data$street_adress,method='jaccard')
# chance for cleaning of numbers
all_data$address_similarity_sound    <- stringdist(all_data$display_address,all_data$street_adress,method='soundex')

all_data$StreetEast  <- ifelse(grepl("east",all_data$street_adress),1,0)
all_data$StreetNorth <- ifelse(grepl("north",all_data$street_adress),1,0)
all_data$StreetSouth <- ifelse(grepl("south",all_data$street_adress),1,0)
all_data$StreetWest  <- ifelse(grepl("west",all_data$street_adress),1,0)


head(all_data$StreetEast)

getFeaturesDTM = function(features){  
  features = Corpus(VectorSource(features))
  features <- tm_map(features, content_transformer(function(x) {
    x = gsub("-"," ", tolower(x)) 
    
    return(x)
  } ))
  features <- tm_map(features, removePunctuation)
  features <- tm_map(features, removeNumbers)
  features <- tm_map(features, removeWords, c(stopwords("english")
                                              
  ))
  #categories <- tm_map(categories, stemDocument, language="english")  
  dtm = DocumentTermMatrix(features)
  return(dtm);
}


#head(all_data$features)

all_data$features <- gsub(","," " , all_data$features)

featuresDTM = getFeaturesDTM(all_data$features)
freqterms = findFreqTerms(featuresDTM, 10) # min 5 freq terms
featuresDTM.matrix = as.data.frame(as.matrix(featuresDTM))
rownames(featuresDTM.matrix) <- all_data$listing_id
term_names = colnames(featuresDTM.matrix)

#names(featuresDTM.matrix)
featuresFreqTerms <- featuresDTM.matrix[, freqterms]

#head(all_data,1)
#all_data <- cbind(all_data, featuresDTM.matrix)
all_data <- cbind(all_data, featuresFreqTerms)

streetaddressDTM = function(streetaddress){
  streetaddress = Corpus(VectorSource(streetaddress))

  streetaddress <- tm_map(streetaddress, removePunctuation)
  streetaddress <- tm_map(streetaddress, removeNumbers)
  streetaddress <- tm_map(streetaddress, removeWords, c(stopwords("english")

  ))
  #categories <- tm_map(categories, stemDocument, language="english")
  dtm = DocumentTermMatrix(streetaddress)
  return(dtm);
}

head(all_data$street_adress)

streetaddressDTM = streetaddressDTM(all_data$street_adress)
streetsfreqterms = findFreqTerms(streetaddressDTM, 50)
streetaddressDTM.matrix = as.data.frame(as.matrix(streetaddressDTM))

streetFreqTerms <- streetaddressDTM.matrix[, streetsfreqterms]
all_data <- cbind(all_data, streetFreqTerms)

#summary(all_data$longitude)

#all_data <- all_data1

all_data <- as.data.frame(all_data)

# MissingCoord <- all_data[all_data$longitude == 0 | all_data$latitude == 0, ]
# AllgoodData  <- all_data[all_data$longitude != 0 & all_data$latitude != 0, ]
# 
# 
# outliers_addrs <- MissingCoord$street_adress
# outliers_addrs
# 
# outliers_ny <- paste(outliers_addrs, ", new york")
# 
# outliers_addrs <- data.frame("street_address" = outliers_addrs)
# 
# coords <- sapply(outliers_ny,
#                  function(x) geocode(x, source = "google")) %>%
#   t %>%
#   data.frame %>%
#   cbind(outliers_addrs, .)
# 
# substrRight <- function(x, n){
#   substr(x, nchar(x)-n+1, nchar(x))
# }
# 
# 
# substrRight(coords$lon, 7)
# 
# coords$lon
# coords <- as.data.frame(coords)
# MissingCoord$street_adress
# MissingCoords <- cbind(MissingCoord, coords)
# MissingCoords$lat
# MissingCoord$longitude <- 
# rownames(coords) <- 1:nrow(coords)
# # Display table
# kable(coords) 
# 
# all_data[all_data$longitude == 0,]$longitude <- coords$lon
# all_data[all_data$latitude == 0,]$longitude <- coords$lat


# New York City Center Coords
ny_lat <- 40.785091
ny_lon <- -73.968285

all_data$distance_city <-
  mapply(function(lon, lat) sqrt((lon - ny_lon)^2  + (lat - ny_lat)^2),
         all_data$longitude,
         all_data$latitude) 

all_data$price_per_bedroom          <- ifelse(all_data$bedrooms == 0, 1,  all_data$price/all_data$bedrooms)
all_data$price_per_bedbathroom      <- ifelse(all_data$bedrooms+all_data$bathrooms == 0, 1,all_data$price/(all_data$bedrooms+all_data$bathrooms))
all_data$price_per_uniquefeature    <- all_data$price/all_data$count_uniquefeatures
all_data$price_per_badminusbathroom <- ifelse(all_data$bedrooms - all_data$bathrooms == 0, 1,all_data$price/(all_data$bedrooms - all_data$bathrooms))
all_data$bed_bath_ration            <- ifelse(all_data$bathrooms ==0,0,all_data$bedrooms/all_data$bathrooms)


summary(all_data$bed_bath_ration)
category.features <- c("display_address", "manager_id", "building_id","street_adress")
head(all_data[, category.features])
cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in category.features) {
  
  levels <- unique(c(all_data[[f]]))
  all_data[[f]] <- as.integer(factor(all_data[[f]], levels=levels))
  
  
}

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

train <- all_data[all_data$interest_level != "none",] #49,352
test  <- all_data[all_data$interest_level == "none",] #74,659


#length(names(train))
#length(unique(names(train)))
train$terrace <- NULL
test$terrace  <- NULL

train <- as.data.frame(train)

Prav_CV_5folds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")
Prav_CV_5folds$interest_level <- NULL

trainingSet <- left_join(train, Prav_CV_5folds, by = "listing_id")

classnames = levels(as.factor(trainingSet$interest_level))

trainingSet$target <- (as.integer(as.factor(trainingSet$interest_level))-1)
num.class = length(classnames)
#summary(trainingSet$target)


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
# 301 features
summary(trainingSet[, feature.names])
##################################################################################

# order of columns are matching 
##################################################################################

# Check for all columns order, all columns orders are same after removing id columns


cv          = 5
bags        = 1
nround.cv   = 4000 
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
                "eta"              = 0.01, 
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
  pred_test_bags <- rep(0, nrow(testingSet[,feature.names])* num.class)
  
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
    pred_test  <- predict(XGModel, data.matrix(testingSet[,feature.names]), type = "prob")
    
    
    pred_cv_bags   <- pred_cv_bags + pred_cv
    pred_test_bags <- pred_test_bags + pred_test
  }
  pred_cv_bags   <- pred_cv_bags / bags
  pred_test_bags <- pred_test_bags / bags
  
  pred_cv_bags <- as.data.frame(t(matrix(pred_cv_bags, nrow=num.class, ncol=length(pred_cv_bags)/num.class)))
  names(pred_cv_bags) <- classnames
  
  pred_test_bags <- as.data.frame(t(matrix(pred_test_bags, nrow=num.class, ncol=length(pred_test_bags)/num.class)))
  names(pred_test_bags) <- classnames
  
  #cat("CV Fold-", i, " ", metric, ": ", score(X_val$loss, pred_cv_bags, metric), "\n", sep = "")
  
  val_predictions <- data.frame(cbind(listing_id=X_val$listing_id,  pred_cv_bags))
  val_predictions <- val_predictions[,c(1,2,4,3)]
  test_predictions <- data.frame(cbind(listing_id=testingSet$listing_id, pred_test_bags))
  test_predictions <- test_predictions[,c(1,2,4,3)]
  
  
  # if(i == 1)
  # {
  #   write.csv(val_predictions,  './submissions/prav.xgb07.fold1.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.xgb07.fold1-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 2)
  # {
  #   write.csv(val_predictions,  './submissions/prav.xgb07.fold2.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.xgb07.fold2-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 3)
  # {
  #   write.csv(val_predictions,  './submissions/prav.xgb07.fold3.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.xgb07.fold3-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 4)
  # {
  #   write.csv(val_predictions,  './submissions/prav.xgb07.fold4.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.xgb07.fold4-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 5)
  # {
  #   write.csv(val_predictions,  './submissions/prav.xgb07.fold5.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.xgb07.fold5-test.csv', row.names=FALSE, quote = FALSE)
  # }
  
}

# Full training

dtrain<-xgb.DMatrix(data=data.matrix(trainingSet[, feature.names]),label=trainingSet$target)
watchlist <- list( train = dtrain)

fulltrainnrounds = as.integer(1.2 * nround.cv)


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
  
  predfull_test         <- predict(XGModelFulltrain, data.matrix(testingSet[,feature.names]), type = "prob")
  fulltest_ensemble     <- fulltest_ensemble + predfull_test
  
}

fulltest_ensemble <- fulltest_ensemble / bags
fulltest_ensemble <- as.data.frame(t(matrix(fulltest_ensemble, nrow=num.class, ncol=length(fulltest_ensemble)/num.class)))
names(fulltest_ensemble) <- classnames

testfull_predictions <- data.frame(cbind(listing_id=testingSet$listing_id, fulltest_ensemble))
testfull_predictions <- testfull_predictions[,c(1,2,4,3)]

write.csv(testfull_predictions, './submissions/prav.xgb07.full.csv', row.names=FALSE, quote = FALSE)




# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

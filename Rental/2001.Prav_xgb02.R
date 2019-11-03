# Sys.time()
# load("Rental_01.RData") # 22 basic features from source files
# Sys.time()
#############################################################################################################################################

#############################################################################################################################################
head(all_data$street_adress)

StandardingString <- function(str) {
  
  str <- gsub("[^[:graph:]]", " ",str, ignore.case = TRUE)
  str <- gsub('[[:digit:]]+', '', str)
  str <- tolower(str)
  str = gsub("th"," ", str, ignore.case =  TRUE)
  str = gsub("  "," ", str, ignore.case =  TRUE)
  str = gsub("^\\s+|\\s+$", "", str, ignore.case =  TRUE)
  return (str)
}
all_data$street_adress         <- StandardingString(all_data$street_adress)
all_data$display_address         <- StandardingString(all_data$display_address)

all_data$address_similarity          <- stringdist(all_data$display_address,all_data$street_adress,method='jw')
# all_data$address_similarity_cosine   <- stringdist(all_data$display_address,all_data$street_adress,method='cosine')
# all_data$address_similarity_jaccard  <- stringdist(all_data$display_address,all_data$street_adress,method='jaccard')
# chance for cleaning of numbers
all_data$address_similarity_sound    <- stringdist(all_data$display_address,all_data$street_adress,method='soundex')

head(all_data$display_address)
head(all_data$street_adress)
summary(all_data$address_similarity)
# summary(all_data$address_similarity_cosine)
# summary(all_data$address_similarity_jaccard)
summary(all_data$address_similarity_sound)


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

  return(dtm);
}



categoriesDTM = getCategoriesDTM(all_data$features)
freqterms = findFreqTerms(categoriesDTM, 10,Inf) # min 5 freq terms
categoriesDTM = as.data.frame(as.matrix(categoriesDTM))

rownames(categoriesDTM) <- all_data$listing_id
term_names = colnames(categoriesDTM)


categoriesFreqTerms <- categoriesDTM[, freqterms]

all_data <- cbind(all_data, categoriesFreqTerms)



category.features <- c("display_address", "manager_id", "building_id", "street_adress")
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
# 301 features
summary(trainingSet[, feature.names])
##################################################################################

# order of columns are matching 
##################################################################################

# Check for all columns order, all columns orders are same after removing id columns


cv          = 5
bags        = 5
nround.cv   = 600 
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
  
  
  if(i == 1)
  {
    write.csv(val_predictions,  './submissions/prav.xgb02.fold1.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb02.fold1-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 2)
  {
    write.csv(val_predictions,  './submissions/prav.xgb02.fold2.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb02.fold2-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 3)
  {
    write.csv(val_predictions,  './submissions/prav.xgb02.fold3.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb02.fold3-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 4)
  {
    write.csv(val_predictions,  './submissions/prav.xgb02.fold4.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb02.fold4-test.csv', row.names=FALSE, quote = FALSE)
  }
  if(i == 5)
  {
    write.csv(val_predictions,  './submissions/prav.xgb02.fold5.csv', row.names=FALSE, quote = FALSE)
    write.csv(test_predictions, './submissions/prav.xgb02.fold5-test.csv', row.names=FALSE, quote = FALSE)
  }
  
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

write.csv(testfull_predictions, './submissions/prav.xgb02.full.csv', row.names=FALSE, quote = FALSE)




# #########################################################################################################
# Bagging - Full train
# #########################################################################################################

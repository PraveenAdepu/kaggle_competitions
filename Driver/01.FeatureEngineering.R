

source.data <- read_csv("./input/EXPORTED_DATA.csv")

source("./models/helper_functions.R")
head(source.data,5)


source.data$`Dosage Code` <- NULL
source.data$`Dosage Description` <- NULL


#######################################################################################################################
# NLP testing
#######################################################################################################################

head(source.data)

source.data <- source.data %>%
                    mutate(  BreakFast = `BreakFast Tray1` + `BreakFast Tray2`
                           , Lunch = `Lunch Tray1` + `Lunch Tray2`
                           , Dinner = `Dinner Tray1` + `Dinner Tray 2`
                           , BedTime = `Bed time Tray1` + `Bed time Tray2`
                           
                           ) 

source.data <- source.data %>%
                  select (`Script Directions`,BreakFast, Lunch, Dinner, BedTime) %>%
                  mutate(ScriptQty = BreakFast+Lunch+Dinner+BedTime)

source.data$BreakFast <- ifelse(source.data$BreakFast > 0 ,1, 0)
source.data$Lunch <- ifelse(source.data$Lunch > 0 ,1, 0)
source.data$Dinner <- ifelse(source.data$Dinner > 0 ,1, 0)
source.data$BedTime <- ifelse(source.data$BedTime > 0 ,1, 0)
                  

head(source.data)

library(tm)
myCorpus<-Corpus(VectorSource(source.data$`Script Directions`)) #converts the relevant part of your file into a corpus

myCorpus = tm_map(myCorpus, PlainTextDocument) # an intermediate preprocessing step

myCorpus = tm_map(myCorpus, tolower) # converts all text to lower case

myCorpus = tm_map(myCorpus, removePunctuation) #removes punctuation

myCorpus = tm_map(myCorpus, removeWords, stopwords("english")) #removes common words like "a", "the" etc

myCorpus = tm_map(myCorpus, stemDocument) # removes the last few letters of similar words such as get, getting, gets

dtm = DocumentTermMatrix(myCorpus) #turns the corpus into a document term matrix

notSparse = removeSparseTerms(dtm, 0.99) # extracts frequently occuring words

finalWords=as.data.frame(as.matrix(notSparse)) # most frequent words remain in a dataframe, with one column per word

head(finalWords)                         


train <- cbind(source.data, finalWords)

head(train)


train$ID <- seq.int(nrow(train))

###########################################################################################################
# CV folds creation #######################################################################################
###########################################################################################################


#Input to function
train.CV <- as.data.frame(train[,c("ID")])
names(train.CV) <- "ID"

Create5Folds <- function(train, CVSourceColumn, RandomSample, RandomSeed)
{
  set.seed(RandomSeed)
  if(RandomSample)
  {
    train <- as.data.frame(train[sample(1:nrow(train)), ])
    names(train)[1] <- CVSourceColumn
  }
  names(train)[1] <- CVSourceColumn
  
  folds <- createFolds(train[[CVSourceColumn]], k = 5)
  
  trainingFold01 <- as.data.frame(train[folds$Fold1, ])
  trainingFold01$CVindices <- 1
  
  trainingFold02 <- as.data.frame(train[folds$Fold2, ])
  trainingFold02$CVindices <- 2
  
  trainingFold03 <- as.data.frame(train[folds$Fold3, ])
  trainingFold03$CVindices <- 3
  
  trainingFold04 <- as.data.frame(train[folds$Fold4, ])
  trainingFold04$CVindices <- 4
  
  trainingFold05 <- as.data.frame(train[folds$Fold5, ])
  trainingFold05$CVindices <- 5
  
  names(trainingFold01)[1] <- CVSourceColumn
  names(trainingFold02)[1] <- CVSourceColumn
  names(trainingFold03)[1] <- CVSourceColumn
  names(trainingFold04)[1] <- CVSourceColumn
  names(trainingFold05)[1] <- CVSourceColumn
  
  trainingFolds <- rbind(trainingFold01, trainingFold02 , trainingFold03, trainingFold04, trainingFold05 )
  rm(trainingFold01,trainingFold02,trainingFold03,trainingFold04,trainingFold05); gc()
  
  return(trainingFolds)
}


###########################################################################################################
# CV folds creation #######################################################################################
###########################################################################################################

Prav_CVindices <- Create5Folds(train.CV, "ID", RandomSample=TRUE, RandomSeed=2017)

train <- left_join(train, Prav_CVindices, by = "ID")

rm(train.CV, Prav_CVindices); gc()









  
  



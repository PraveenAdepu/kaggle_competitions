
require(gbm)
require(readr)
require(xgboost)
require(car)
require(dplyr)
require(caret)
require(randomForest)
require(stringdist)
require(sqldf)
require(RecordLinkage)
require(e1071)
require(tm)
require(RTextTools)
require(SnowballC) 
require(parallel)  
require(tau)       
require(stringr)
require(data.table)
require(stringi)
require(geosphere)
require(plyr)
require(data.table)
require(splitstackshape)
require(ngram)
#rm(list=ls())

# In progress, start from this script on next attemp
setwd("C:/Users/padepu/Documents/R/09Avito")
getwd()
Sys.getlocale(category = "LC_ALL")
#> Sys.getlocale(category = "LC_ALL")
#[1] "LC_COLLATE=English_Australia.1252;LC_CTYPE=English_Australia.1252;LC_MONETARY=English_Australia.1252;LC_NUMERIC=C;LC_TIME=English_Australia.1252"
Sys.setlocale('LC_ALL', 'russian');
#Sys.setlocale('LC_ALL', 'English_Australia');
#[1] "LC_COLLATE=Russian_Russia.1251;LC_CTYPE=Russian_Russia.1251;LC_MONETARY=Russian_Russia.1251;LC_NUMERIC=C;LC_TIME=Russian_Russia.1251"
cat("Reading data\n")
trainItemInfo     <- read_csv("./ItemInfo_train.csv")
testItemInfo      <- read_csv("./ItemInfo_test.csv")
trainItemPairs    <- read_csv("./ItemPairs_train.csv")
testItemPairs     <- read_csv("./ItemPairs_test.csv")
location          <- read_csv("./Location.csv")

trainItemPairs$rowNum <- 1:nrow(trainItemPairs)
testItemPairs$rowNum  <- 1:nrow(testItemPairs)

trainItemPairs    <- data.table(trainItemPairs)
testItemPairs     <- data.table(testItemPairs)
location          <- data.table(location)


image_hash00          <- read_csv("./image_hash_0_0.csv")
image_hash01          <- read_csv("./image_hash_1_1.csv")
image_hash02          <- read_csv("./image_hash_2_2.csv")
image_hash03          <- read_csv("./image_hash_3_3.csv")
image_hash04          <- read_csv("./image_hash_4_4.csv")
image_hash05          <- read_csv("./image_hash_5_5.csv")
image_hash06          <- read_csv("./image_hash_6_6.csv")
image_hash07          <- read_csv("./image_hash_7_7.csv")
image_hash08          <- read_csv("./image_hash_8_8.csv")
image_hash09          <- read_csv("./image_hash_9_9.csv")


image_hash0001 <- rbind(image_hash00,image_hash01)
image_hash0001 <- rbind(image_hash0001,image_hash02)
image_hash0001 <- rbind(image_hash0001,image_hash03)
image_hash0001 <- rbind(image_hash0001,image_hash04)
image_hash0001 <- rbind(image_hash0001,image_hash05)
image_hash0001 <- rbind(image_hash0001,image_hash06)
image_hash0001 <- rbind(image_hash0001,image_hash07)
image_hash0001 <- rbind(image_hash0001,image_hash08)
image_hash0001 <- rbind(image_hash0001,image_hash09)


image_hash0001 <- select(image_hash0001, image_id,image_hash, width, height, image_file_size)

image_hash0001$imageKey <- as.integer(sapply(strsplit(image_hash0001$image_id,'/'), "[", 3))


image_hash0001  <- data.table(image_hash0001)
setkey(image_hash0001,imageKey)

image_hash00          <- NULL
image_hash01          <- NULL
image_hash02          <- NULL
image_hash03          <- NULL
image_hash04          <- NULL
image_hash05          <- NULL
image_hash06          <- NULL
image_hash07          <- NULL
image_hash08          <- NULL
image_hash09          <- NULL

image_info00          <- read_csv("./image_info_0_0.csv")
image_info01          <- read_csv("./image_info_1_1.csv")
image_info02          <- read_csv("./image_info_2_2.csv")
image_info03          <- read_csv("./image_info_3_3.csv")
image_info04          <- read_csv("./image_info_4_4.csv")
image_info05          <- read_csv("./image_info_5_5.csv")
image_info06          <- read_csv("./image_info_6_6.csv")
image_info07          <- read_csv("./image_info_7_7.csv")
image_info08          <- read_csv("./image_info_8_8.csv")
image_info09          <- read_csv("./image_info_9_9.csv")



image_info00 <- filter(image_info00, image_size != 0 )

image_info0001 <- rbind(image_info00,image_info01)
image_info0001 <- rbind(image_info0001,image_info02)
image_info0001 <- rbind(image_info0001,image_info03)
image_info0001 <- rbind(image_info0001,image_info04)
image_info0001 <- rbind(image_info0001,image_info05)
image_info0001 <- rbind(image_info0001,image_info06)
image_info0001 <- rbind(image_info0001,image_info07)
image_info0001 <- rbind(image_info0001,image_info08)
image_info0001 <- rbind(image_info0001,image_info09)


image_info0001 <- select(image_info0001, image,image_compressed_size,  image_size)
image_info0001 <- filter(image_info0001, image_size != 0 )

image_info0001$image <- gsub('.jpg','',image_info0001$image)
image_info0001$imageKey <- as.integer(sapply(strsplit(image_info0001$image,'/'), "[", 3))

image_info0001  <- data.table(image_info0001)
setkey(image_info0001,imageKey)

image_info00          <- NULL
image_info01          <- NULL
image_info02          <- NULL
image_info03          <- NULL
image_info04          <- NULL
image_info05          <- NULL
image_info06          <- NULL
image_info07          <- NULL
image_info08          <- NULL
image_info09          <- NULL

image_hash0001 <- join(image_hash0001, image_info0001, type="left")

image_hash0001$image_file_size <- NULL
image_hash0001$image <- NULL

####################################################################################################
#######################################################
# Sys.time()
# save.image(file = "Avito.RData" , safe = TRUE)
# Sys.time()
# load("Avito.RData")
# Sys.time()
########################################################

####################################################################################################

names(trainItemInfo)
names(trainItemPairs)
trainItemInfo <- trainItemInfo[,c("itemID","categoryID","locationID","lat","lon","title","description","price","images_array","metroID","attrsJSON")]

names(trainItemInfo) <- c("itemID_1","categoryID1","locationID1","lat1","lon1","title1","description1","price1","images_array1","metroID1","attrsJSON1")

trainItemInfo <- data.table(trainItemInfo)

setkey(trainItemInfo,"itemID_1")
setkey(trainItemPairs,"itemID_1")
#setkey(location, locationID)
names(location)

names(trainItemPairs)
trainItemPairs <- merge(trainItemPairs,trainItemInfo,all.x=TRUE)

names(trainItemInfo) <- c("itemID_2","categoryID2","locationID2","lat2","lon2","title2","description2","price2","images_array2","metroID2","attrsJSON2")

setkey(trainItemInfo,"itemID_2")
setkey(trainItemPairs,"itemID_2")

trainItemPairs <- merge(trainItemPairs,trainItemInfo,all.x=TRUE)

names(location) <- c("locationID1","regionID1")
setkey(trainItemPairs,"locationID1")
setkey(location,"locationID1")
trainItemPairs <- merge(trainItemPairs,location,all.x=TRUE)

names(location) <- c("locationID2","regionID2")
setkey(trainItemPairs,"locationID2")
setkey(location,"locationID2")
trainItemPairs <- merge(trainItemPairs,location,all.x=TRUE)

names(trainItemPairs)

######################################################################################################

testItemInfo <- testItemInfo[,c("itemID","categoryID","locationID","lat","lon","title","description","price","images_array","metroID","attrsJSON")]

names(testItemInfo) <- c("itemID_1","categoryID1","locationID1","lat1","lon1","title1","description1","price1","images_array1","metroID1","attrsJSON1")

testItemInfo <- data.table(testItemInfo)

setkey(testItemInfo,"itemID_1")
setkey(testItemPairs,"itemID_1")

testItemPairs <- merge(testItemPairs,testItemInfo,all.x=TRUE)

names(testItemInfo) <- c("itemID_2","categoryID2","locationID2","lat2","lon2","title2","description2","price2","images_array2","metroID2","attrsJSON2")

setkey(testItemInfo,"itemID_2")
setkey(testItemPairs,"itemID_2")

testItemPairs <- merge(testItemPairs,testItemInfo,all.x=TRUE)

names(location) <- c("locationID1","regionID1")
setkey(testItemPairs,"locationID1")
setkey(location,"locationID1")
testItemPairs <- merge(testItemPairs,location,all.x=TRUE)

names(location) <- c("locationID2","regionID2")
setkey(testItemPairs,"locationID2")
setkey(location,"locationID2")
testItemPairs <- merge(testItemPairs,location,all.x=TRUE)
##########################################################################################################

trainImages <- select(trainItemPairs, rowNum,images_array1,images_array2)

FilteredtrainImages <- trainImages[!is.na(trainImages$images_array1) |!is.na(trainImages$images_array2) ,]

FilteredtrainImages1 <- select(FilteredtrainImages, rowNum,images_array1)
FilteredtrainImages2 <- select(FilteredtrainImages, rowNum,images_array2)

FilteredtrainImages1 <- cSplit(FilteredtrainImages1, "images_array1", ",", direction = "long")
FilteredtrainImages2 <- cSplit(FilteredtrainImages2, "images_array2", ",", direction = "long")

# Cross apply to get all combinations
trainImageCombs <- merge(FilteredtrainImages1 ,FilteredtrainImages2, by = "rowNum"  , all = TRUE ,allow.cartesian=TRUE)
trainImageCombs <- data.table(trainImageCombs)

names(image_hash0001)[names(image_hash0001) == "imageKey"] <- "images_array1"
trainImageCombsHashs<-join(trainImageCombs, image_hash0001, type="left")
names(trainImageCombsHashs)[names(trainImageCombsHashs) == "image_hash"] <- "images_array1Hash"
names(trainImageCombsHashs)[names(trainImageCombsHashs) == "image_id"]   <- "images_array1id"
names(trainImageCombsHashs)[names(trainImageCombsHashs) == "width"]      <- "images_array1width"
names(trainImageCombsHashs)[names(trainImageCombsHashs) == "height"]     <- "images_array1height"
names(trainImageCombsHashs)[names(trainImageCombsHashs) == "image_compressed_size"] <- "images_array1compressed_size"
names(trainImageCombsHashs)[names(trainImageCombsHashs) == "image_size"] <- "images_array1size"

trainImageCombsHashs$images_array1id <- NULL

names(image_hash0001)[names(image_hash0001) == "images_array1"] <- "images_array2"
trainImageCombsHashs<-join(trainImageCombsHashs, image_hash0001, type="left")
names(trainImageCombsHashs)[names(trainImageCombsHashs) == "image_hash"] <- "images_array2Hash"
names(trainImageCombsHashs)[names(trainImageCombsHashs) == "image_id"]   <- "images_array2id"
names(trainImageCombsHashs)[names(trainImageCombsHashs) == "width"]      <- "images_array2width"
names(trainImageCombsHashs)[names(trainImageCombsHashs) == "height"]     <- "images_array2height"
names(trainImageCombsHashs)[names(trainImageCombsHashs) == "image_compressed_size"] <- "images_array2compressed_size"
names(trainImageCombsHashs)[names(trainImageCombsHashs) == "image_size"] <- "images_array2size"

trainImageCombsHashs$images_array2id <- NULL
names(image_hash0001)[names(image_hash0001) == "images_array2"] <- "imageKey"

trainImageCombsHashs$hammingdistance <-  stringdist(trainImageCombsHashs$images_array1Hash, trainImageCombsHashs$images_array2Hash, method = c("hamming"))

trainImageCombsHashs$images_array1and2samewidth  <- ifelse(trainImageCombsHashs$images_array1width == trainImageCombsHashs$images_array2width , 1, 0)
trainImageCombsHashs$images_array1and2sameheight <- ifelse(trainImageCombsHashs$images_array1height == trainImageCombsHashs$images_array2height , 1, 0)

trainImageCombsHashs$images_array1and2Diffwidth  <- trainImageCombsHashs$images_array1width - trainImageCombsHashs$images_array2width 
trainImageCombsHashs$images_array1and2Diffheight <- trainImageCombsHashs$images_array1height - trainImageCombsHashs$images_array2height

trainImageCombsHashs$images_array1Entropy        <- trainImageCombsHashs$images_array1compressed_size /( trainImageCombsHashs$images_array1width * trainImageCombsHashs$images_array1height )
trainImageCombsHashs$images_array2Entropy        <- trainImageCombsHashs$images_array2compressed_size /( trainImageCombsHashs$images_array2width * trainImageCombsHashs$images_array2height )

trainImageCombsHashs$images_array1and2sameEntropy  <- ifelse(trainImageCombsHashs$images_array1Entropy == trainImageCombsHashs$images_array2Entropy , 1, 0)
trainImageCombsHashs$images_array1and2DiffEntropy  <- trainImageCombsHashs$images_array1Entropy - trainImageCombsHashs$images_array2Entropy

trainImageCombsHashs$images_array1and2samecompressedsize  <- ifelse(trainImageCombsHashs$images_array1compressed_size == trainImageCombsHashs$images_array2compressed_size , 1, 0)
trainImageCombsHashs$images_array1and2Diffcompressedsize  <- trainImageCombsHashs$images_array1compressed_size - trainImageCombsHashs$images_array2compressed_size 


trainImagesHashs <- select(trainImageCombsHashs, rowNum,hammingdistance,images_array1and2samewidth ,images_array1and2sameheight,images_array1and2Diffwidth,images_array1and2Diffheight,images_array1and2sameEntropy,images_array1and2DiffEntropy ,images_array1and2samecompressedsize,images_array1and2Diffcompressedsize
                           ,images_array1Entropy,images_array2Entropy, images_array1width, images_array2width, images_array1height,images_array2height,images_array1compressed_size,images_array2compressed_size, images_array1size, images_array2size)

trainRowHashs <- sqldf("SELECT rowNum, COUNT(*) as [ImageCombinations]
                       , Min(hammingdistance) MinHammingDistance
                       , Max(hammingdistance) MaxHammingDistance
                       , (Max(hammingdistance) - Min(hammingdistance) ) as [MaxMinDifference]
                       , stdev(hammingdistance) stdHammingDistance
                       , Avg(hammingdistance) AvgHammingDistance 
                       , sum(CASE WHEN hammingdistance <= 10 then 1 else 0 end ) [CountBelow10distance]
                       , sum(CASE WHEN hammingdistance BETWEEN 11 AND 20 then 1 else 0 end ) [CountBetween11and20distance]
                       , sum(CASE WHEN hammingdistance BETWEEN 21 AND 30 then 1 else 0 end ) [CountBetween21and30distance]
                       , sum(CASE WHEN hammingdistance BETWEEN 31 AND 40 then 1 else 0 end ) [CountBetween31and40distance]                       
                       , sum(CASE WHEN hammingdistance BETWEEN 41 AND 50 then 1 else 0 end ) [CountBetween41and50distance]
                       , sum(CASE WHEN hammingdistance > 50  then 1 else 0 end ) [CountAbove50distance]
                       , sum(images_array1and2samewidth) SameWidthCount
                       , Max(images_array1and2Diffwidth) MaxWidthDiff
                       , Min(images_array1and2Diffwidth) MinWidthDiff
                       , Avg(images_array1and2Diffwidth) AvgWidthDiff
                       , sum(images_array1and2sameheight) SameHeightCount
                       , Max(images_array1and2Diffheight) MaxHeightDiff
                       , Min(images_array1and2Diffheight) MinHeightDiff
                       , Avg(images_array1and2Diffheight) AvgHeightDiff
                       , sum(images_array1and2sameEntropy) SameEntropyCount
                       , Max(images_array1and2DiffEntropy) MaxEntropyDiff
                       , Min(images_array1and2DiffEntropy) MinEntropyDiff
                       , Avg(images_array1and2DiffEntropy) AvgEntropyDiff
                       , sum(images_array1and2samecompressedsize) SamecompressedsizeCount
                       , Max(images_array1and2Diffcompressedsize) MaxcompressedsizeDiff
                       , Min(images_array1and2Diffcompressedsize) MincompressedsizeDiff
                       , Avg(images_array1and2Diffcompressedsize) AvgcompressedsizeDiff
                       , CASE WHEN Avg(images_array2Entropy) = 0 THEN 0 ELSE (Avg(images_array1Entropy)/ Avg(images_array2Entropy)) END [AvgEntropyarray1and2Ratio]
                       , CASE WHEN Avg(images_array2width) = 0 THEN 0 ELSE (Avg(images_array1width)/ Avg(images_array2width)) END [AvgWidthyarray1and2Ratio]
                       , CASE WHEN Avg(images_array2height) = 0 THEN 0 ELSE (Avg(images_array1height)/ Avg(images_array2height)) END [AvgHeightarray1and2Ratio]
                       , CASE WHEN Avg(images_array2compressed_size) = 0 THEN 0 ELSE (Avg(images_array1compressed_size)/ Avg(images_array2compressed_size)) END [Avgcompressed_sizearray1and2Ratio]
                       , CASE WHEN Avg(images_array2size) = 0 THEN 0 ELSE (Avg(images_array1size)/ Avg(images_array2size)) END [Avgsizearray1and2Ratio]
                       , CASE WHEN (Avg(images_array2width) * Avg(images_array2height)) = 0 THEN 0 ELSE (Avg(images_array2compressed_size)/ (Avg(images_array2width) * Avg(images_array2height))) END AvgEntropy_array2Images
                       , CASE WHEN (Avg(images_array1width) * Avg(images_array1height)) = 0 THEN 0 ELSE (Avg(images_array1compressed_size)/ (Avg(images_array1width) * Avg(images_array1height))) END AvgEntropy_array1Images
                       , CASE WHEN Avg(images_array2width) = 0 THEN 0 ELSE Avg(images_array2height)/Avg(images_array2width) END AvgWidthtoHeight_array2Ratio
                       , CASE WHEN Avg(images_array1width) = 0 THEN 0 ELSE Avg(images_array1height)/Avg(images_array1width) END AvgWidthtoHeight_array1Ratio 
                       , CASE WHEN sum(images_array2width) = 0 THEN 0 ELSE sum(images_array2height)/sum(images_array2width) END SumWidthtoHeight_array2Ratio
                       , CASE WHEN sum(images_array1width) = 0 THEN 0 ELSE sum(images_array1height)/sum(images_array1width) END SumWidthtoHeight_array1Ratio 
                       , CASE WHEN Max(images_array2width) = 0 THEN 0 ELSE Max(images_array1width)/Max(images_array2width)  END MaxWidth_array1_array2Ratio
                       , CASE WHEN Max(images_array1width) = 0 THEN 0 ELSE Max(images_array2width)/Max(images_array1width)  END MaxWidth_array2_array1Ratio
                       , CASE WHEN Max(images_array2height) = 0 THEN 0 ELSE Max(images_array1height)/Max(images_array2height)  END MaxHeight_array1_array2Ratio
                       , CASE WHEN Max(images_array1height) = 0 THEN 0 ELSE Max(images_array2height)/Max(images_array1height)  END MaxHeight_array2_array1Ratio
                       , CASE WHEN (Avg(images_array2width) * Avg(images_array2height)) = 0 THEN 0 ELSE (Avg(images_array1compressed_size)/ (Avg(images_array2width) * Avg(images_array2height))) END AvgEntropy_array1_array2Images
                       , CASE WHEN (Avg(images_array1width) * Avg(images_array1height)) = 0 THEN 0 ELSE (Avg(images_array2compressed_size)/ (Avg(images_array1width) * Avg(images_array1height))) END AvgEntropy_array2_array1Images
                       , CASE WHEN (sum(images_array2width) * Sum(images_array2height)) = 0 THEN 0 ELSE (sum(images_array1compressed_size)/ (sum(images_array2width) * sum(images_array2height))) END SumEntropy_array1_array2Images
                       , CASE WHEN (sum(images_array1width) * sum(images_array1height)) = 0 THEN 0 ELSE (sum(images_array2compressed_size)/ (sum(images_array1width) * sum(images_array1height))) END SumEntropy_array2_array1Images
                       FROM trainImagesHashs  GROUP BY rowNum")

trainRowHashs$MaxEntropyDiff <- round(trainRowHashs$MaxEntropyDiff,7)
trainRowHashs$MinEntropyDiff <- round(trainRowHashs$MinEntropyDiff,7)
trainRowHashs$AvgEntropyDiff <- round(trainRowHashs$AvgEntropyDiff,7)

#head(trainRowHashs$Avgsizearray1and2Ratio)
############################################
############################################

testImages <- select(testItemPairs, rowNum,images_array1,images_array2)

FilteredtestImages <- testImages[!is.na(testImages$images_array1) |!is.na(testImages$images_array2) ,]

FilteredtestImages1 <- select(FilteredtestImages, rowNum,images_array1)
FilteredtestImages2 <- select(FilteredtestImages, rowNum,images_array2)

FilteredtestImages1 <- cSplit(FilteredtestImages1, "images_array1", ",", direction = "long")
FilteredtestImages2 <- cSplit(FilteredtestImages2, "images_array2", ",", direction = "long")

# Cross apply to get all combinations
testImageCombs <- merge(FilteredtestImages1 ,FilteredtestImages2, by = "rowNum"  , all = TRUE ,allow.cartesian=TRUE)
testImageCombs <- data.table(testImageCombs)

names(image_hash0001)[names(image_hash0001) == "imageKey"] <- "images_array1"
testImageCombsHashs<-join(testImageCombs, image_hash0001, type="left")
names(testImageCombsHashs)[names(testImageCombsHashs) == "image_hash"] <- "images_array1Hash"
names(testImageCombsHashs)[names(testImageCombsHashs) == "image_id"]   <- "images_array1id"
names(testImageCombsHashs)[names(testImageCombsHashs) == "width"]      <- "images_array1width"
names(testImageCombsHashs)[names(testImageCombsHashs) == "height"]     <- "images_array1height"
names(testImageCombsHashs)[names(testImageCombsHashs) == "image_compressed_size"] <- "images_array1compressed_size"
names(testImageCombsHashs)[names(testImageCombsHashs) == "image_size"] <- "images_array1size"

testImageCombsHashs$images_array1id <- NULL

names(image_hash0001)[names(image_hash0001) == "images_array1"] <- "images_array2"
testImageCombsHashs<-join(testImageCombsHashs, image_hash0001, type="left")
names(testImageCombsHashs)[names(testImageCombsHashs) == "image_hash"] <- "images_array2Hash"
names(testImageCombsHashs)[names(testImageCombsHashs) == "image_id"]   <- "images_array2id"
names(testImageCombsHashs)[names(testImageCombsHashs) == "width"]      <- "images_array2width"
names(testImageCombsHashs)[names(testImageCombsHashs) == "height"]     <- "images_array2height"
names(testImageCombsHashs)[names(testImageCombsHashs) == "image_compressed_size"] <- "images_array2compressed_size"
names(testImageCombsHashs)[names(testImageCombsHashs) == "image_size"] <- "images_array2size"

testImageCombsHashs$images_array2id <- NULL
names(image_hash0001)[names(image_hash0001) == "images_array2"] <- "imageKey"

testImageCombsHashs$hammingdistance <-  stringdist(testImageCombsHashs$images_array1Hash, testImageCombsHashs$images_array2Hash, method = c("hamming"))

testImageCombsHashs$images_array1and2samewidth  <- ifelse(testImageCombsHashs$images_array1width == testImageCombsHashs$images_array2width , 1, 0)
testImageCombsHashs$images_array1and2sameheight <- ifelse(testImageCombsHashs$images_array1height == testImageCombsHashs$images_array2height , 1, 0)

testImageCombsHashs$images_array1and2Diffwidth  <- testImageCombsHashs$images_array1width - testImageCombsHashs$images_array2width 
testImageCombsHashs$images_array1and2Diffheight <- testImageCombsHashs$images_array1height - testImageCombsHashs$images_array2height

testImageCombsHashs$images_array1Entropy        <- testImageCombsHashs$images_array1compressed_size /( testImageCombsHashs$images_array1width * testImageCombsHashs$images_array1height )
testImageCombsHashs$images_array2Entropy        <- testImageCombsHashs$images_array2compressed_size /( testImageCombsHashs$images_array2width * testImageCombsHashs$images_array2height )

testImageCombsHashs$images_array1and2sameEntropy  <- ifelse(testImageCombsHashs$images_array1Entropy == testImageCombsHashs$images_array2Entropy , 1, 0)
testImageCombsHashs$images_array1and2DiffEntropy  <- testImageCombsHashs$images_array1Entropy - testImageCombsHashs$images_array2Entropy

testImageCombsHashs$images_array1and2samecompressedsize  <- ifelse(testImageCombsHashs$images_array1compressed_size == testImageCombsHashs$images_array2compressed_size , 1, 0)
testImageCombsHashs$images_array1and2Diffcompressedsize  <- testImageCombsHashs$images_array1compressed_size - testImageCombsHashs$images_array2compressed_size 


testImagesHashs <- select(testImageCombsHashs, rowNum,hammingdistance,images_array1and2samewidth ,images_array1and2sameheight,images_array1and2Diffwidth,images_array1and2Diffheight,images_array1and2sameEntropy,images_array1and2DiffEntropy ,images_array1and2samecompressedsize,images_array1and2Diffcompressedsize
                          ,images_array1Entropy,images_array2Entropy, images_array1width, images_array2width, images_array1height,images_array2height,images_array1compressed_size,images_array2compressed_size, images_array1size, images_array2size)

testRowHashs <- sqldf("SELECT rowNum, COUNT(*) as [ImageCombinations]
                      , Min(hammingdistance) MinHammingDistance
                      , Max(hammingdistance) MaxHammingDistance
                      , (Max(hammingdistance) - Min(hammingdistance) ) as [MaxMinDifference]
                      , stdev(hammingdistance) stdHammingDistance
                      , Avg(hammingdistance) AvgHammingDistance 
                      , sum(CASE WHEN hammingdistance <= 10 then 1 else 0 end ) [CountBelow10distance]
                      , sum(CASE WHEN hammingdistance BETWEEN 11 AND 20 then 1 else 0 end ) [CountBetween11and20distance]
                      , sum(CASE WHEN hammingdistance BETWEEN 21 AND 30 then 1 else 0 end ) [CountBetween21and30distance]
                      , sum(CASE WHEN hammingdistance BETWEEN 31 AND 40 then 1 else 0 end ) [CountBetween31and40distance]                       
                      , sum(CASE WHEN hammingdistance BETWEEN 41 AND 50 then 1 else 0 end ) [CountBetween41and50distance]
                      , sum(CASE WHEN hammingdistance > 50  then 1 else 0 end ) [CountAbove50distance]
                      , sum(images_array1and2samewidth) SameWidthCount
                      , Max(images_array1and2Diffwidth) MaxWidthDiff
                      , Min(images_array1and2Diffwidth) MinWidthDiff
                      , Avg(images_array1and2Diffwidth) AvgWidthDiff
                      , sum(images_array1and2sameheight) SameHeightCount
                      , Max(images_array1and2Diffheight) MaxHeightDiff
                      , Min(images_array1and2Diffheight) MinHeightDiff
                      , Avg(images_array1and2Diffheight) AvgHeightDiff
                      , sum(images_array1and2sameEntropy) SameEntropyCount
                      , Max(images_array1and2DiffEntropy) MaxEntropyDiff
                      , Min(images_array1and2DiffEntropy) MinEntropyDiff
                      , Avg(images_array1and2DiffEntropy) AvgEntropyDiff
                      , sum(images_array1and2samecompressedsize) SamecompressedsizeCount
                      , Max(images_array1and2Diffcompressedsize) MaxcompressedsizeDiff
                      , Min(images_array1and2Diffcompressedsize) MincompressedsizeDiff
                      , Avg(images_array1and2Diffcompressedsize) AvgcompressedsizeDiff
                      , CASE WHEN Avg(images_array2Entropy) = 0 THEN 0 ELSE (Avg(images_array1Entropy)/ Avg(images_array2Entropy)) END [AvgEntropyarray1and2Ratio]
                      , CASE WHEN Avg(images_array2width) = 0 THEN 0 ELSE (Avg(images_array1width)/ Avg(images_array2width)) END [AvgWidthyarray1and2Ratio]
                      , CASE WHEN Avg(images_array2height) = 0 THEN 0 ELSE (Avg(images_array1height)/ Avg(images_array2height)) END [AvgHeightarray1and2Ratio]
                      , CASE WHEN Avg(images_array2compressed_size) = 0 THEN 0 ELSE (Avg(images_array1compressed_size)/ Avg(images_array2compressed_size)) END [Avgcompressed_sizearray1and2Ratio]
                      , CASE WHEN Avg(images_array2size) = 0 THEN 0 ELSE (Avg(images_array1size)/ Avg(images_array2size)) END [Avgsizearray1and2Ratio]
                      , CASE WHEN (Avg(images_array2width) * Avg(images_array2height)) = 0 THEN 0 ELSE (Avg(images_array2compressed_size)/ (Avg(images_array2width) * Avg(images_array2height))) END AvgEntropy_array2Images
                      , CASE WHEN (Avg(images_array1width) * Avg(images_array1height)) = 0 THEN 0 ELSE (Avg(images_array1compressed_size)/ (Avg(images_array1width) * Avg(images_array1height))) END AvgEntropy_array1Images
                      , CASE WHEN Avg(images_array2width) = 0 THEN 0 ELSE Avg(images_array2height)/Avg(images_array2width) END AvgWidthtoHeight_array2Ratio
                      , CASE WHEN Avg(images_array1width) = 0 THEN 0 ELSE Avg(images_array1height)/Avg(images_array1width) END AvgWidthtoHeight_array1Ratio
                      , CASE WHEN sum(images_array2width) = 0 THEN 0 ELSE sum(images_array2height)/sum(images_array2width) END SumWidthtoHeight_array2Ratio
                      , CASE WHEN sum(images_array1width) = 0 THEN 0 ELSE sum(images_array1height)/sum(images_array1width) END SumWidthtoHeight_array1Ratio 
                      , CASE WHEN Max(images_array2width) = 0 THEN 0 ELSE Max(images_array2height)/Max(images_array2width) END MaxWidthtoHeight_array2Ratio
                      , CASE WHEN Min(images_array1width) = 0 THEN 0 ELSE Min(images_array1height)/Min(images_array1width) END MinWidthtoHeight_array1Ratio
                      , CASE WHEN Max(images_array2width) = 0 THEN 0 ELSE Max(images_array1width)/Max(images_array2width)  END MaxWidth_array1_array2Ratio
                      , CASE WHEN Max(images_array1width) = 0 THEN 0 ELSE Max(images_array2width)/Max(images_array1width)  END MaxWidth_array2_array1Ratio
                      , CASE WHEN Max(images_array2height) = 0 THEN 0 ELSE Max(images_array1height)/Max(images_array2height)  END MaxHeight_array1_array2Ratio
                      , CASE WHEN Max(images_array1height) = 0 THEN 0 ELSE Max(images_array2height)/Max(images_array1height)  END MaxHeight_array2_array1Ratio
                      , CASE WHEN (Avg(images_array2width) * Avg(images_array2height)) = 0 THEN 0 ELSE (Avg(images_array1compressed_size)/ (Avg(images_array2width) * Avg(images_array2height))) END AvgEntropy_array1_array2Images
                      , CASE WHEN (Avg(images_array1width) * Avg(images_array1height)) = 0 THEN 0 ELSE (Avg(images_array2compressed_size)/ (Avg(images_array1width) * Avg(images_array1height))) END AvgEntropy_array2_array1Images
                      , CASE WHEN (sum(images_array2width) * Sum(images_array2height)) = 0 THEN 0 ELSE (sum(images_array1compressed_size)/ (sum(images_array2width) * sum(images_array2height))) END SumEntropy_array1_array2Images
                      , CASE WHEN (sum(images_array1width) * sum(images_array1height)) = 0 THEN 0 ELSE (sum(images_array2compressed_size)/ (sum(images_array1width) * sum(images_array1height))) END SumEntropy_array2_array1Images
                      FROM testImagesHashs  GROUP BY rowNum"
)

testRowHashs$MaxEntropyDiff <- round(as.numeric(testRowHashs$MaxEntropyDiff),7)
testRowHashs$MinEntropyDiff <- round(as.numeric(testRowHashs$MinEntropyDiff),7)
testRowHashs$AvgEntropyDiff <- round(as.numeric(testRowHashs$AvgEntropyDiff),7)

testRowHashs$AvgEntropyarray1and2Ratio <- round(as.numeric(testRowHashs$AvgEntropyarray1and2Ratio),7)
testRowHashs$AvgWidthyarray1and2Ratio <- round(as.numeric(testRowHashs$AvgWidthyarray1and2Ratio),7)
testRowHashs$AvgHeightarray1and2Ratio <- round(as.numeric(testRowHashs$AvgHeightarray1and2Ratio),7)
testRowHashs$Avgcompressed_sizearray1and2Ratio <- round(as.numeric(testRowHashs$Avgcompressed_sizearray1and2Ratio),7)
testRowHashs$Avgsizearray1and2Ratio <- round(as.numeric(testRowHashs$Avgsizearray1and2Ratio),7)

testRowHashs$AvgEntropy_array2Images <- round(as.numeric(testRowHashs$AvgEntropy_array2Images),7)
testRowHashs$AvgEntropy_array1Images <- round(as.numeric(testRowHashs$AvgEntropy_array1Images),7)
testRowHashs$AvgWidthtoHeight_array2Ratio <- round(as.numeric(testRowHashs$AvgWidthtoHeight_array2Ratio),7)
testRowHashs$AvgWidthtoHeight_array1Ratio <- round(as.numeric(testRowHashs$AvgWidthtoHeight_array1Ratio),7)
testRowHashs$SumWidthtoHeight_array2Ratio <- round(as.numeric(testRowHashs$SumWidthtoHeight_array2Ratio),7)
testRowHashs$SumWidthtoHeight_array1Ratio <- round(as.numeric(testRowHashs$SumWidthtoHeight_array1Ratio),7)

testRowHashs$MaxWidth_array1_array2Ratio  <- round(as.numeric(testRowHashs$MaxWidth_array1_array2Ratio),7)
testRowHashs$MaxWidth_array2_array1Ratio  <- round(as.numeric(testRowHashs$MaxWidth_array2_array1Ratio),7)
testRowHashs$MaxHeight_array1_array2Ratio <- round(as.numeric(testRowHashs$MaxHeight_array1_array2Ratio),7)
testRowHashs$MaxHeight_array2_array1Ratio <- round(as.numeric(testRowHashs$MaxHeight_array2_array1Ratio),7)

testRowHashs$AvgEntropy_array1_array2Images <- round(as.numeric(testRowHashs$AvgEntropy_array1_array2Images),7)
testRowHashs$AvgEntropy_array2_array1Images <- round(as.numeric(testRowHashs$AvgEntropy_array2_array1Images),7)

testRowHashs$SumEntropy_array1_array2Images <- round(as.numeric(testRowHashs$SumEntropy_array1_array2Images),7)
testRowHashs$SumEntropy_array2_array1Images <- round(as.numeric(testRowHashs$SumEntropy_array2_array1Images),7)

#head(testRowHashs$AvgEntropy_array1Images)
##################################################################################


trainItemPairs$immages_array1_count <- stri_count_fixed(trainItemPairs$images_array1, ",") + 1
trainItemPairs$immages_array2_count <- stri_count_fixed(trainItemPairs$images_array2, ",") + 1

testItemPairs$immages_array1_count <- stri_count_fixed(testItemPairs$images_array1, ",") + 1
testItemPairs$immages_array2_count <- stri_count_fixed(testItemPairs$images_array2, ",") + 1

trainItemPairs$immages_array1_count[is.na(trainItemPairs$immages_array1_count)] <- 0
trainItemPairs$immages_array2_count[is.na(trainItemPairs$immages_array2_count)] <- 0

testItemPairs$immages_array1_count[is.na(testItemPairs$immages_array1_count)] <- 0
testItemPairs$immages_array2_count[is.na(testItemPairs$immages_array2_count)] <- 0


##########################################################################################################

trainItemPairs$sameLat          <- as.numeric(trainItemPairs$lat1 == trainItemPairs$lat2)
trainItemPairs$sameLon          <- as.numeric(trainItemPairs$lon1 == trainItemPairs$lon2)
trainItemPairs$sameLocation     <- as.numeric(trainItemPairs$locationID1 == trainItemPairs$locationID2)
trainItemPairs$sameregion       <- as.numeric(trainItemPairs$regionID1 == trainItemPairs$regionID2)
trainItemPairs$priceDifference        <- (trainItemPairs$price1 - trainItemPairs$price2)
trainItemPairs$sameTitle        <- as.numeric(trainItemPairs$title1 == trainItemPairs$title2)
trainItemPairs$sameDescription  <- as.numeric(trainItemPairs$description1 == trainItemPairs$description2)
trainItemPairs$imageArrayDiff   <- as.numeric(trainItemPairs$immages_array1_count - trainItemPairs$immages_array2_count)

trainItemPairs$SimilarityTitle        <- jarowinkler(trainItemPairs$title1,trainItemPairs$title2)
trainItemPairs$SimilarityDescription  <- jarowinkler(trainItemPairs$description1,trainItemPairs$description2)

trainItemPairs$samemetro  <- as.numeric(trainItemPairs$metroID1 == trainItemPairs$metroID2)
trainItemPairs$SimilarityJSON        <- jarowinkler(trainItemPairs$attrsJSON1,trainItemPairs$attrsJSON2)
trainItemPairs$distance = sqrt((trainItemPairs$lat1-trainItemPairs$lat2)^2+(trainItemPairs$lon1-trainItemPairs$lon2)^2)

trainItemPairs$nchartitle1          <- nchar(trainItemPairs$title1)
trainItemPairs$nchartitle2          <- nchar(trainItemPairs$title2)
trainItemPairs$nchardescription1    <- nchar(trainItemPairs$description1)
trainItemPairs$nchardescription2    <- nchar(trainItemPairs$description2)
trainItemPairs$priceDiff            <- pmax(trainItemPairs$price1/trainItemPairs$price2, trainItemPairs$price2/trainItemPairs$price1)

trainItemPairs$priceMin             <- pmin(trainItemPairs$price1, trainItemPairs$price2, na.rm=TRUE)
trainItemPairs$priceMax             <- pmax(trainItemPairs$price1, trainItemPairs$price2, na.rm=TRUE)
trainItemPairs$titleStringDistance2 <- (stringdist(trainItemPairs$title1, trainItemPairs$title2, method = "lcs") / 
                                          pmax(trainItemPairs$nchartitle1, trainItemPairs$nchartitle2, na.rm=TRUE))

trainItemPairs$title1StartsWithTitle2 <- as.numeric(substr(trainItemPairs$title1, 1, nchar(trainItemPairs$title2)) == trainItemPairs$title2)
trainItemPairs$title2StartsWithTitle1 <- as.numeric(substr(trainItemPairs$title2, 1, nchar(trainItemPairs$title1)) == trainItemPairs$title1)

trainItemPairs$titleCharDiff      <- pmax(trainItemPairs$nchartitle1/ trainItemPairs$nchartitle2, trainItemPairs$nchartitle2/trainItemPairs$nchartitle1)
trainItemPairs$titleCharMin       <- pmin(trainItemPairs$nchartitle1, trainItemPairs$nchartitle2, na.rm=TRUE)
trainItemPairs$titleCharMax       <- pmax(trainItemPairs$nchartitle1, trainItemPairs$nchartitle2, na.rm=TRUE)

trainItemPairs$descriptionCharDiff <- pmax(trainItemPairs$nchardescription1/ trainItemPairs$nchardescription2, trainItemPairs$nchardescription2/trainItemPairs$nchardescription1)
trainItemPairs$descriptionCharMin  <- pmin(trainItemPairs$nchardescription1, trainItemPairs$nchardescription2, na.rm=TRUE)
trainItemPairs$descriptionCharMax  <- pmax(trainItemPairs$nchardescription1, trainItemPairs$nchardescription2, na.rm=TRUE)

trainItemPairs$title1EndsWithTitle2     <- ifelse(stri_extract_last_words(trainItemPairs$title1) == stri_extract_last_words(trainItemPairs$title2), 1, 0 )
trainItemPairs$title1StartingWithTitle2 <- ifelse(stri_extract_first_words(trainItemPairs$title1) == stri_extract_first_words(trainItemPairs$title2), 1, 0 )

trainItemPairs$title2EndsWithTitle1     <- ifelse(stri_extract_last_words(trainItemPairs$title2) == stri_extract_last_words(trainItemPairs$title1), 1, 0 )
trainItemPairs$title2StartingWithTitle1 <- ifelse(stri_extract_first_words(trainItemPairs$title2) == stri_extract_first_words(trainItemPairs$title1), 1, 0 )


trainItemPairs$title1SoundexWithTitle2     <- stringdist(trainItemPairs$title1,trainItemPairs$title2,method='soundex')

trainItemPairs$title1EndSoundexWithTitle2     <- stringdist(stri_extract_last_words(trainItemPairs$title1)  , stri_extract_last_words(trainItemPairs$title2),   method='soundex' )
trainItemPairs$title1StartSoundexWithTitle2   <- stringdist(stri_extract_first_words(trainItemPairs$title1) , stri_extract_first_words(trainItemPairs$title2) , method='soundex' )


testItemPairs$sameLat         <- as.numeric(testItemPairs$lat1 == testItemPairs$lat2)
testItemPairs$sameLon         <- as.numeric(testItemPairs$lon1 == testItemPairs$lon2)
testItemPairs$sameLocation    <- as.numeric(testItemPairs$locationID1 == testItemPairs$locationID2)
testItemPairs$sameregion      <- as.numeric(testItemPairs$regionID1 == testItemPairs$regionID2)
testItemPairs$priceDifference       <- (testItemPairs$price1 - testItemPairs$price2)
testItemPairs$sameTitle       <- as.numeric(testItemPairs$title1 == testItemPairs$title2)
testItemPairs$sameDescription <- as.numeric(testItemPairs$description1 == testItemPairs$description2)
testItemPairs$imageArrayDiff   <- as.numeric(testItemPairs$immages_array1_count - testItemPairs$immages_array2_count)

testItemPairs$SimilarityTitle        <- jarowinkler(testItemPairs$title1,testItemPairs$title2)
testItemPairs$SimilarityDescription  <- jarowinkler(testItemPairs$description1,testItemPairs$description2)

testItemPairs$samemetro  <- as.numeric(testItemPairs$metroID1 == testItemPairs$metroID2)
testItemPairs$SimilarityJSON        <- jarowinkler(testItemPairs$attrsJSON1,testItemPairs$attrsJSON2)
testItemPairs$distance = sqrt((testItemPairs$lat1-testItemPairs$lat2)^2+(testItemPairs$lon1-testItemPairs$lon2)^2)

testItemPairs$nchartitle1          <- nchar(testItemPairs$title1)
testItemPairs$nchartitle2          <- nchar(testItemPairs$title2)
testItemPairs$nchardescription1    <- nchar(testItemPairs$description1)
testItemPairs$nchardescription2    <- nchar(testItemPairs$description2)
testItemPairs$nchartitle1          <- nchar(testItemPairs$title1)
testItemPairs$nchartitle2          <- nchar(testItemPairs$title2)
testItemPairs$nchardescription1    <- nchar(testItemPairs$description1)
testItemPairs$nchardescription2    <- nchar(testItemPairs$description2)
testItemPairs$priceDiff            <- pmax(testItemPairs$price1/testItemPairs$price2, testItemPairs$price2/testItemPairs$price1)
testItemPairs$priceMin             <- pmin(testItemPairs$price1, testItemPairs$price2, na.rm=TRUE)
testItemPairs$priceMax             <- pmax(testItemPairs$price1, testItemPairs$price2, na.rm=TRUE)
testItemPairs$titleStringDistance2 <- (stringdist(testItemPairs$title1, testItemPairs$title2, method = "lcs") / 
                                         pmax(testItemPairs$nchartitle1, testItemPairs$nchartitle2, na.rm=TRUE))

testItemPairs$title1StartsWithTitle2 <- as.numeric(substr(testItemPairs$title1, 1, nchar(testItemPairs$title2)) == testItemPairs$title2)
testItemPairs$title2StartsWithTitle1 <- as.numeric(substr(testItemPairs$title2, 1, nchar(testItemPairs$title1)) == testItemPairs$title1)

testItemPairs$titleCharDiff       <- pmax(testItemPairs$nchartitle1/ testItemPairs$nchartitle2, testItemPairs$nchartitle2/testItemPairs$nchartitle1)
testItemPairs$titleCharMin        <- pmin(testItemPairs$nchartitle1, testItemPairs$nchartitle2, na.rm=TRUE)
testItemPairs$titleCharMax        <- pmax(testItemPairs$nchartitle1, testItemPairs$nchartitle2, na.rm=TRUE)

testItemPairs$descriptionCharDiff <- pmax(testItemPairs$nchardescription1/ testItemPairs$nchardescription2, testItemPairs$nchardescription2/testItemPairs$nchardescription1)
testItemPairs$descriptionCharMin  <- pmin(testItemPairs$nchardescription1, testItemPairs$nchardescription2, na.rm=TRUE)
testItemPairs$descriptionCharMax  <- pmax(testItemPairs$nchardescription1, testItemPairs$nchardescription2, na.rm=TRUE)

testItemPairs$title1EndsWithTitle2     <- ifelse(stri_extract_last_words(testItemPairs$title1) == stri_extract_last_words(testItemPairs$title2), 1, 0 )
testItemPairs$title1StartingWithTitle2 <- ifelse(stri_extract_first_words(testItemPairs$title1) == stri_extract_first_words(testItemPairs$title2), 1, 0 )
testItemPairs$title1SoundexWithTitle2  <- stringdist(testItemPairs$title1,testItemPairs$title2,method='soundex')

testItemPairs$title2EndsWithTitle1     <-  ifelse(stri_extract_last_words(testItemPairs$title2) ==  stri_extract_last_words(testItemPairs$title1), 1, 0 )
testItemPairs$title2StartingWithTitle1 <- ifelse(stri_extract_first_words(testItemPairs$title2) == stri_extract_first_words(testItemPairs$title1), 1, 0 )

testItemPairs$title1EndSoundexWithTitle2     <- stringdist(stri_extract_last_words(testItemPairs$title1)  , stri_extract_last_words(testItemPairs$title2),   method='soundex' )
testItemPairs$title1StartSoundexWithTitle2   <- stringdist(stri_extract_first_words(testItemPairs$title1) , stri_extract_first_words(testItemPairs$title2) , method='soundex' )

trainItemPairs$priceDiff            <- ifelse(is.na(trainItemPairs$priceDiff) | trainItemPairs$priceDiff == Inf, -9999, trainItemPairs$priceDiff)
trainItemPairs$titleStringDistance2 <- ifelse(is.na(trainItemPairs$titleStringDistance2) | trainItemPairs$titleStringDistance2 == Inf, -9999, trainItemPairs$titleStringDistance2)
trainItemPairs$titleCharDiff        <- ifelse(is.na(trainItemPairs$titleCharDiff)| trainItemPairs$titleCharDiff == Inf, -9999, trainItemPairs$titleCharDiff)
trainItemPairs$descriptionCharDiff  <- ifelse(is.na(trainItemPairs$descriptionCharDiff)| trainItemPairs$descriptionCharDiff == Inf, -9999, trainItemPairs$descriptionCharDiff)

testItemPairs$priceDiff            <- ifelse(is.na(testItemPairs$priceDiff) | testItemPairs$priceDiff == Inf, -9999, testItemPairs$priceDiff)
testItemPairs$titleStringDistance2 <- ifelse(is.na(testItemPairs$titleStringDistance2) | testItemPairs$titleStringDistance2 == Inf, -9999, testItemPairs$titleStringDistance2)
testItemPairs$titleCharDiff        <- ifelse(is.na(testItemPairs$titleCharDiff)| testItemPairs$titleCharDiff == Inf, -9999, testItemPairs$titleCharDiff)
testItemPairs$descriptionCharDiff  <- ifelse(is.na(testItemPairs$descriptionCharDiff)| testItemPairs$descriptionCharDiff == Inf, -9999, testItemPairs$descriptionCharDiff)

#######################################################
# Initial Step
# Sys.time()
# save.image(file = "AvitoInitialStep.RData" , safe = TRUE)
# Sys.time()
# load("AvitoInitialStep.RData")
# Sys.time()
########################################################

#stringdist('???????????????????? ???????????????? delphi zexel denso','???????????????????? ???????????????? delphi zexel densoi',method='soundex')

###################################################################################################
# Trying for spell correction but giving null values so not using for now bur requires to re-think for features
# StandardingString <- function(str) {
#   str <- gsub(",", "",str, ignore.case = TRUE) 
#   str <- gsub(".", "",str, ignore.case = TRUE)
#   return (str)
# }

word_match <- function(firsttitle1,secondtitle2){
  n_title      <- 0
  firsttitle1  <- unlist(strsplit(firsttitle1," "))
  ntitle1 <- length(firsttitle1)
  #if(ntitle1 > 0) {
  for(i in 1:length(firsttitle1)){
    
    #pattern <- paste("(^| )",firsttitle1[i],"($| )",sep="")
    pattern     <- firsttitle1[i]
    n_title     <- n_title  + ifelse(grepl(pattern, secondtitle2,fixed = TRUE,ignore.case=TRUE)>= 1, 1,0 )
    
  }
  
  return(c(ntitle1,n_title))
}

word_strength <- function(firsttitle1,secondtitle2){
  n_title      <- 0
  firsttitle1  <- unlist(strsplit(firsttitle1," "))
  ntitle1 <- length(firsttitle1)
  #if(ntitle1 > 0) {
  for(i in 1:length(firsttitle1)){
    
    #pattern <- paste("(^| )",firsttitle1[i],"($| )",sep="")
    pattern     <- firsttitle1[i]
    n_title     <- n_title  + ifelse(jarowinkler(pattern,secondtitle2)>=0.3,  1  , 0 )
    
  }
  
  return(c(ntitle1,n_title))
}

stem_text<- function(text, language = 'ru', mc.cores = 1) {
  # stem each word in a block of text
  stem_string <- function(str, language) {
    str <- tokenize(x = str)
    str <- wordStem(str, language = language)
    str <- paste(str, collapse = "")
    return(str)
  }
  
  # stem each text block in turn
  x <- mclapply(X = text, FUN = stem_string, language, mc.cores = mc.cores)
  
  # return stemed text blocks
  return(unlist(x))
}

trainItemPairs$title1       <- stem_text(trainItemPairs$title1, language = 'ru', mc.cores = 1)
trainItemPairs$title2       <- stem_text(trainItemPairs$title2, language = 'ru', mc.cores = 1)
trainItemPairs$description1 <- stem_text(trainItemPairs$description1, language = 'ru', mc.cores = 1)
trainItemPairs$description2 <- stem_text(trainItemPairs$description2, language = 'ru', mc.cores = 1)

testItemPairs$title1       <- stem_text(testItemPairs$title1, language = 'ru', mc.cores = 1)
testItemPairs$title2       <- stem_text(testItemPairs$title2, language = 'ru', mc.cores = 1)
testItemPairs$description1 <- stem_text(testItemPairs$description1, language = 'ru', mc.cores = 1)
testItemPairs$description2 <- stem_text(testItemPairs$description2, language = 'ru', mc.cores = 1)


train_all_words <- as.data.frame( t(mapply(word_match,trainItemPairs$title1,trainItemPairs$title2)))
trainItemPairs$Title1Length                 <- train_all_words[,1]
trainItemPairs$Title1WordsMatchedinTitle2   <- train_all_words[,2]

trainItemPairs$Title1WordsMatchedinTitle2ratio <- ifelse(trainItemPairs$Title1Length == 0 , 0 , trainItemPairs$Title1WordsMatchedinTitle2/trainItemPairs$Title1Length)

train_all_words <- as.data.frame( t(mapply(word_match,trainItemPairs$title2,trainItemPairs$title1)))
trainItemPairs$Title2Length                 <- train_all_words[,1]
trainItemPairs$Title2WordsMatchedinTitle1   <- train_all_words[,2]
trainItemPairs$Title2WordsMatchedinTitle1ratio <- ifelse(trainItemPairs$Title2Length == 0 , 0 , trainItemPairs$Title2WordsMatchedinTitle1/trainItemPairs$Title2Length)



train_all_words <- as.data.frame( t(mapply(word_match,trainItemPairs$title1,trainItemPairs$description2)))
trainItemPairs$Title1InDescLength           <- train_all_words[,1]
trainItemPairs$Title1WordsMatchedinDesc2    <- train_all_words[,2]
trainItemPairs$Title1WordsMatchedinDesc2ratio <- ifelse(trainItemPairs$Title1InDescLength == 0 , 0 , trainItemPairs$Title1WordsMatchedinDesc2/trainItemPairs$Title1InDescLength)


train_all_words <- as.data.frame( t(mapply(word_match,trainItemPairs$title2,trainItemPairs$description1)))
trainItemPairs$Title2InDescLength           <- train_all_words[,1]
trainItemPairs$Title2WordsMatchedinDesc1    <- train_all_words[,2]
trainItemPairs$Title2WordsMatchedinDesc1ratio <- ifelse(trainItemPairs$Title2InDescLength == 0 , 0 , trainItemPairs$Title2WordsMatchedinDesc1/trainItemPairs$Title2InDescLength)

train_all_words <- as.data.frame( t(mapply(word_match,trainItemPairs$attrsJSON1,trainItemPairs$attrsJSON2)))
trainItemPairs$attrsJSON1Length                      <- train_all_words[,1]
trainItemPairs$attrsJSON1WordsMatchedinattrsJSON2    <- train_all_words[,2]
trainItemPairs$attrsJSON1WordsMatchedinattrsJSON2ratio        <- ifelse(trainItemPairs$attrsJSON1Length == 0 , 0 , trainItemPairs$attrsJSON1WordsMatchedinattrsJSON2/trainItemPairs$attrsJSON1Length)

trainItemPairs$attrsJSON1WordsMatchedinattrsJSON2ratio <- round(trainItemPairs$attrsJSON1WordsMatchedinattrsJSON2ratio,2)



trainItemPairs$Title1WordsMatchedinTitle2ratio <- round(trainItemPairs$Title1WordsMatchedinTitle2ratio,2)
trainItemPairs$Title2WordsMatchedinTitle1ratio <- round(trainItemPairs$Title2WordsMatchedinTitle1ratio,2)

trainItemPairs$Title1WordsMatchedinDesc2ratio  <- round(trainItemPairs$Title1WordsMatchedinDesc2ratio,2)
trainItemPairs$Title2WordsMatchedinDesc1ratio  <- round(trainItemPairs$Title2WordsMatchedinDesc1ratio,2)
# trainItemPairs$Title1WordsToTitle2MatchRatio <- ifelse(trainItemPairs$Title1Length == 0, 0, trainItemPairs$Title1WordsMatchedinTitle2/trainItemPairs$Title1Length)
# 
test_all_words <- as.data.frame(t(mapply(word_match,testItemPairs$title1,testItemPairs$title2)))
testItemPairs$Title1Length                 <- test_all_words[,1]
testItemPairs$Title1WordsMatchedinTitle2   <- test_all_words[,2]
testItemPairs$Title1WordsMatchedinTitle2ratio <- ifelse(testItemPairs$Title1Length == 0 , 0 , testItemPairs$Title1WordsMatchedinTitle2/testItemPairs$Title1Length)

test_all_words <- as.data.frame(t(mapply(word_match,testItemPairs$title2,testItemPairs$title1)))
testItemPairs$Title2Length                 <- test_all_words[,1]
testItemPairs$Title2WordsMatchedinTitle1   <- test_all_words[,2]
testItemPairs$Title2WordsMatchedinTitle1ratio <- ifelse(testItemPairs$Title2Length == 0 , 0 , testItemPairs$Title2WordsMatchedinTitle1/testItemPairs$Title2Length)

test_all_words <- as.data.frame( t(mapply(word_match,testItemPairs$title1,testItemPairs$description2)))
testItemPairs$Title1InDescLength           <- test_all_words[,1]
testItemPairs$Title1WordsMatchedinDesc2    <- test_all_words[,2]
testItemPairs$Title1WordsMatchedinDesc2ratio <- ifelse(testItemPairs$Title1InDescLength == 0 , 0 , testItemPairs$Title1WordsMatchedinDesc2/testItemPairs$Title1InDescLength)


test_all_words <- as.data.frame( t(mapply(word_match,testItemPairs$title2,testItemPairs$description1)))
testItemPairs$Title2InDescLength           <- test_all_words[,1]
testItemPairs$Title2WordsMatchedinDesc1    <- test_all_words[,2]
testItemPairs$Title2WordsMatchedinDesc1ratio <- ifelse(testItemPairs$Title2InDescLength == 0 , 0 , testItemPairs$Title2WordsMatchedinDesc1/testItemPairs$Title2InDescLength)

testItemPairs$Title1WordsMatchedinTitle2ratio <- round(testItemPairs$Title1WordsMatchedinTitle2ratio,2)
testItemPairs$Title2WordsMatchedinTitle1ratio <- round(testItemPairs$Title2WordsMatchedinTitle1ratio,2)

testItemPairs$Title1WordsMatchedinDesc2ratio  <- round(testItemPairs$Title1WordsMatchedinDesc2ratio,2)
testItemPairs$Title2WordsMatchedinDesc1ratio  <- round(testItemPairs$Title2WordsMatchedinDesc1ratio,2)

train_all_words <- as.data.frame( t(mapply(word_match,trainItemPairs$description1,trainItemPairs$description2)))
trainItemPairs$description1Length                 <- train_all_words[,1]
trainItemPairs$description1WordsMatchedinDesc2    <- train_all_words[,2]
trainItemPairs$description1WordsMatchedinDesc2ratio <- ifelse(trainItemPairs$description1Length == 0 , 0 , trainItemPairs$description1WordsMatchedinDesc2/trainItemPairs$description1Length)

trainItemPairs$description1WordsMatchedinDesc2ratio <- round(trainItemPairs$description1WordsMatchedinDesc2ratio , 2)

test_all_words <- as.data.frame( t(mapply(word_match,testItemPairs$description1,testItemPairs$description2)))
testItemPairs$description1Length                 <- test_all_words[,1]
testItemPairs$description1WordsMatchedinDesc2    <- test_all_words[,2]
testItemPairs$description1WordsMatchedinDesc2ratio <- ifelse(testItemPairs$description1Length == 0 , 0 , testItemPairs$description1WordsMatchedinDesc2/testItemPairs$description1Length)

testItemPairs$description1WordsMatchedinDesc2ratio <- round(testItemPairs$description1WordsMatchedinDesc2ratio , 2)

test_all_words <- as.data.frame( t(mapply(word_match,testItemPairs$attrsJSON1,testItemPairs$attrsJSON2)))
testItemPairs$attrsJSON1Length                      <- test_all_words[,1]
testItemPairs$attrsJSON1WordsMatchedinattrsJSON2    <- test_all_words[,2]
testItemPairs$attrsJSON1WordsMatchedinattrsJSON2ratio        <- ifelse(testItemPairs$attrsJSON1Length == 0 , 0 , testItemPairs$attrsJSON1WordsMatchedinattrsJSON2/testItemPairs$attrsJSON1Length)

testItemPairs$attrsJSON1WordsMatchedinattrsJSON2ratio <- round(testItemPairs$attrsJSON1WordsMatchedinattrsJSON2ratio,2)

trainItemPairs$Title1WordsMatchedinTitle2ratioDiff      <- pmax(trainItemPairs$Title1WordsMatchedinTitle2ratio/ trainItemPairs$Title2WordsMatchedinTitle1ratio, trainItemPairs$Title2WordsMatchedinTitle1ratio/trainItemPairs$Title1WordsMatchedinTitle2ratio)
trainItemPairs$Title1WordsMatchedinTitle2ratioMin       <- pmin(trainItemPairs$Title1WordsMatchedinTitle2, trainItemPairs$Title2WordsMatchedinTitle1, na.rm=TRUE)
trainItemPairs$Title1WordsMatchedinTitle2ratioMax       <- pmax(trainItemPairs$Title1WordsMatchedinTitle2, trainItemPairs$Title2WordsMatchedinTitle1, na.rm=TRUE)

trainItemPairs$Title1WordsMatchedinDesc2ratioDiff      <- pmax(trainItemPairs$Title1WordsMatchedinDesc2ratio/ trainItemPairs$Title2WordsMatchedinDesc1ratio, trainItemPairs$Title2WordsMatchedinDesc1ratio/trainItemPairs$Title1WordsMatchedinDesc2ratio)
trainItemPairs$Title1WordsMatchedinDesc2ratioMin       <- pmin(trainItemPairs$Title1WordsMatchedinDesc2, trainItemPairs$Title2WordsMatchedinDesc1, na.rm=TRUE)
trainItemPairs$Title1WordsMatchedinDesc2ratioMax       <- pmax(trainItemPairs$Title1WordsMatchedinDesc2, trainItemPairs$Title2WordsMatchedinDesc1, na.rm=TRUE)

trainItemPairs$Title1WordsMatchedinTitle2Diff      <- pmax(trainItemPairs$Title1WordsMatchedinTitle2/ trainItemPairs$Title2WordsMatchedinTitle1, trainItemPairs$Title2WordsMatchedinTitle1/trainItemPairs$Title1WordsMatchedinTitle2)
trainItemPairs$Title1WordsMatchedinDesc2Diff       <- pmax(trainItemPairs$Title1WordsMatchedinDesc2/ trainItemPairs$Title2WordsMatchedinDesc1, trainItemPairs$Title2WordsMatchedinDesc1/trainItemPairs$Title1WordsMatchedinDesc2)

testItemPairs$Title1WordsMatchedinTitle2Diff      <- pmax(testItemPairs$Title1WordsMatchedinTitle2/ testItemPairs$Title2WordsMatchedinTitle1, testItemPairs$Title2WordsMatchedinTitle1/testItemPairs$Title1WordsMatchedinTitle2)
testItemPairs$Title1WordsMatchedinDesc2Diff       <- pmax(testItemPairs$Title1WordsMatchedinDesc2/  testItemPairs$Title2WordsMatchedinDesc1,  testItemPairs$Title2WordsMatchedinDesc1/ testItemPairs$Title1WordsMatchedinDesc2)


testItemPairs$Title1WordsMatchedinDesc2ratioDiff      <- pmax(testItemPairs$Title1WordsMatchedinDesc2ratio/ testItemPairs$Title2WordsMatchedinDesc1ratio, testItemPairs$Title2WordsMatchedinDesc1ratio/testItemPairs$Title1WordsMatchedinDesc2ratio)
testItemPairs$Title1WordsMatchedinDesc2ratioMin       <- pmin(testItemPairs$Title1WordsMatchedinDesc2, testItemPairs$Title2WordsMatchedinDesc1, na.rm=TRUE)
testItemPairs$Title1WordsMatchedinDesc2ratioMax       <- pmax(testItemPairs$Title1WordsMatchedinDesc2, testItemPairs$Title2WordsMatchedinDesc1, na.rm=TRUE)

testItemPairs$Title1WordsMatchedinTitle2ratioDiff      <- pmax(testItemPairs$Title1WordsMatchedinTitle2ratio/ testItemPairs$Title2WordsMatchedinTitle1ratio, testItemPairs$Title2WordsMatchedinTitle1ratio/testItemPairs$Title1WordsMatchedinTitle2ratio)
testItemPairs$Title1WordsMatchedinTitle2ratioMin       <- pmin(testItemPairs$Title1WordsMatchedinTitle2, testItemPairs$Title2WordsMatchedinTitle1, na.rm=TRUE)
testItemPairs$Title1WordsMatchedinTitle2ratioMax       <- pmax(testItemPairs$Title1WordsMatchedinTitle2, testItemPairs$Title2WordsMatchedinTitle1, na.rm=TRUE)

trainItemPairs$Title1WordsMatchedinTitle2ratioDiff    <- ifelse(is.na(trainItemPairs$Title1WordsMatchedinTitle2ratioDiff) | trainItemPairs$Title1WordsMatchedinTitle2ratioDiff == Inf, -9999, trainItemPairs$Title1WordsMatchedinTitle2ratioDiff)
trainItemPairs$Title1WordsMatchedinDesc2ratioDiff     <- ifelse(is.na(trainItemPairs$Title1WordsMatchedinDesc2ratioDiff) | trainItemPairs$Title1WordsMatchedinDesc2ratioDiff == Inf, -9999, trainItemPairs$Title1WordsMatchedinDesc2ratioDiff)
trainItemPairs$Title1WordsMatchedinTitle2Diff         <- ifelse(is.na(trainItemPairs$Title1WordsMatchedinTitle2Diff) | trainItemPairs$Title1WordsMatchedinTitle2Diff == Inf, -9999, trainItemPairs$Title1WordsMatchedinTitle2Diff)
trainItemPairs$Title1WordsMatchedinDesc2Diff          <- ifelse(is.na(trainItemPairs$Title1WordsMatchedinDesc2Diff) | trainItemPairs$Title1WordsMatchedinDesc2Diff == Inf, -9999, trainItemPairs$Title1WordsMatchedinDesc2Diff)

testItemPairs$Title1WordsMatchedinTitle2ratioDiff     <- ifelse(is.na(testItemPairs$Title1WordsMatchedinTitle2ratioDiff) | testItemPairs$Title1WordsMatchedinTitle2ratioDiff == Inf, -9999, testItemPairs$Title1WordsMatchedinTitle2ratioDiff)
testItemPairs$Title1WordsMatchedinDesc2ratioDiff      <- ifelse(is.na(testItemPairs$Title1WordsMatchedinDesc2ratioDiff) | testItemPairs$Title1WordsMatchedinDesc2ratioDiff == Inf, -9999, testItemPairs$Title1WordsMatchedinDesc2ratioDiff)
testItemPairs$Title1WordsMatchedinTitle2Diff          <- ifelse(is.na(testItemPairs$Title1WordsMatchedinTitle2Diff) | testItemPairs$Title1WordsMatchedinTitle2Diff == Inf, -9999, testItemPairs$Title1WordsMatchedinTitle2Diff)
testItemPairs$Title1WordsMatchedinDesc2Diff           <- ifelse(is.na(testItemPairs$Title1WordsMatchedinDesc2Diff) | testItemPairs$Title1WordsMatchedinDesc2Diff == Inf, -9999, testItemPairs$Title1WordsMatchedinDesc2Diff)


train_all_wordsstrength <- as.data.frame( t(mapply(word_strength,trainItemPairs$title1,trainItemPairs$title2)))
trainItemPairs$Title1LengthStrength             <- train_all_wordsstrength[,1]
trainItemPairs$Title1WordsStrengthinTitle2      <- train_all_wordsstrength[,2]
trainItemPairs$Title1WordsStrengthinTitle2ratio <- ifelse(trainItemPairs$Title1LengthStrength == 0 , 0 , trainItemPairs$Title1WordsStrengthinTitle2/trainItemPairs$Title1LengthStrength)

test_all_wordsstrength <- as.data.frame( t(mapply(word_strength,testItemPairs$title1,testItemPairs$title2)))
testItemPairs$Title1LengthStrength             <- test_all_wordsstrength[,1]
testItemPairs$Title1WordsStrengthinTitle2      <- test_all_wordsstrength[,2]
testItemPairs$Title1WordsStrengthinTitle2ratio <- ifelse(testItemPairs$Title1LengthStrength == 0 , 0 , testItemPairs$Title1WordsStrengthinTitle2/testItemPairs$Title1LengthStrength)
###########################################################################################################
## new features trails

# # str_count(trainItemPairs$description2, trainItemPairs$title1)
# # word(head(trainItemPairs$title1), 1, 2, sep = fixed(' '))   # frist two words
# # word(head(trainItemPairs$title1), -2, -1, sep = fixed(' ')) # last two words
# 
# string_fun <- function(x) {
#   ul = unlist(strsplit(x, split = " "))[1:2]
#   paste(ul,collapse=" ")
# }
# trainItemPairs$title1firsttwowords                 <- unlist(lapply(trainItemPairs$title1, string_fun))
# trainItemPairs$title2firsttwowords                 <- unlist(lapply(trainItemPairs$title2, string_fun))
# trainItemPairs$title1firsttwowordsdistanceintitle2 <- jarowinkler(trainItemPairs$title1firsttwowords,trainItemPairs$title2)
#   
#   #stringdist(iconv(trainItemPairs$title1firsttwowords,to="ASCII//TRANSLIT"),iconv(trainItemPairs$title2firsttwowords,to="ASCII//TRANSLIT"),method='soundex')
# 
#   # ifelse(jarowinkler(trainItemPairs$title1firsttwowords,trainItemPairs$title2)>= 0.2 , 1, 0)
# 
# # iconv(trainItemPairs$title1firsttwowords,to="ASCII//TRANSLIT")
# testItemPairs$title1firsttwowords                 <- unlist(lapply(testItemPairs$title1, string_fun))
# testItemPairs$title2firsttwowords                 <- unlist(lapply(testItemPairs$title2, string_fun))
# testItemPairs$title1firsttwowordsdistanceintitle2 <- jarowinkler(testItemPairs$title1firsttwowords,testItemPairs$title2)
#   #stringdist(iconv(testItemPairs$title1firsttwowords,to="ASCII//TRANSLIT"),iconv(testItemPairs$title2firsttwowords,to="ASCII//TRANSLIT"),method='soundex') 
#   #stringdist(testItemPairs$title1firsttwowords,testItemPairs$title2,method='soundex')
#   #ifelse(jarowinkler(testItemPairs$title1firsttwowords,testItemPairs$title2)>= 0.2 , 1, 0)
# 
# string_fun <- function(x) {
#   ul = unlist(strsplit(x, split = " "))[1:5]
#   paste(ul,collapse=" ")
# }
# trainItemPairs$description1firstfivewords          <- unlist(lapply(trainItemPairs$description1, string_fun))
# #trainItemPairs$description2firstfivewords          <- unlist(lapply(trainItemPairs$description2, string_fun))
# trainItemPairs$desc1firstfivewordsdistanceindesc2  <- jarowinkler(trainItemPairs$description1firstfivewords,trainItemPairs$description2)
# 
# testItemPairs$description1firstfivewords          <- unlist(lapply(testItemPairs$description1, string_fun))
# #testItemPairs$description2firstfivewords          <- unlist(lapply(testItemPairs$description2, string_fun))
# testItemPairs$desc1firstfivewordsdistanceindesc2  <- jarowinkler(testItemPairs$description1firstfivewords,testItemPairs$description2)
# 
# trainItemPairs$TitletoDescwordsratio  <- pmax(trainItemPairs$title1firsttwowordsdistanceintitle2/ trainItemPairs$desc1firstfivewordsdistanceindesc2, trainItemPairs$desc1firstfivewordsdistanceindesc2/trainItemPairs$title1firsttwowordsdistanceintitle2)
# trainItemPairs$TitletoDescwordsratio          <- ifelse(is.na(trainItemPairs$TitletoDescwordsratio) | trainItemPairs$TitletoDescwordsratio == Inf, -9999, trainItemPairs$TitletoDescwordsratio)
# 
# testItemPairs$TitletoDescwordsratio  <- pmax(testItemPairs$title1firsttwowordsdistanceintitle2/ testItemPairs$desc1firstfivewordsdistanceindesc2, testItemPairs$desc1firstfivewordsdistanceindesc2/testItemPairs$title1firsttwowordsdistanceintitle2)
# testItemPairs$TitletoDescwordsratio          <- ifelse(is.na(testItemPairs$TitletoDescwordsratio) | testItemPairs$TitletoDescwordsratio == Inf, -9999, testItemPairs$TitletoDescwordsratio)

trainItemPairs$title12gramtitle2  <- stringdist(trainItemPairs$title1, trainItemPairs$title2, method='jaccard', q=2)
testItemPairs$title12gramtitle2  <- stringdist(testItemPairs$title1, testItemPairs$title2, method='jaccard', q=2)

trainItemPairs$title12gramtitle2 <- ifelse(is.na(trainItemPairs$title12gramtitle2) | trainItemPairs$title12gramtitle2 == Inf, -9999, trainItemPairs$title12gramtitle2)
testItemPairs$title12gramtitle2 <- ifelse(is.na(testItemPairs$title12gramtitle2) | testItemPairs$title12gramtitle2 == Inf, -9999, testItemPairs$title12gramtitle2)

trainItemPairs$title13gramtitle2  <- stringdist(trainItemPairs$title1, trainItemPairs$title2, method='jaccard', q=3)
testItemPairs$title13gramtitle2  <- stringdist(testItemPairs$title1, testItemPairs$title2, method='jaccard', q=3)

trainItemPairs$title13gramtitle2 <- ifelse(is.na(trainItemPairs$title13gramtitle2) | trainItemPairs$title13gramtitle2 == Inf, -9999, trainItemPairs$title13gramtitle2)
testItemPairs$title13gramtitle2 <- ifelse(is.na(testItemPairs$title13gramtitle2) | testItemPairs$title13gramtitle2 == Inf, -9999, testItemPairs$title13gramtitle2)

trainItemPairs$title14gramtitle2  <- stringdist(trainItemPairs$title1, trainItemPairs$title2, method='jaccard', q=4)
testItemPairs$title14gramtitle2  <- stringdist(testItemPairs$title1, testItemPairs$title2, method='jaccard', q=4)

trainItemPairs$title14gramtitle2 <- ifelse(is.na(trainItemPairs$title14gramtitle2) | trainItemPairs$title14gramtitle2 == Inf, -9999, trainItemPairs$title14gramtitle2)
testItemPairs$title14gramtitle2 <- ifelse(is.na(testItemPairs$title14gramtitle2) | testItemPairs$title14gramtitle2 == Inf, -9999, testItemPairs$title14gramtitle2)

trainItemPairs$title15gramtitle2  <- stringdist(trainItemPairs$title1, trainItemPairs$title2, method='jaccard', q=5)
testItemPairs$title15gramtitle2  <- stringdist(testItemPairs$title1, testItemPairs$title2, method='jaccard', q=5)

trainItemPairs$title15gramtitle2 <- ifelse(is.na(trainItemPairs$title15gramtitle2) | trainItemPairs$title15gramtitle2 == Inf, -9999, trainItemPairs$title15gramtitle2)
testItemPairs$title15gramtitle2 <- ifelse(is.na(testItemPairs$title15gramtitle2) | testItemPairs$title15gramtitle2 == Inf, -9999, testItemPairs$title15gramtitle2)


trainItemPairs$title12gramdescription2  <- stringdist(trainItemPairs$title1, trainItemPairs$description2, method='jaccard', q=2)
testItemPairs$title12gramdescription2  <- stringdist(testItemPairs$title1, testItemPairs$description2, method='jaccard', q=2)

trainItemPairs$title12gramdescription2 <- ifelse(is.na(trainItemPairs$title12gramdescription2) | trainItemPairs$title12gramdescription2 == Inf, -9999, trainItemPairs$title12gramdescription2)
testItemPairs$title12gramdescription2 <- ifelse(is.na(testItemPairs$title12gramdescription2) | testItemPairs$title12gramdescription2 == Inf, -9999, testItemPairs$title12gramdescription2)

trainItemPairs$title13gramdescription2  <- stringdist(trainItemPairs$title1, trainItemPairs$description2, method='jaccard', q=3)
testItemPairs$title13gramdescription2  <- stringdist(testItemPairs$title1, testItemPairs$description2, method='jaccard', q=3)

trainItemPairs$title13gramdescription2 <- ifelse(is.na(trainItemPairs$title13gramdescription2) | trainItemPairs$title13gramdescription2 == Inf, -9999, trainItemPairs$title13gramdescription2)
testItemPairs$title13gramdescription2 <- ifelse(is.na(testItemPairs$title13gramdescription2) | testItemPairs$title13gramdescription2 == Inf, -9999, testItemPairs$title13gramdescription2)

trainItemPairs$title14gramdescription2  <- stringdist(trainItemPairs$title1, trainItemPairs$description2, method='jaccard', q=4)
testItemPairs$title14gramdescription2  <- stringdist(testItemPairs$title1, testItemPairs$description2, method='jaccard', q=4)

trainItemPairs$title14gramdescription2 <- ifelse(is.na(trainItemPairs$title14gramdescription2) | trainItemPairs$title14gramdescription2 == Inf, -9999, trainItemPairs$title14gramdescription2)
testItemPairs$title14gramdescription2 <- ifelse(is.na(testItemPairs$title14gramdescription2) | testItemPairs$title14gramdescription2 == Inf, -9999, testItemPairs$title14gramdescription2)

trainItemPairs$title15gramdescription2  <- stringdist(trainItemPairs$title1, trainItemPairs$description2, method='jaccard', q=5)
testItemPairs$title15gramdescription2  <- stringdist(testItemPairs$title1, testItemPairs$description2, method='jaccard', q=5)

trainItemPairs$title15gramdescription2 <- ifelse(is.na(trainItemPairs$title15gramdescription2) | trainItemPairs$title15gramdescription2 == Inf, -9999, trainItemPairs$title15gramdescription2)
testItemPairs$title15gramdescription2 <- ifelse(is.na(testItemPairs$title15gramdescription2) | testItemPairs$title15gramdescription2 == Inf, -9999, testItemPairs$title15gramdescription2)


trainItemPairs$description12gramdescription2  <- stringdist(trainItemPairs$description1, trainItemPairs$description2, method='jaccard', q=2)
testItemPairs$description12gramdescription2  <- stringdist(testItemPairs$description1, testItemPairs$description2, method='jaccard', q=2)
trainItemPairs$description12gramdescription2 <- ifelse(is.na(trainItemPairs$description12gramdescription2) | trainItemPairs$description12gramdescription2 == Inf, -9999, trainItemPairs$description12gramdescription2)
testItemPairs$description12gramdescription2 <- ifelse(is.na(testItemPairs$description12gramdescription2) | testItemPairs$description12gramdescription2 == Inf, -9999, testItemPairs$description12gramdescription2)
####################################################################################################################
# New features in Avito_05
allTwoword_match <- function(firsttitle1,secondtitle2){
  wordmatch         <- 0
  wordscombinations <- 0
  unlistfirsttitle1  <- unlist(strsplit(firsttitle1," "))
  ntitle1 <- length(unlistfirsttitle1)
  #if(ntitle1 > 0) {
  for(i in 2){
    ng1 <- ngram_asweka(firsttitle1, min = i, max = i, sep = " ")
    ng2 <- ngram_asweka(secondtitle2, min = i, max = i, sep = " ")
    wordmatch <- wordmatch + length(which(ng1 %in% ng2))
    
  }
  
  return(c(ntitle1,wordmatch))
}

train_allTwoword_match <- as.data.frame( t(mapply(allTwoword_match,trainItemPairs$title1,trainItemPairs$title2)))

trainItemPairs$Title1TwoWordsMatchedinTitle2   <- train_allTwoword_match[,2]
trainItemPairs$Title1TwoWordsMatchedinTitle2 <- ifelse((trainItemPairs$Title1Length -1) == 0, 0 , trainItemPairs$Title1TwoWordsMatchedinTitle2/(trainItemPairs$Title1Length -1))


train_allTwoword_match <- as.data.frame( t(mapply(allTwoword_match,trainItemPairs$title1,trainItemPairs$description2)))

trainItemPairs$Title1TwoWordsMatchedinDesc2   <- train_allTwoword_match[,2]
trainItemPairs$Title1TwoWordsMatchedinDesc2   <- ifelse((trainItemPairs$Title1Length -1) == 0, 0 , trainItemPairs$Title1TwoWordsMatchedinDesc2/(trainItemPairs$Title1Length -1))

test_allTwoword_match <- as.data.frame( t(mapply(allTwoword_match,testItemPairs$title1,testItemPairs$title2)))

testItemPairs$Title1TwoWordsMatchedinTitle2   <- test_allTwoword_match[,2]
testItemPairs$Title1TwoWordsMatchedinTitle2 <- ifelse((testItemPairs$Title1Length -1) == 0, 0 , testItemPairs$Title1TwoWordsMatchedinTitle2/(testItemPairs$Title1Length -1))


test_allTwoword_match <- as.data.frame( t(mapply(allTwoword_match,testItemPairs$title1,testItemPairs$description2)))

testItemPairs$Title1TwoWordsMatchedinDesc2   <- test_allTwoword_match[,2]
testItemPairs$Title1TwoWordsMatchedinDesc2   <- ifelse((testItemPairs$Title1Length -1) == 0, 0 , testItemPairs$Title1TwoWordsMatchedinDesc2/(testItemPairs$Title1Length -1))


# train_allTwoword_match <- as.data.frame( t(mapply(allTwoword_match,trainItemPairs$title2,trainItemPairs$description1)))
# 
# trainItemPairs$Title2TwoWordsMatchedinDesc1   <- train_allTwoword_match[,2]
# trainItemPairs$Title2TwoWordsMatchedinDesc1   <-  ifelse((trainItemPairs$Title2Length -1) == 0, 0 , trainItemPairs$Title2TwoWordsMatchedinDesc1/(trainItemPairs$Title2Length -1))
# 
# train_allTwoword_match <- as.data.frame( t(mapply(allTwoword_match,trainItemPairs$description1,trainItemPairs$description2)))
# 
# trainItemPairs$Desc1TwoWordsMatchedinDesc2   <- train_allTwoword_match[,2]
# trainItemPairs$Desc1TwoWordsMatchedinDesc2   <- ifelse((trainItemPairs$description1Length -1) == 0, 0 , trainItemPairs$Desc1TwoWordsMatchedinDesc2/(trainItemPairs$description1Length -1))
# 
# 
# trainItemPairs$Title1TwoWordsMatchedinDescratio     <- pmax(trainItemPairs$Title2TwoWordsMatchedinDesc1/ trainItemPairs$Title1TwoWordsMatchedinDesc2, trainItemPairs$Title1TwoWordsMatchedinDesc2/trainItemPairs$Title2TwoWordsMatchedinDesc1)
# trainItemPairs$Title1TwoWordsMatchedinDescratio  <-  ifelse(is.na(trainItemPairs$Title1TwoWordsMatchedinDescratio) | trainItemPairs$Title1TwoWordsMatchedinDescratio == Inf, -9999, trainItemPairs$Title1TwoWordsMatchedinDescratio)

trainItemPairs$titlesCosineDist <-  stringdist(trainItemPairs$title1,trainItemPairs$title2, method = 'lv') #'cosine'
testItemPairs$titlesCosineDist <-  stringdist(testItemPairs$title1,testItemPairs$title2, method = 'lv')
trainItemPairs$titlesCosinesDist <-  stringdist(trainItemPairs$title1,trainItemPairs$title2, method = 'cosine') #'cosine'
testItemPairs$titlesCosinesDist <-  stringdist(testItemPairs$title1,testItemPairs$title2, method = 'cosine')


# trainItemPairs$description15gramdescription2  <- stringdist(trainItemPairs$description1, trainItemPairs$description2, method='jaccard', q=5)
# gc()
# testItemPairs$description15gramdescription2  <- stringdist(testItemPairs$description1, testItemPairs$description2, method='jaccard', q=5)
# gc()
# trainItemPairs$description15gramdescription2 <- ifelse(is.na(trainItemPairs$description15gramdescription2) | trainItemPairs$description15gramdescription2 == Inf, -9999, trainItemPairs$description15gramdescription2)
# testItemPairs$description15gramdescription2 <- ifelse(is.na(testItemPairs$description15gramdescription2) | testItemPairs$description15gramdescription2 == Inf, -9999, testItemPairs$description15gramdescription2)
##########################################################################################################
# New features -- 20160609, from v.07
WordPunctCount <- function(firsttitle1,secondtitle2){
  SpecialCharCount1  <- 0
  SpecialCharCount2  <- 0
  {
    SpecialCharCount1  <- str_count(firsttitle1) - str_count(str_replace_all(firsttitle1, "[[:punct:]]", ""))
    SpecialCharCount2  <- str_count(secondtitle2) - str_count(str_replace_all(secondtitle2, "[[:punct:]]", ""))
  }
  return(c(SpecialCharCount1,SpecialCharCount2 ))
}

train_WordPunct <- as.data.frame( t(mapply(WordPunctCount,trainItemPairs$title1,trainItemPairs$title2)))
trainItemPairs$title1PunctCount <-  train_WordPunct[,1]
trainItemPairs$title2PunctCount <-  train_WordPunct[,2]


test_WordPunct <- as.data.frame( t(mapply(WordPunctCount,testItemPairs$title1,testItemPairs$title2)))
testItemPairs$title1PunctCount <-  test_WordPunct[,1]
testItemPairs$title2PunctCount <-  test_WordPunct[,2]

trainItemPairs$Title1PunctsMatchedinTitle2ratio   <- ifelse(trainItemPairs$title1PunctCount == 0, 0, trainItemPairs$title2PunctCount/trainItemPairs$title1PunctCount)
trainItemPairs$Title2PunctsMatchedinTitle1ratio   <- ifelse(trainItemPairs$title2PunctCount == 0, 0, trainItemPairs$title1PunctCount/trainItemPairs$title2PunctCount)

trainItemPairs$Title1PunctsMatchedinTitle2ratioMin    <- pmin(trainItemPairs$Title1PunctsMatchedinTitle2ratio, trainItemPairs$Title2PunctsMatchedinTitle1ratio, na.rm=TRUE)
trainItemPairs$Title1PunctsMatchedinTitle2ratioMax    <- pmax(trainItemPairs$Title1PunctsMatchedinTitle2ratio, trainItemPairs$Title2PunctsMatchedinTitle1ratio, na.rm=TRUE)

trainItemPairs$Title1PunctsMatchedinTitle2ratioMin    <- ifelse(is.na(trainItemPairs$Title1PunctsMatchedinTitle2ratioMin) | trainItemPairs$Title1PunctsMatchedinTitle2ratioMin == Inf, -9999, trainItemPairs$Title1PunctsMatchedinTitle2ratioMin)
trainItemPairs$Title1PunctsMatchedinTitle2ratioMax    <- ifelse(is.na(trainItemPairs$Title1PunctsMatchedinTitle2ratioMax) | trainItemPairs$Title1PunctsMatchedinTitle2ratioMax == Inf, -9999, trainItemPairs$Title1PunctsMatchedinTitle2ratioMax)

testItemPairs$Title1PunctsMatchedinTitle2ratio   <- ifelse(testItemPairs$title1PunctCount == 0, 0, testItemPairs$title2PunctCount/testItemPairs$title1PunctCount)
testItemPairs$Title2PunctsMatchedinTitle1ratio   <- ifelse(testItemPairs$title2PunctCount == 0, 0, testItemPairs$title1PunctCount/testItemPairs$title2PunctCount)

testItemPairs$Title1PunctsMatchedinTitle2ratioMin    <- pmin(testItemPairs$Title1PunctsMatchedinTitle2ratio, testItemPairs$Title2PunctsMatchedinTitle1ratio, na.rm=TRUE)
testItemPairs$Title1PunctsMatchedinTitle2ratioMax    <- pmax(testItemPairs$Title1PunctsMatchedinTitle2ratio, testItemPairs$Title2PunctsMatchedinTitle1ratio, na.rm=TRUE)

testItemPairs$Title1PunctsMatchedinTitle2ratioMin    <- ifelse(is.na(testItemPairs$Title1PunctsMatchedinTitle2ratioMin) | testItemPairs$Title1PunctsMatchedinTitle2ratioMin == Inf, -9999, testItemPairs$Title1PunctsMatchedinTitle2ratioMin)
testItemPairs$Title1PunctsMatchedinTitle2ratioMax    <- ifelse(is.na(testItemPairs$Title1PunctsMatchedinTitle2ratioMax) | testItemPairs$Title1PunctsMatchedinTitle2ratioMax == Inf, -9999, testItemPairs$Title1PunctsMatchedinTitle2ratioMax)

trainItemPairs$Title1PunctsMatchedinTitle2ratioDiff    <- trainItemPairs$Title1PunctsMatchedinTitle2ratio -  trainItemPairs$Title2PunctsMatchedinTitle1ratio
trainItemPairs$Title1PunctsMatchedinTitle2ratioDiff    <- ifelse(is.na(trainItemPairs$Title1PunctsMatchedinTitle2ratioDiff) | trainItemPairs$Title1PunctsMatchedinTitle2ratioDiff == Inf, -9999, trainItemPairs$Title1PunctsMatchedinTitle2ratioDiff)

testItemPairs$Title1PunctsMatchedinTitle2ratioDiff    <- testItemPairs$Title1PunctsMatchedinTitle2ratio -  testItemPairs$Title2PunctsMatchedinTitle1ratio
testItemPairs$Title1PunctsMatchedinTitle2ratioDiff    <- ifelse(is.na(testItemPairs$Title1PunctsMatchedinTitle2ratioDiff) | testItemPairs$Title1PunctsMatchedinTitle2ratioDiff == Inf, -9999, testItemPairs$Title1PunctsMatchedinTitle2ratioDiff)

trainRowHashs$AvgWidthtoHeight_array2RatioDiff  <- trainRowHashs$AvgWidthtoHeight_array2Ratio  - trainRowHashs$AvgWidthtoHeight_array1Ratio
trainRowHashs$SumWidthtoHeight_array2RatioDiff  <- trainRowHashs$SumWidthtoHeight_array2Ratio  - trainRowHashs$SumWidthtoHeight_array1Ratio
trainRowHashs$MaxWidth_array1_array2RatioDiff   <- trainRowHashs$MaxWidth_array1_array2Ratio   - trainRowHashs$MaxWidth_array2_array1Ratio
trainRowHashs$MaxHeight_array1_array2RatioDiff  <- trainRowHashs$MaxHeight_array1_array2Ratio  - trainRowHashs$MaxHeight_array2_array1Ratio

testRowHashs$AvgWidthtoHeight_array2RatioDiff  <- testRowHashs$AvgWidthtoHeight_array2Ratio  - testRowHashs$AvgWidthtoHeight_array1Ratio
testRowHashs$SumWidthtoHeight_array2RatioDiff  <- testRowHashs$SumWidthtoHeight_array2Ratio  - testRowHashs$SumWidthtoHeight_array1Ratio
testRowHashs$MaxWidth_array1_array2RatioDiff   <- testRowHashs$MaxWidth_array1_array2Ratio   - testRowHashs$MaxWidth_array2_array1Ratio
testRowHashs$MaxHeight_array1_array2RatioDiff  <- testRowHashs$MaxHeight_array1_array2Ratio  - testRowHashs$MaxHeight_array2_array1Ratio


trainRowHashs$AvgEntropy_array2ImagesDiff    <- trainRowHashs$AvgEntropy_array2Images -  trainRowHashs$AvgEntropy_array1Images
trainRowHashs$AvgEntropy_array2ImagesDiff    <- ifelse(is.na(trainRowHashs$AvgEntropy_array2ImagesDiff) | trainRowHashs$AvgEntropy_array2ImagesDiff == Inf, -9999, trainRowHashs$AvgEntropy_array2ImagesDiff)

testRowHashs$AvgEntropy_array2ImagesDiff    <- testRowHashs$AvgEntropy_array2Images -  testRowHashs$AvgEntropy_array1Images
testRowHashs$AvgEntropy_array2ImagesDiff    <- ifelse(is.na(testRowHashs$AvgEntropy_array2ImagesDiff) | testRowHashs$AvgEntropy_array2ImagesDiff == Inf, -9999, testRowHashs$AvgEntropy_array2ImagesDiff)

###########################################################################################################
## New Features from v.08

ExtractNumbersFromString <- function(firsttitle1,secondtitle2){
  
  { temp1 <- ifelse(length(firsttitle1) >0 ,  gregexpr("[0-9]+", firsttitle1), "")
  NumbersFromString1 <- ifelse(length(temp1) == 0, "", as.numeric(unique(unlist(regmatches(firsttitle1, temp1)))))
  temp2 <- ifelse(length(secondtitle2) >0 ,  gregexpr("[0-9]+", secondtitle2), "")
  NumbersFromString2 <- ifelse(length(temp2) == 0, "",as.numeric(unique(unlist(regmatches(secondtitle2, temp2)))))
  NumbersFromString1 <- ifelse(is.na(NumbersFromString1),0,NumbersFromString1)
  NumbersFromString2 <- ifelse(is.na(NumbersFromString2),0,NumbersFromString2)
  }
  return(c(NumbersFromString1,NumbersFromString2 ))
}

train_ExtractNumbersFromString  <- as.data.frame( t(mapply(ExtractNumbersFromString,trainItemPairs$title1,trainItemPairs$title2)))

trainItemPairs$title1Numbers <-  train_ExtractNumbersFromString[,1]
trainItemPairs$title2NUmbers <-  train_ExtractNumbersFromString[,2] 

test_ExtractNumbersFromString  <- as.data.frame( t(mapply(ExtractNumbersFromString,testItemPairs$title1,testItemPairs$title2)))

testItemPairs$title1Numbers <-  test_ExtractNumbersFromString[,1]
testItemPairs$title2NUmbers <-  test_ExtractNumbersFromString[,2] 

trainItemPairs$titleNumbersMatch <- ifelse((trainItemPairs$title1Numbers == trainItemPairs$title2NUmbers) & trainItemPairs$title1Numbers != 0 , 1, 0)

testItemPairs$titleNumbersMatch <- ifelse((testItemPairs$title1Numbers == testItemPairs$title2NUmbers) & testItemPairs$title1Numbers != 0 , 1, 0)

##########################################################################################################
# new features -- 20160704
# library(qdapDictionaries)
# is.Englishword  <- function(x) x %in% GradyAugmented
# CountEnglishWords <- function(inputString1,inputString2 )
# {
#   
#   EnglishWordsCount1 <- 0
#   EnglishWordsCount2 <- 0
#   inputString1       <- unlist(strsplit(inputString1," "))
#   inputString2       <- unlist(strsplit(inputString2," "))
#   inputString1Length <- length(inputString1)
#   inputString2Length <- length(inputString2)
#   
#   for(i in 1:length(inputString1)){
#     inputStringWord1   <- inputString1[i]
#     EnglishWordsCount1 <- EnglishWordsCount1 + ifelse(is.Englishword(inputStringWord1)>=1,1,0)
#   }
#   for(i in 1:length(inputString2)){
#     inputStringWord2   <- inputString2[i]
#     EnglishWordsCount2 <- EnglishWordsCount2 + ifelse(is.Englishword(inputStringWord2)>=1,1,0)
#   }
#   
#   return(c(inputString1Length,EnglishWordsCount1,inputString2Length,EnglishWordsCount2 ))
# }

library(stringi)
library(stringr)
library(qdap)

CountEnglishWords <- function(inuptString1,inuptString2){
  inuptString1 <- paste("a " , inuptString1)
  inuptString2 <- paste("a " , inuptString2)
  EnglishWordsCount1  <- 0
  EnglishWordsCount2  <- 0
  {
    EnglishWordsCount1  <-   sum( word_count(str_extract_all(str_replace_all(inuptString1, "[[:punct:]]", ""), "[a-z]+"))) - 1
    EnglishWordsCount2  <-   sum( word_count(str_extract_all(str_replace_all(inuptString2, "[[:punct:]]", ""), "[a-z]+"))) - 1
  }
  return(c(EnglishWordsCount1,EnglishWordsCount2 ))
}

# validate function results with mix of both languages
train_CountEnglishWords  <- as.data.frame( t(mapply(CountEnglishWords,trainItemPairs$title1,trainItemPairs$title2)))

trainItemPairs$EWtitle1WordsEnglish <-  train_CountEnglishWords[,1] 
trainItemPairs$EWtitle2WordsEnglish <-  train_CountEnglishWords[,2]

trainItemPairs$EWtitle1totitle2Ratios <- ifelse(trainItemPairs$EWtitle2WordsEnglish == 0 , 0 , trainItemPairs$EWtitle1WordsEnglish /trainItemPairs$EWtitle2WordsEnglish)

trainItemPairs$EWtitle1Towords <- ifelse(trainItemPairs$Title1Length == 0, 0 , trainItemPairs$EWtitle1WordsEnglish / trainItemPairs$Title1Length)
trainItemPairs$EWtitle2Towords <- ifelse(trainItemPairs$Title2Length == 0, 0 , trainItemPairs$EWtitle2WordsEnglish / trainItemPairs$Title2Length)

trainItemPairs$EWtitle1totitle2Diff  <- trainItemPairs$EWtitle1Towords - trainItemPairs$EWtitle2Towords
trainItemPairs$EWTitlesToWordsRatios <- ifelse(trainItemPairs$EWtitle2Towords == 0 , 0 , trainItemPairs$EWtitle1Towords /trainItemPairs$EWtitle2Towords)


# validate function results with mix of both languages
test_CountEnglishWords  <- as.data.frame( t(mapply(CountEnglishWords,testItemPairs$title1,testItemPairs$title2)))

testItemPairs$EWtitle1WordsEnglish <-  test_CountEnglishWords[,1] 
testItemPairs$EWtitle2WordsEnglish <-  test_CountEnglishWords[,2]

testItemPairs$EWtitle1totitle2Ratios <- ifelse(testItemPairs$EWtitle2WordsEnglish == 0 , 0 , testItemPairs$EWtitle1WordsEnglish /testItemPairs$EWtitle2WordsEnglish)

testItemPairs$EWtitle1Towords <- ifelse(testItemPairs$Title1Length == 0, 0 , testItemPairs$EWtitle1WordsEnglish / testItemPairs$Title1Length)
testItemPairs$EWtitle2Towords <- ifelse(testItemPairs$Title2Length == 0, 0 , testItemPairs$EWtitle2WordsEnglish / testItemPairs$Title2Length)

testItemPairs$EWtitle1totitle2Diff  <- testItemPairs$EWtitle1Towords - testItemPairs$EWtitle2Towords
testItemPairs$EWTitlesToWordsRatios <- ifelse(testItemPairs$EWtitle2Towords == 0 , 0 , testItemPairs$EWtitle1Towords /testItemPairs$EWtitle2Towords)





###########################################################################################################

trainItemPairsFull<- join(trainItemPairs, trainRowHashs, type="left")
testItemPairsFull <- join(testItemPairs , testRowHashs , type="left")

####################################################################################################

# Sys.time()
# save.image(file = "Avito_10.RData" , safe = TRUE)
# Sys.time()
# load("Avito_09.RData")
# Sys.time()

# head(trainItemPairs$title1)
# head(trainItemPairs$title2)
###################################################################################################


# 
# trainItemPairsFull$priceMinHammingRation         <- ifelse(trainItemPairsFull$MinHammingDistance == 0, 0,  trainItemPairsFull$priceDiff/trainItemPairsFull$MinHammingDistance )   
# trainItemPairsFull$MinHammingCountBelow10Ration  <- ifelse(trainItemPairsFull$CountBelow10distance == 0, 0, trainItemPairsFull$MinHammingDistance/	trainItemPairsFull$CountBelow10distance)
# trainItemPairsFull$descCharCountBelow10Ration    <- ifelse(trainItemPairsFull$CountBelow10distance == 0, 0, trainItemPairsFull$descriptionCharMax/trainItemPairsFull$CountBelow10distance)
# trainItemPairsFull$pricetitles2gramRation        <- ifelse(trainItemPairsFull$title12gramtitle2 == 0 , 0 ,trainItemPairsFull$priceDiff	/ trainItemPairsFull$title12gramtitle2)
# trainItemPairsFull$MinHammingtitles2gramRation   <- ifelse(trainItemPairsFull$title12gramtitle2 == 0 , 0 ,trainItemPairsFull$MinHammingDistance	/trainItemPairsFull$title12gramtitle2)
# trainItemPairsFull$MaxHammingCountBelow10Ration  <- ifelse(trainItemPairsFull$CountBelow10distance == 0 , 0 , trainItemPairsFull$MaxHammingDistance/	trainItemPairsFull$CountBelow10distance)
# trainItemPairsFull$MinHammingEntropyarraysRation <- ifelse(trainItemPairsFull$AvgEntropy_array2_array1Images == 0 , 0 ,trainItemPairsFull$MinHammingDistance/	trainItemPairsFull$AvgEntropy_array2_array1Images)
# trainItemPairsFull$priceTitlesMatchRation        <- ifelse(trainItemPairsFull$Title2WordsMatchedinTitle1ratio == 0 , 0 ,trainItemPairsFull$priceDiff/	trainItemPairsFull$Title2WordsMatchedinTitle1ratio)
# trainItemPairsFull$priceCountBelow10Ration       <- ifelse(trainItemPairsFull$CountBelow10distance == 0 , 0, trainItemPairsFull$priceDiff/	trainItemPairsFull$CountBelow10distance)
# trainItemPairsFull$pricetitles1MatchRation       <- ifelse(trainItemPairsFull$Title1WordsMatchedinTitle2ratio == 0 , 0, trainItemPairsFull$priceDiff/	trainItemPairsFull$Title1WordsMatchedinTitle2ratio)
# 
# trainItemPairsFull$priceMinHammingRation2         <- ifelse(trainItemPairsFull$priceDiff          == 0 , 0  , trainItemPairsFull$MinHammingDistance           	 / trainItemPairsFull$priceDiff           )   
# trainItemPairsFull$MinHammingCountBelow10Ration2  <- ifelse(trainItemPairsFull$MinHammingDistance == 0 , 0  , trainItemPairsFull$CountBelow10distance			 / trainItemPairsFull$MinHammingDistance  )
# trainItemPairsFull$descCharCountBelow10Ration2    <- ifelse(trainItemPairsFull$descriptionCharMax == 0 , 0  , trainItemPairsFull$CountBelow10distance			 / trainItemPairsFull$descriptionCharMax  )
# trainItemPairsFull$pricetitles2gramRation2        <- ifelse(trainItemPairsFull$priceDiff	        == 0 , 0  , trainItemPairsFull$title12gramtitle2				 / trainItemPairsFull$priceDiff	       )
# trainItemPairsFull$MinHammingtitles2gramRation2   <- ifelse(trainItemPairsFull$MinHammingDistance == 0 , 0  , trainItemPairsFull$title12gramtitle2				 / trainItemPairsFull$MinHammingDistance  )
# trainItemPairsFull$MaxHammingCountBelow10Ration2  <- ifelse(trainItemPairsFull$MaxHammingDistance == 0 , 0  , trainItemPairsFull$CountBelow10distance			 / trainItemPairsFull$MaxHammingDistance  )
# trainItemPairsFull$MinHammingEntropyarraysRation2 <- ifelse(trainItemPairsFull$MinHammingDistance == 0 , 0  , trainItemPairsFull$AvgEntropy_array2_array1Images	 / trainItemPairsFull$MinHammingDistance  )
# trainItemPairsFull$priceTitlesMatchRation2        <- ifelse(trainItemPairsFull$priceDiff          == 0 , 0  , trainItemPairsFull$Title2WordsMatchedinTitle1ratio / trainItemPairsFull$priceDiff           )
# trainItemPairsFull$priceCountBelow10Ration2       <- ifelse(trainItemPairsFull$priceDiff          == 0 , 0  , trainItemPairsFull$CountBelow10distance			 / trainItemPairsFull$priceDiff           )
# trainItemPairsFull$pricetitles1MatchRation2       <- ifelse(trainItemPairsFull$priceDiff          == 0 , 0  , trainItemPairsFull$Title1WordsMatchedinTitle2ratio / trainItemPairsFull$priceDiff           )
# 
# 
# trainItemPairsFull$priceDiff3wayRatio	                     <- ifelse( trainItemPairsFull$title12gramtitle2   == 0, 0 , (trainItemPairsFull$priceDiff * (trainItemPairsFull$MinHammingDistance/trainItemPairsFull$title12gramtitle2)))
# trainItemPairsFull$priceDiff3wayRatio2	                     <- ifelse(trainItemPairsFull$MinHammingDistance   == 0 , 0 ,(trainItemPairsFull$priceDiff * (trainItemPairsFull$Title2WordsMatchedinTitle1ratio/trainItemPairsFull$MinHammingDistance)))
# trainItemPairsFull$descriptionCharMax3wayRatio	             <- ifelse(trainItemPairsFull$CountBelow10distance == 0 , 0 ,(trainItemPairsFull$descriptionCharMax * (trainItemPairsFull$MinHammingDistance/trainItemPairsFull$CountBelow10distance)))
# trainItemPairsFull$priceDiff3wayRatio3	                     <- ifelse(trainItemPairsFull$CountBelow10distance == 0 , 0 ,(trainItemPairsFull$priceDiff	                    * (trainItemPairsFull$MinHammingDistance	            / trainItemPairsFull$CountBelow10distance)))
# trainItemPairsFull$Title2WordsMatchedinTitle1ratio3wayRatio <- ifelse(trainItemPairsFull$CountBelow10distance == 0 , 0 ,(trainItemPairsFull$Title2WordsMatchedinTitle1ratio	* (trainItemPairsFull$MinHammingDistance	            / trainItemPairsFull$CountBelow10distance)))
# trainItemPairsFull$nchardescription13wayRatio	              <- ifelse(trainItemPairsFull$CountBelow10distance == 0 , 0 ,(trainItemPairsFull$nchardescription1	            * (trainItemPairsFull$MinHammingDistance	            / trainItemPairsFull$CountBelow10distance)))
# trainItemPairsFull$priceDiff3wayRatio4	                     <- ifelse(trainItemPairsFull$MinHammingDistance   == 0 , 0 ,(trainItemPairsFull$priceDiff	                    * (trainItemPairsFull$Title1WordsMatchedinTitle2ratio	/ trainItemPairsFull$MinHammingDistance  )))
# trainItemPairsFull$MaxHammingDistance3wayRatio	             <- ifelse(trainItemPairsFull$title12gramtitle2    == 0 , 0 ,(trainItemPairsFull$MaxHammingDistance	            * (trainItemPairsFull$CountBelow10distance	            / trainItemPairsFull$title12gramtitle2   )))
# trainItemPairsFull$Title1WordsMatchedinTitle2ratio3wayRatio <- ifelse(trainItemPairsFull$CountBelow10distance == 0 , 0 ,(trainItemPairsFull$Title1WordsMatchedinTitle2ratio	* (trainItemPairsFull$MinHammingDistance	            / trainItemPairsFull$CountBelow10distance)))
# trainItemPairsFull$MinHammingDistance3wayRatio	             <- ifelse(trainItemPairsFull$title12gramtitle2    == 0 , 0 ,(trainItemPairsFull$MinHammingDistance	            * (trainItemPairsFull$priceDifference	                / trainItemPairsFull$title12gramtitle2   )))
# 
# 
# testItemPairsFull$priceMinHammingRation         <- ifelse(testItemPairsFull$MinHammingDistance == 0, 0,  testItemPairsFull$priceDiff/testItemPairsFull$MinHammingDistance )   
# testItemPairsFull$MinHammingCountBelow10Ration  <- ifelse(testItemPairsFull$CountBelow10distance == 0, 0, testItemPairsFull$MinHammingDistance/	testItemPairsFull$CountBelow10distance)
# testItemPairsFull$descCharCountBelow10Ration    <- ifelse(testItemPairsFull$CountBelow10distance == 0, 0, testItemPairsFull$descriptionCharMax/testItemPairsFull$CountBelow10distance)
# testItemPairsFull$pricetitles2gramRation        <- ifelse(testItemPairsFull$title12gramtitle2 == 0 , 0 ,testItemPairsFull$priceDiff	/ testItemPairsFull$title12gramtitle2)
# testItemPairsFull$MinHammingtitles2gramRation   <- ifelse(testItemPairsFull$title12gramtitle2 == 0 , 0 ,testItemPairsFull$MinHammingDistance	/testItemPairsFull$title12gramtitle2)
# testItemPairsFull$MaxHammingCountBelow10Ration  <- ifelse(testItemPairsFull$CountBelow10distance == 0 , 0 , testItemPairsFull$MaxHammingDistance/	testItemPairsFull$CountBelow10distance)
# testItemPairsFull$MinHammingEntropyarraysRation <- ifelse(testItemPairsFull$AvgEntropy_array2_array1Images == 0 , 0 ,testItemPairsFull$MinHammingDistance/	testItemPairsFull$AvgEntropy_array2_array1Images)
# testItemPairsFull$priceTitlesMatchRation        <- ifelse(testItemPairsFull$Title2WordsMatchedinTitle1ratio == 0 , 0 ,testItemPairsFull$priceDiff/	testItemPairsFull$Title2WordsMatchedinTitle1ratio)
# testItemPairsFull$priceCountBelow10Ration       <- ifelse(testItemPairsFull$CountBelow10distance == 0 , 0, testItemPairsFull$priceDiff/	testItemPairsFull$CountBelow10distance)
# testItemPairsFull$pricetitles1MatchRation       <- ifelse(testItemPairsFull$Title1WordsMatchedinTitle2ratio == 0 , 0, testItemPairsFull$priceDiff/	testItemPairsFull$Title1WordsMatchedinTitle2ratio)
# 
# testItemPairsFull$priceMinHammingRation2         <- ifelse(testItemPairsFull$priceDiff          == 0 , 0  , testItemPairsFull$MinHammingDistance           	 / testItemPairsFull$priceDiff           )   
# testItemPairsFull$MinHammingCountBelow10Ration2  <- ifelse(testItemPairsFull$MinHammingDistance == 0 , 0  , testItemPairsFull$CountBelow10distance			 / testItemPairsFull$MinHammingDistance  )
# testItemPairsFull$descCharCountBelow10Ration2    <- ifelse(testItemPairsFull$descriptionCharMax == 0 , 0  , testItemPairsFull$CountBelow10distance			 / testItemPairsFull$descriptionCharMax  )
# testItemPairsFull$pricetitles2gramRation2        <- ifelse(testItemPairsFull$priceDiff	        == 0 , 0  , testItemPairsFull$title12gramtitle2				 / testItemPairsFull$priceDiff	       )
# testItemPairsFull$MinHammingtitles2gramRation2   <- ifelse(testItemPairsFull$MinHammingDistance == 0 , 0  , testItemPairsFull$title12gramtitle2				 / testItemPairsFull$MinHammingDistance  )
# testItemPairsFull$MaxHammingCountBelow10Ration2  <- ifelse(testItemPairsFull$MaxHammingDistance == 0 , 0  , testItemPairsFull$CountBelow10distance			 / testItemPairsFull$MaxHammingDistance  )
# testItemPairsFull$MinHammingEntropyarraysRation2 <- ifelse(testItemPairsFull$MinHammingDistance == 0 , 0  , testItemPairsFull$AvgEntropy_array2_array1Images	 / testItemPairsFull$MinHammingDistance  )
# testItemPairsFull$priceTitlesMatchRation2        <- ifelse(testItemPairsFull$priceDiff          == 0 , 0  , testItemPairsFull$Title2WordsMatchedinTitle1ratio / testItemPairsFull$priceDiff           )
# testItemPairsFull$priceCountBelow10Ration2       <- ifelse(testItemPairsFull$priceDiff          == 0 , 0  , testItemPairsFull$CountBelow10distance			 / testItemPairsFull$priceDiff           )
# testItemPairsFull$pricetitles1MatchRation2       <- ifelse(testItemPairsFull$priceDiff          == 0 , 0  , testItemPairsFull$Title1WordsMatchedinTitle2ratio / testItemPairsFull$priceDiff           )
# 
# 
# testItemPairsFull$priceDiff3wayRatio	                     <- ifelse( testItemPairsFull$title12gramtitle2   == 0, 0 , (testItemPairsFull$priceDiff * (testItemPairsFull$MinHammingDistance/testItemPairsFull$title12gramtitle2)))
# testItemPairsFull$priceDiff3wayRatio2	                     <- ifelse(testItemPairsFull$MinHammingDistance   == 0 , 0 ,(testItemPairsFull$priceDiff * (testItemPairsFull$Title2WordsMatchedinTitle1ratio/testItemPairsFull$MinHammingDistance)))
# testItemPairsFull$descriptionCharMax3wayRatio	             <- ifelse(testItemPairsFull$CountBelow10distance == 0 , 0 ,(testItemPairsFull$descriptionCharMax * (testItemPairsFull$MinHammingDistance/testItemPairsFull$CountBelow10distance)))
# testItemPairsFull$priceDiff3wayRatio3	                     <- ifelse(testItemPairsFull$CountBelow10distance == 0 , 0 ,(testItemPairsFull$priceDiff	                    * (testItemPairsFull$MinHammingDistance	            / testItemPairsFull$CountBelow10distance)))
# testItemPairsFull$Title2WordsMatchedinTitle1ratio3wayRatio <- ifelse(testItemPairsFull$CountBelow10distance == 0 , 0 ,(testItemPairsFull$Title2WordsMatchedinTitle1ratio	* (testItemPairsFull$MinHammingDistance	            / testItemPairsFull$CountBelow10distance)))
# testItemPairsFull$nchardescription13wayRatio	             <- ifelse(testItemPairsFull$CountBelow10distance == 0 , 0 ,(testItemPairsFull$nchardescription1	            * (testItemPairsFull$MinHammingDistance	            / testItemPairsFull$CountBelow10distance)))
# testItemPairsFull$priceDiff3wayRatio4	                     <- ifelse(testItemPairsFull$MinHammingDistance   == 0 , 0 ,(testItemPairsFull$priceDiff	                    * (testItemPairsFull$Title1WordsMatchedinTitle2ratio	/ testItemPairsFull$MinHammingDistance  )))
# testItemPairsFull$MaxHammingDistance3wayRatio	             <- ifelse(testItemPairsFull$title12gramtitle2    == 0 , 0 ,(testItemPairsFull$MaxHammingDistance	            * (testItemPairsFull$CountBelow10distance	            / testItemPairsFull$title12gramtitle2   )))
# testItemPairsFull$Title1WordsMatchedinTitle2ratio3wayRatio <- ifelse(testItemPairsFull$CountBelow10distance == 0 , 0 ,(testItemPairsFull$Title1WordsMatchedinTitle2ratio	* (testItemPairsFull$MinHammingDistance	            / testItemPairsFull$CountBelow10distance)))
# testItemPairsFull$MinHammingDistance3wayRatio	             <- ifelse(testItemPairsFull$title12gramtitle2    == 0 , 0 ,(testItemPairsFull$MinHammingDistance	            * (testItemPairsFull$priceDifference	                / testItemPairsFull$title12gramtitle2   )))
# 


####################################################################################################

# saveRDS(trainItemPairs ,  "trainItemPairs.rds")
# saveRDS(trainRowHashs  ,  "trainRowHashs.rds")
# saveRDS(testItemPairs  ,  "testItemPairs.rds")
# saveRDS(testRowHashs   ,  "testRowHashs.rds")
###########################################################################################################

#summary(trainItemPairs$description12gramdescription2)


trainItemPairsFull[is.na(trainItemPairsFull) | trainItemPairsFull==-9999] <- 0
testItemPairsFull[is.na(testItemPairsFull)| testItemPairsFull==-9999]     <- 0



# head(testItemPairsFull$MaxEntropyDiff)

features  <- c(
  "itemID_1"
  ,"itemID_2"
  #,"isDuplicate"
  ,"id"
  , "sameLat"
  ,"sameLon"
  ,"sameLocation"
  ,"sameregion"
  ,"priceDifference"
  ,"sameTitle"
  ,"sameDescription"
  ,"imageArrayDiff"                         
  ,"SimilarityTitle"
  ,"SimilarityDescription"
  ,"samemetro"
  ,"SimilarityJSON"                        
  ,"distance"
  ,"nchardescription1"   
  ,"nchardescription2"   
  ,"priceDiff"          
  ,"priceMin"            
  ,"priceMax"            
  ,"titleStringDistance2"                  
  ,"title1StartsWithTitle2"
  ,"title2StartsWithTitle1"
  ,"titleCharDiff"
  ,"titleCharMin"
  ,"titleCharMax"
  ,"descriptionCharDiff"
  ,"descriptionCharMin" 
  ,"descriptionCharMax"
  ,"Title1WordsMatchedinTitle2ratio"
  ,"Title2WordsMatchedinTitle1ratio"         
  ,"ImageCombinations"
  ,"MinHammingDistance"
  ,"MaxHammingDistance"
  ,"MaxMinDifference"                       
  #,"stdHammingDistance"                    # not using, due to noise from test results
  ,"AvgHammingDistance"
  ,"CountBelow10distance"
  ,"CountAbove50distance"
  ,"Title1WordsMatchedinDesc2ratio"
  ,"Title2WordsMatchedinDesc1ratio"
  ,"description1WordsMatchedinDesc2ratio"    
  ,"CountBetween11and20distance"
  ,"CountBetween21and30distance"
  ,"CountBetween31and40distance"
  ,"CountBetween41and50distance"
  ,"attrsJSON1WordsMatchedinattrsJSON2ratio"
  ,"Title1WordsMatchedinTitle2ratioDiff"
  ,"Title1WordsMatchedinTitle2ratioMin" 
  ,"Title1WordsMatchedinTitle2ratioMax"
  ,"Title1WordsMatchedinDesc2ratioDiff" 
  ,"Title1WordsMatchedinDesc2ratioMin"  
  ,"Title1WordsMatchedinDesc2ratioMax"
  ,"Title1WordsMatchedinTitle2Diff"
  ,"Title1WordsMatchedinDesc2Diff"
  ,"title1StartingWithTitle2" 
  #,"title2StartingWithTitle1"          # not using, due to duplicate with above features
  ,"title1EndsWithTitle2" 
  #,"title2EndsWithTitle1"              # not using, due to duplicate with above features
  ,"title1SoundexWithTitle2"
  ,"title1EndSoundexWithTitle2"  
  ,"title1StartSoundexWithTitle2"
  ,"Title1WordsStrengthinTitle2"
  ,"Title1WordsStrengthinTitle2ratio"
  ,"title12gramtitle2"
  #,"title13gramtitle2"                # not using, due to noise
  #,"title14gramtitle2"                # not using, due to noise
  #,"title15gramtitle2"                # not using, due to noise
  #,"title12gramdescription2"          # not using, due to noise
  #,"description12gramdescription2"    # not using, due to noise
  ,"Title1TwoWordsMatchedinTitle2"     
  ,"Title1TwoWordsMatchedinDesc2"      
  #,"Title2TwoWordsMatchedinDesc1"     # not using, due to noise
  #,"Title1TwoWordsMatchedinDescratio" # not using, due to noise
  #,"Desc1TwoWordsMatchedinDesc2"      # not using, due to noise
  #,"titlesCosineDist"                 # not using, due to noise
  #,"titlesCosinesDist"                # not using, due to noise
  ,"SameWidthCount"
  ,"MaxWidthDiff"                           
  ,"MinWidthDiff"                            
  ,"AvgWidthDiff"                            
  ,"SameHeightCount"                        
  ,"MaxHeightDiff"                           
  ,"MinHeightDiff"                           
  ,"AvgHeightDiff"                          
  ,"SameEntropyCount"                        
  ,"MaxEntropyDiff"                          
  ,"MinEntropyDiff"                         
  ,"AvgEntropyDiff"                          
  ,"SamecompressedsizeCount"                 
  ,"MaxcompressedsizeDiff"                  
  ,"MincompressedsizeDiff"                   
  ,"AvgcompressedsizeDiff"
  ,"AvgEntropyarray1and2Ratio"
  ,"AvgWidthyarray1and2Ratio" 
  ,"AvgHeightarray1and2Ratio" 
  ,"Avgcompressed_sizearray1and2Ratio"
  ,"Avgsizearray1and2Ratio"
  ,"AvgEntropy_array2Images"
  ,"AvgEntropy_array1Images"
  ,"AvgWidthtoHeight_array2Ratio"
  ,"AvgWidthtoHeight_array1Ratio"
  ,"SumWidthtoHeight_array2Ratio"
  ,"SumWidthtoHeight_array1Ratio"
  ,"MaxWidth_array1_array2Ratio" 
  ,"MaxWidth_array2_array1Ratio" 
  ,"MaxHeight_array1_array2Ratio"
  ,"MaxHeight_array2_array1Ratio"
  ,"AvgEntropy_array1_array2Images"
  ,"AvgEntropy_array2_array1Images"
  #,"SumEntropy_array1_array2Images"  # not using, due to noise
  #,"SumEntropy_array2_array1Images"  # not using, due to noise
  #,"Title1PunctsMatchedinTitle2ratioMin"
  #,"Title1PunctsMatchedinTitle2ratioMax"
  ,"Title1PunctsMatchedinTitle2ratioDiff"
  ,"AvgWidthtoHeight_array2RatioDiff"
  ,"SumWidthtoHeight_array2RatioDiff"
  ,"MaxWidth_array1_array2RatioDiff" 
  ,"MaxHeight_array1_array2RatioDiff"
  ,"AvgEntropy_array2ImagesDiff"
  ,"titleNumbersMatch"
  ,"EWtitle1totitle2Ratios"
  ,"EWtitle1totitle2Diff" 
  ,"EWTitlesToWordsRatios"
)

# summary(trainItemPairsFull$Title1WordsStrengthinTitle2)
# summary(trainItemPairsFull$Title1WordsStrengthinTitle2ratio)

# names(trainItemPairsFull)
training <- trainItemPairsFull[,features,with=FALSE]
testing  <- testItemPairsFull[,features,with=FALSE]
response <- trainItemPairsFull$isDuplicate

# names(training)
####################### feature files for Kele ####################################################
# remove isDuplicate from features while generating testing files

# write.csv(training, './featurefiles_20160706/TrainingFeatures.csv', row.names=FALSE, quote = FALSE)
# write.csv(testing , './featurefiles_20160706/TestingFeatures.csv', row.names=FALSE, quote = FALSE)

####################################################################################################


dtrain<-xgb.DMatrix(data=data.matrix(training),label=trainItemPairsFull$isDuplicate)

#watchlist<-list(val=dval,train=dtrain)
param <- list(  "objective"        = "binary:logistic", 
                "booster"          = "gbtree",
                "eval_metric"      = "auc",#"merror",#"auc",
                "nthread"          = 6,    # number of threads to be used 
                "max_depth"        = 8,    # 6 used for all test but 8, 10 giving best results, 
                # this is partly tree learning to perticular case hence possibility of over fit
                # use 6 for now
                "eta"              = 0.02, #
                #"gamma"            = 0,   # minimum loss reduction 
                "subsample"        = 0.6,  # 0.6 - try with 0.6 to get 4th decimal change
                "colsample_bytree" = 0.9,  # 0.9 - seems the best fit
                "min_child_weight" = 3     # 6 - no change in the results i.e. use 3
                #num_parallel_tree   = 2
                # alpha = 0.0001, 
                # lambda = 1
)


nround.cv = 10     # Originally tested with 100 and 1500, now using 500
system.time( bst.cv <- xgb.cv(  param=param
                                , data=dtrain
                                , early.stop.round=10
                                , nfold=5
                                , nrounds=nround.cv
                                , prediction=TRUE
                                , verbose=TRUE
                                , maximize = TRUE
                                , print.every.n=1
                                , set.seed(2500)
) 
)


max.auc.idx = which.max(bst.cv$dt[, test.auc.mean]) 
max.auc.idx
bst.cv$dt[max.auc.idx ,]
# Baseline from previous features, v.08
#[0]	train-auc:0.912172+0.000624	test-auc:0.911630+0.000862
#1:       0.919048      0.000314      0.918505     0.000185 , 10 rounds

#########################################################################################################
# 1:        0.76798      0.000111      0.767445     0.000371, 1000 
# 1:        0.77159      0.000118      0.7708       0.000386, 1000
# 1:        0.789031     0.00012       0.787475     0.000396, 1000 lb : 0.76132
# 1:        0.81114      0.000136      0.809064     0.000222, 1500 lb : 0.77298
# 1:        0.859027     0.000119      0.856071     0.000306, 1500 lb : 0.80344
# 1:        0.861738     0.000146      0.8588       0.000426, 1500 lb : 0.80529
# 1:        0.862457     0.000199      0.85951      0.000445, 1500 lb : 
# 1:        0.913595     0.000164      0.913349     0.000293,  100 lb : 0.89417
# 1:        0.934599     6.3e-05       0.932783     0.000164, 1500 lb : 0.90505
# 1:        0.920888      2.4e-05      0.920176     0.000282,  500 lb : 0.89       # not used 4 features
# 1:        0.927314      7.7e-05      0.926601     0.000278,  500 lb : 0.902      # not used 2 features, used 2 new features
# 1:        0.935356     0.000117      0.933587     0.000189, 1500 lb : 0.906      # not used 2 features, used 2 new features
# 1:        0.936887      6.9e-05      0.935059     0.000196, 1500 lb : 0.9082     # not used 1 features, used 2 new features
# 1:        0.93699       7.7e-05      0.935173     0.000222, 1500 lb : 0.90842    # used all features
# 1:        0.937902      2.9e-05      0.936096     0.000206, 1500 lb : 0.9089     # included one new feature
# 1:        0.938861      2.6e-05      0.937020     0.000234, 1500 lb : 0.909      # included new features
# 1:        0.948569      3.6e-05      0.943164     0.000171, 1500 lb : 0.9120     # param 8 and 0.6
# 1:       0.949022       4.8e-05      0.943647     0.000187, 1500 lb : 0.9127     #  
# 1:       0.949784       7.4e-05      0.944389     0.000228, 1500 lb : 0.9137     # exclude 3 and include 2 new features
# 1:       0.950859       4.2e-05      0.945434     0.000233, 1500 lb : 0.915      # exclude 6 features due to noise
# 1:       0.950825         9e-06      0.945412     0.000194, 1500 lb : 0.915      # excluding all noise features
#########################################################################################################
XGModel <- xgb.train(   params              = param, 
                        data                = dtrain, 
                        nrounds             = max.auc.idx, #1800, 
                        verboseIter         = TRUE,  #1
                        #early.stop.round   = 150,
                        #watchlist          = watchlist,
                        maximize            = TRUE
)

testXGBoostpred <- predict(XGModel, data.matrix(testing))

submission <- data.frame(id=testItemPairsFull$id,probability = testXGBoostpred)
head(submission)
write.csv(submission, 'Avito_02_XGB_cv94.csv', row.names=FALSE, quote = FALSE)



##########################################################################################################


image_hash0001 <- NULL
FilteredtestImages <- NULL
FilteredtrainImages <- NULL
trainImageCombs <- NULL
testImageCombs <- NULL
trainImageCombsHashs <- NULL
testImageCombsHashs <- NULL
FilteredtrainImages1 <- NULL
FilteredtestImages1 <- NULL
FilteredtrainImages2 <- NULL
FilteredtestImages2 <- NULL
trainImagesHashs <- NULL
testImagesHashs <- NULL
gc()


library(qdapDictionaries)

is.Englishword  <- function(x) x %in% GradyAugmented

CountEnglishWords <- function(inputString)
{
  
  EnglishWordsCount <- 0
  inputString       <- unlist(strsplit(inputString," "))
  inputStringLength <- length(inputString)
  
  for(i in 1:length(inputString)){
    inputStringWord     <- inputString[i]
    EnglishWordsCount <- EnglishWordsCount + ifelse(is.Englishword(inputStringWord)>=1,1,0)
  }
  
  return(c(inputStringLength,EnglishWordsCount))
}

# validate function results with mix of both languages
train_CountEnglishWords  <- as.data.frame( t(mapply(CountEnglishWords,"don't know russian language ???????????????????????????? ?????????????? ???????????????? ????????????")))

trainItemPairs$title1Numbers <-  train_CountEnglishWords[,1]
trainItemPairs$title2NUmbers <-  train_CountEnglishWords[,2] 

# validating the function
is.Englishword(c("don't", "know", "russian", "language", "????????????????????????????", "??????????????", "????????????????", "????????????"))

head(testItemInfo$title,2)

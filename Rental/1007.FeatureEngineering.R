require(tidyr)

Image_hashs <- read_csv("./input/Image_Hash_02.csv")
head(Image_hashs)


Image_hashs02 = Image_hashs

names(Image_hashs02)[1] <- "folder2"
names(Image_hashs02)[2] <- "imageNo2"
names(Image_hashs02)[3] <- "hash02"

head(Image_hashs02)

trainImageCombs <-  merge(Image_hashs ,Image_hashs02,by = "listing_id", all = TRUE ,allow.cartesian=TRUE)
temp <- trainImageCombs[trainImageCombs$folder == 6811957,]
trainImageCombs <- data.table(trainImageCombs)

head(trainImageCombs)

trainImageCombs$hammingdistance <-  stringdist(trainImageCombs$hash, trainImageCombs$hash02, method = c("hamming"))

DuplicateImages <- trainImageCombs[trainImageCombs$hammingdistance <= 25 & trainImageCombs$imageNo != trainImageCombs$imageNo2 ,]

DuplicateImageCounts <- sqldf("SELECT listing_id, count(*) as [DuplicateImageCount] FROM DuplicateImages group by listing_id")

head(DuplicateImageCounts)

write.csv(DuplicateImageCounts,  './input/Prav_listingDuplicateImageCounts.csv', row.names=FALSE, quote = FALSE)

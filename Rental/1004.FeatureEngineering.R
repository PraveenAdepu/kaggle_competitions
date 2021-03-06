
PhotoFeatures <- read_csv("./input/Image_Exif_01.csv")
# 
# train_ListingIdPhotos <- read_csv("./input/train_listingIdPhotos.csv")
# train_ListingIdPhotos <- as.data.table(train_ListingIdPhotos)

# 
# temp <- PhotoFeatures[grep("7170325|6811957",PhotoFeatures$imageNo), ]
# temp$listing_id  <-  as.integer(lapply(strsplit(as.character(temp$imageNo), split="_"),head, n=1))
# temp = as.data.table(temp)
# 
# temp$Meansizebytes
# 
# temp[,c( "MeanWidth"
#         ,"MeanHeight" 
#         ,"Meansizebytes"
#         , "meanextrema00"
#         , "meanextrema01"
#         , "meanextrema10"
#         , "meanextrema11"
#         , "meanextrema20"
#         , "meanextrema21"
#         , "meancount00"
#         , "meancount01"
#         , "meancount02"
#         , "meansum00"
#         , "meansum01"
#         , "meansum02"
#         , "meanmean00"
#         , "meanmean01"
#         , "meanmean02"
#         , "meanmedian00"
#         , "meanmedian01"
#         , "meanmedian02"
#         , "meanrms00"
#         , "meanrms01"
#         , "meanrms02"
#         , "meanvar00"
#         , "meanvar01"
#         , "meanvar02"
#         , "meanstddev00"
#         , "meanstddev01"
#         , "meanstddev02"):=list(mean(width)
#                                                          , mean(height)
#                                                          , mean(sizebytes)
#                                                          , mean(extrema00)
#                                                          , mean(extrema01)
#                                                          , mean(extrema10)
#                                                          , mean(extrema11)
#                                                          , mean(extrema20)
#                                                          , mean(extrema21)
#                                                          , mean(count00)
#                                                          , mean(count01)
#                                                          , mean(count02)
#                                                          , mean(sum00)
#                                                          , mean(sum01)
#                                                          , mean(sum02)
#                                                          , mean(mean00)
#                                                          , mean(mean01)
#                                                          , mean(mean02)
#                                                          , mean(median00)
#                                                          , mean(median01)
#                                                          , mean(median02)
#                                                          , mean(rms00)
#                                                          , mean(rms01)
#                                                          , mean(rms02)
#                                                          , mean(var00)
#                                                          , mean(var01)
#                                                          , mean(var02)
#                                                          , mean(stddev00)
#                                                          , mean(stddev01)
#                                                          , mean(stddev02)
#                                                          )
#      ,by=listing_id]
# 
# head(temp)
head(PhotoFeatures)

PhotoFeatures$listing_id  <-  as.integer(lapply(strsplit(as.character(PhotoFeatures$imageNo), split="_"),head, n=1))

PhotoFeatures = as.data.table(PhotoFeatures)

PhotoFeatures[,c(   "meanColorSpace"
                  , "meanContrast" 
                  , "meanCustomRendered"
                  , "meanExifOffset"
                  , "meanExposureMode"
                  , "meanFlash"
                  , "meanLightSource"
                  , "meanSharpness"
                  , "meanSubjectDistanceRange"
                  , "meanWhiteBalance"
                  , "meanYCbCrPositioning"
                
):=list(                    mean(ColorSpace)
                          , mean(Contrast)
                          , mean(CustomRendered)
                          , mean(ExifOffset)
                          , mean(ExposureMode)
                          , mean(Flash)
                          , mean(LightSource)
                          , mean(Sharpness)
                          , mean(SubjectDistanceRange)
                          , mean(WhiteBalance)
                          , mean(YCbCrPositioning)
)
,by=listing_id]

head(PhotoFeatures)



featureColumns = c("listing_id", "meanColorSpace"
                   , "meanContrast" 
                   , "meanCustomRendered"
                   , "meanExifOffset"
                   , "meanExposureMode"
                   , "meanFlash"
                   , "meanLightSource"
                   , "meanSharpness"
                   , "meanSubjectDistanceRange"
                   , "meanWhiteBalance"
                   , "meanYCbCrPositioning")

PhotoFeatures <- as.data.frame(PhotoFeatures) # 57,702
Photo.features = PhotoFeatures[, featureColumns]

Photo.features <- unique(Photo.features) # 9,226

length(Photo.features$listing_id)
length(unique(Photo.features$listing_id))


head(Photo.features)
Photo.features[grep("7170325|6811957",Photo.features$listing_id), ] #114,361

write.csv(Photo.features,  './input/Prav_ImageExif_01_features.csv', row.names=FALSE, quote = FALSE) #


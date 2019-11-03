
PhotoFeatures <- read_csv("./input/Image_Exif_02.csv")

head(PhotoFeatures)



PhotoFeatures$listing_id  <-  as.integer(lapply(strsplit(as.character(PhotoFeatures$imageNo), split="_"),head, n=1))

PhotoFeatures = as.data.table(PhotoFeatures)

unique(PhotoFeatures$Make)

StandardingString <- function(str) {
  str <- tolower(str)
  str = gsub("co.,","", str, ignore.case =  TRUE)
  str = gsub("co.","", str, ignore.case =  TRUE)
  str = gsub("ltd","", str, ignore.case =  TRUE)
  str = gsub("[.]","", str, ignore.case =  TRUE)
  str = gsub("\\\\","", str, ignore.case =  TRUE)
  str = gsub("\"canon\"","canon", str, ignore.case =  TRUE)
  str = gsub("samsung techwin","samsung", str, ignore.case =  TRUE)
  str = gsub("nikon poration","nikon", str, ignore.case =  TRUE)
  str = gsub("olympus imaging p","olympus", str, ignore.case =  TRUE)
  str = gsub("olympus optical ","olympus", str, ignore.case =  TRUE)
  str = gsub("olympus poration","olympus", str, ignore.case =  TRUE)
  str = gsub("konica minolta","konica", str, ignore.case =  TRUE)
  str = gsub("minolta  ","konica", str, ignore.case =  TRUE)
  
  str = gsub("^\\s+|\\s+$", "", str, ignore.case =  TRUE)
  return (str)
}

PhotoFeatures$Make <- StandardingString(PhotoFeatures$Make)

PhotoFeatures[,c(     "meanColorSpace"
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
                    , "maxDateTimeOriginal"
                    , "maxMake"
                    , "maxOrientation"
                    , "maxSaturation"
                    , "maxSceneCaptureType"
                    , "maxSensingMethod"
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
                            , max(DateTimeOriginal)
                            , max(Make)
                            , max(Orientation)
                            , max(Saturation)
                            , max(SceneCaptureType)
                            , max(SensingMethod)
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
                   , "meanYCbCrPositioning"
                   , "maxDateTimeOriginal"
                   , "maxMake"
                   , "maxOrientation"
                   , "maxSaturation"
                   , "maxSceneCaptureType"
                   , "maxSensingMethod"
                   )

PhotoFeatures <- as.data.frame(PhotoFeatures) # 57,702
Photo.features = PhotoFeatures[, featureColumns]

Photo.features <- unique(Photo.features) # 9,226

length(Photo.features$listing_id)
length(unique(Photo.features$listing_id))


CorrColumns = c( "meanColorSpace"
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
                   #, "maxDateTimeOriginal"
                   #, "maxMake"
                   , "maxOrientation"
                   , "maxSaturation"
                   , "maxSceneCaptureType"
                   , "maxSensingMethod"
)

cor(Photo.features[, CorrColumns])

head(Photo.features)
Photo.features[grep("7170325|6811957",Photo.features$listing_id), ] #114,361

write.csv(Photo.features,  './input/Prav_ImageExif_02_features.csv', row.names=FALSE, quote = FALSE) #


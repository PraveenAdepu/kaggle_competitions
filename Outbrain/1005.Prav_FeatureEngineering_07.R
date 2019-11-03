
# X_build   <- read_csv("./input/trainingSet5_20161215.csv")
# X_val     <- read_csv("./input/testingSet5_20161215.csv")
# 
# 
# event_features <- read_csv("./input/events_features.csv")
# 
# names(event_features)
# 
# eventCols <- c("display_id","traffic_source")
# 
# eventFeature <- event_features[, eventCols]
# 
# head(eventFeature)
# 
# eventFeature$traffic_source <- gsub("\\\r","", eventFeature$traffic_source)
# 
# build <- left_join(X_build, eventFeature, by = c("display_id"))
# val   <- left_join(X_val, eventFeature, by = c("display_id"))
# 
# build[is.na(build)] <- 0
# val[is.na(val)]     <- 0
# 
# names(val)
# 
# write_csv(build, "./input/trainingSet06.csv")
# write_csv(val, "./input/testingSet06.csv")
# 
# rm(build, val); gc()

# 
# X_build   <- read_csv("./input/trainingSet06_fold1to4.csv")
# X_val     <- read_csv("./input/trainingSet06_fold5.csv")
# 
# 
# event_features <- read_csv("./input/event_weekdayFeatures.csv")
# 
# names(event_features)
# 
# 
# build <- left_join(X_build, event_features, by = c("display_id"))
# val   <- left_join(X_val, event_features, by = c("display_id"))
# 
# build[is.na(build)] <- 0
# val[is.na(val)]     <- 0
# 
# names(val)
# 
# write_csv(build, "./input/trainingSet10_fold1to4.csv")
# write_csv(val, "./input/trainingSet10_fold5.csv")
# 
# rm(build, val); gc()

################################################################################################
# Change Log

# 05_03 + traffic_source       == 06 files
#    06 + weekday and weekflag == 11 files
#    11 + Lastcategory_id, Lasttopic_id, Lastevent_id == 12_old files (36 features)
#    12_old + Catconf, Entconf, topconf = 12 files (42 features)

#################################################################################################

X_build   <- read_csv("./input/trainingSet06_fold1to4.csv")
X_val     <- read_csv("./input/trainingSet06_fold5.csv")


event_features <- read_csv("./input/event_weekdayFeatures.csv")

event_features$weekflag <- ifelse(event_features$weekday == "Saturday" |event_features$weekday == "Sunday", "No", "Yes" )

unique(event_features$weekflag)
names(event_features)

build <- left_join(X_build, event_features, by = c("display_id"))
val   <- left_join(X_val, event_features, by = c("display_id"))

build[is.na(build)] <- 0
val[is.na(val)]     <- 0

names(val)
gc()
write_csv(build, "./input/trainingSet11_fold1to4.csv")
write_csv(val, "./input/trainingSet11_fold5.csv")

#################################################################################################

X_build   <- read_csv("./input/trainingSet06.csv")
X_val     <- read_csv("./input/testingSet06.csv")


event_features <- read_csv("./input/event_weekdayFeatures.csv")

event_features$weekflag <- ifelse(event_features$weekday == "Saturday" |event_features$weekday == "Sunday", "No", "Yes" )

unique(event_features$weekflag)
names(event_features)


build <- left_join(X_build, event_features, by = c("display_id"))
val   <- left_join(X_val, event_features, by = c("display_id"))

build[is.na(build)] <- 0
val[is.na(val)]     <- 0

names(val)

write_csv(build, "./input/trainingSet11.csv")
write_csv(val, "./input/testingSet11.csv")

rm(build, val); gc()

#################################################################################################
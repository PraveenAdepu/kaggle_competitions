
train.json <- fromJSON( "./input/train.json")
train.featurePhotos <- data.table( listing_id=unlist(train.json$listing_id)
                                   ,features=unlist(train.json$features) 
                                  
)

head(train.featurePhotos)


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

# Summarize count of features
feature = data.frame(feature = tolower(unlist(train.featurePhotos$features))) %>% # convert all features to lower case
  group_by(feature) %>%
  summarise(feature_count = n()) %>%
  arrange(desc(feature_count)) %>%
  filter(feature_count >= 50)

kable(feature, caption = "Feature Count")

# Hardwood
feature %>%
  filter(str_detect(feature, 'wood')) %>%
  kable(caption = "hardwood")

# Laundry in unit
feature %>%
  filter(str_detect(feature, paste(c('laundry', 'dryer', 'washer'), collapse="|"))) %>%
  filter(!str_detect(feature, "dishwasher")) %>%
  kable(caption = "Laundry in unit")

# Roof deck
feature %>%
  filter(str_detect(feature, 'roof')) %>%
  kable(caption = "Roof Deck")

# Outdoor space
feature %>%
  filter(str_detect(feature, 'outdoor')) %>%
  kable(caption = "Outdoor Space")

# Garden
feature %>%
  filter(str_detect(feature, 'garden')) %>%
  kable(caption = "Garden")

# Park
feature %>%
  filter(str_detect(feature, 'park')) %>%
  kable(caption = "Parking")


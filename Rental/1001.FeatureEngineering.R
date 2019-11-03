# Sys.time()
# load("Rental_01.RData") # 22 basic features from source files
# Sys.time()

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

all_data$address_similarity_jw          <- stringdist(all_data$display_address,all_data$street_adress,method='jw')
# all_data$address_similarity_cosine   <- stringdist(all_data$display_address,all_data$street_adress,method='cosine')
# all_data$address_similarity_jaccard  <- stringdist(all_data$display_address,all_data$street_adress,method='jaccard')
# chance for cleaning of numbers
all_data$address_similarity_sound    <- stringdist(all_data$display_address,all_data$street_adress,method='soundex')

train <- all_data[all_data$interest_level != "none",] #49,352
test  <- all_data[all_data$interest_level == "none",] #74,659

features <- c("listing_id","address_similarity_jw","address_similarity_sound")

head(train[,features])

write_csv(train[,features],"./input/train_features_00.csv")
write_csv(test[,features],"./input/test_features_00.csv")

########################################################################################################################


head(train$features)

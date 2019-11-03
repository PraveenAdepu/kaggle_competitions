
set.seed(2016)

######################################################################################################################
# 1. Get page view day
page_views_sample <- fread("./input/page_views.csv", select = c("timestamp") )
names(page_views_sample)

head(page_views_sample)

page_views_sample$day <- as.integer(page_views_sample$timestamp/ (3600 * 24 * 1000))

page_views_sample$timestamp <- NULL
gc()

######################################################################################################################

######################################################################################################################
# 1. Get page view day # 2,034,275,448
page_views <- fread("./input/page_views.csv", select = c("uuid","document_id","platform","geo_location","traffic_source") )
names(page_views)
head(page_views)

gc()

page_views <- cbind(page_views, page_views_sample)

gc()

length(unique(page_views$day))

rm(page_views_sample); gc()

######################################################################################################################


# events <- fread("./input/events.csv") 
# names(events)
# head(events)
# 
# events$day <- as.integer(as.integer(events$timestamp/ (3600 * 24 * 1000)))
# 
# events$platform <- as.integer(events$platform)
# gc()
# 23,120,126
######################################################################################################################



# events <- left_join(events, page_views, by = c("uuid","document_id","platform","geo_location","day"))
# gc()
# 
# head(events)


#mpExpenses2012 is the large dataframe containing data for each MP
#Get the list of unique MP names
for (name in unique(page_views$day)){
  #Subset the data by MP
  cat(name ," day Processing\n")
  
  tmp=subset(page_views,day==name)
  rowCount = nrow(tmp)
  cat(rowCount ," rows Processing\n")
  #Create a new filename for each MP - the folder 'mpExpenses2012' should already exist
  fn=paste(paste('./input/page_views_',gsub(' ','',name),sep=''),'.csv')
  #Save the CSV file containing separate expenses data for each MP
  write_csv(tmp,fn)
}


######################################################################################################################


events <- fread("./input/events.csv")
names(events)
head(events)

events$day <- as.integer(as.integer(events$timestamp/ (3600 * 24 * 1000)))

events$platform <- as.integer(events$platform)

events[is.na(events)] <- 0
gc()
# 23,120,126
######################################################################################################################



# events <- left_join(events, page_views, by = c("uuid","document_id","platform","geo_location","day"))
# gc()
# 
# head(events)


#mpExpenses2012 is the large dataframe containing data for each MP
#Get the list of unique MP names
TotalRows <- 0
for (name in unique(events$day)){
  #Subset the data by MP
  cat(name ," day Processing\n")
  
  tmp=subset(events,day==name)
  rowCount = nrow(tmp)
  TotalRows <- TotalRows + rowCount
  cat(rowCount ," rows Processing\n")
  cat(TotalRows ," Total rows Processing\n")
  #Create a new filename for each MP - the folder 'mpExpenses2012' should already exist
  fn=paste(paste('./input/events_',gsub(' ','',name),sep=''),'.csv')
  #Save the CSV file containing separate expenses data for each MP
  write_csv(tmp,fn)
}




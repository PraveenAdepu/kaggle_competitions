events <- fread("./input/events.csv") 
names(events)

head(events)

unique(events$weekday)

events$Date =  as.POSIXct((events$timestamp+1465876799998)/1000, origin="1970-01-01", tz="UTC")


events$weekday <- weekdays(as.Date(events$Date))


IntCols <- c("display_id","weekday")

events<- as.data.frame(events)
eventFeatures <- events[, IntCols]

write_csv(eventFeatures, "./input/event_weekdayFeatures.csv")
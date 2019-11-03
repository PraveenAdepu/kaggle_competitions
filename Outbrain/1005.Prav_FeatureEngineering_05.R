events_0     <- fread("./input/events0.csv")
page_views_0 <- fread("./input/page_views0.csv")

head(events_0)
head(page_views_0)
events_0 <- left_join(events_0, page_views_0, by = c("uuid","document_id","platform","geo_location","day"))

rm(page_views_0); gc()

##################################################################################################################

events_1     <- fread("./input/events1.csv")
page_views_1 <- fread("./input/page_views1.csv")

head(events_1)
head(page_views_1)
events_1 <- left_join(events_1, page_views_1, by = c("uuid","document_id","platform","geo_location","day"))

rm(page_views_1); gc()

##################################################################################################################

events_2     <- fread("./input/events2.csv")
page_views_2 <- fread("./input/page_views2.csv")

head(events_2)
head(page_views_2)
events_2 <- left_join(events_2, page_views_2, by = c("uuid","document_id","platform","geo_location","day"))

rm(page_views_2); gc()

##################################################################################################################

events_3     <- fread("./input/events3.csv")
page_views_3 <- fread("./input/page_views3.csv")

head(events_3)
head(page_views_3)
events_3 <- left_join(events_3, page_views_3, by = c("uuid","document_id","platform","geo_location","day"))

rm(page_views_3); gc()

##################################################################################################################

events_4     <- fread("./input/events4.csv")
page_views_4 <- fread("./input/page_views4.csv")

head(events_4)
head(page_views_4)
events_4 <- left_join(events_4, page_views_4, by = c("uuid","document_id","platform","geo_location","day"))

rm(page_views_4); gc()

##################################################################################################################

events_5     <- fread("./input/events5.csv")
page_views_5 <- fread("./input/page_views5.csv")

head(events_5)
head(page_views_5)
events_5 <- left_join(events_5, page_views_5, by = c("uuid","document_id","platform","geo_location","day"))

rm(page_views_5); gc()

##################################################################################################################

events_6     <- fread("./input/events6.csv")
page_views_6 <- fread("./input/page_views6.csv")

head(events_6)
head(page_views_6)
events_6 <- left_join(events_6, page_views_6, by = c("uuid","document_id","platform","geo_location","day"))

rm(page_views_6); gc()

##################################################################################################################

events_7     <- fread("./input/events7.csv")
page_views_7 <- fread("./input/page_views7.csv")

head(events_7)
head(page_views_7)
events_7 <- left_join(events_7, page_views_7, by = c("uuid","document_id","platform","geo_location","day"))

rm(page_views_7); gc()

##################################################################################################################

events_8     <- fread("./input/events8.csv")
page_views_8 <- fread("./input/page_views8.csv")

head(events_8)
head(page_views_8)
events_8 <- left_join(events_8, page_views_8, by = c("uuid","document_id","platform","geo_location","day"))

rm(page_views_8); gc()

##################################################################################################################

events_9     <- fread("./input/events9.csv")
page_views_9 <- fread("./input/page_views9.csv")

head(events_9)
head(page_views_9)
events_9 <- left_join(events_9, page_views_9, by = c("uuid","document_id","platform","geo_location","day"))

rm(page_views_9); gc()

##################################################################################################################

events_10     <- fread("./input/events10.csv")
page_views_10 <- fread("./input/page_views10.csv")

head(events_10)
head(page_views_10)
events_10 <- left_join(events_10, page_views_10, by = c("uuid","document_id","platform","geo_location","day"))

rm(page_views_10); gc()

##################################################################################################################

events_11     <- fread("./input/events11.csv")
page_views_11 <- fread("./input/page_views11.csv")

head(events_11)
head(page_views_11)
events_11 <- left_join(events_11, page_views_11, by = c("uuid","document_id","platform","geo_location","day"))

rm(page_views_11); gc()

##################################################################################################################

events_12     <- fread("./input/events12.csv")
page_views_12 <- fread("./input/page_views12.csv")

head(events_12)
head(page_views_12)
events_12 <- left_join(events_12, page_views_12, by = c("uuid","document_id","platform","geo_location","day"))

rm(page_views_12); gc()

##################################################################################################################

events_13     <- fread("./input/events13.csv")
page_views_13 <- fread("./input/page_views13.csv")

head(events_13)
head(page_views_13)
events_13 <- left_join(events_13, page_views_13, by = c("uuid","document_id","platform","geo_location","day"))

rm(page_views_13); gc()

##################################################################################################################

events_14     <- fread("./input/events14.csv")
page_views_14 <- fread("./input/page_views14.csv")

head(events_14)
head(page_views_14)
events_14 <- left_join(events_14, page_views_14, by = c("uuid","document_id","platform","geo_location","day"))

rm(page_views_14); gc()

##################################################################################################################

events_final <- rbind(events_0, events_1, events_2, events_3, events_4, events_5, events_6, events_7, events_8, events_9, events_10, events_11, events_12, events_13, events_14)


write_csv(events_final, "./input/events_features.csv")
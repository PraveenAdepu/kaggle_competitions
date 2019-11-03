
page_views_0 <- fread("./input/pageviewfeatures/page_views0.csv", select = c("geo_location","document_id"))


page_views_0[ , docfreq := .N, by = list(geo_location, document_id)]
setkeyv(page_views_0, c('geo_location','document_id'))
page_views_00 <- unique(page_views_0)


rm(page_views_0); gc()

##################################################################################################################


page_views_1 <- fread("./input/pageviewfeatures/page_views1.csv", select = c("geo_location","document_id"))
page_views_1[ , docfreq := .N, by = list(geo_location, document_id)]
setkeyv(page_views_1, c('geo_location','document_id'))
page_views_011 <- unique(page_views_1)

rm(page_views_1); gc()

##################################################################################################################


page_views_2 <- fread("./input/pageviewfeatures/page_views2.csv", select = c("geo_location","document_id"))
page_views_2[ , docfreq := .N, by = list(geo_location, document_id)]
setkeyv(page_views_2, c('geo_location','document_id'))
page_views_22 <- unique(page_views_2)

rm(page_views_2); gc()

##################################################################################################################


page_views_3 <- fread("./input/pageviewfeatures/page_views3.csv", select = c("geo_location","document_id"))
page_views_3[ , docfreq := .N, by = list(geo_location, document_id)]
setkeyv(page_views_3, c('geo_location','document_id'))
page_views_33 <- unique(page_views_3)

rm(page_views_3); gc()

##################################################################################################################

page_views_4 <- fread("./input/pageviewfeatures/page_views4.csv", select = c("geo_location","document_id"))
page_views_4[ , docfreq := .N, by = list(geo_location, document_id)]
setkeyv(page_views_4, c('geo_location','document_id'))
page_views_44 <- unique(page_views_4)

rm(page_views_4); gc()

##################################################################################################################
page_views_5 <- fread("./input/pageviewfeatures/page_views5.csv", select = c("geo_location","document_id"))
page_views_5[ , docfreq := .N, by = list(geo_location, document_id)]
setkeyv(page_views_5, c('geo_location','document_id'))
page_views_55 <- unique(page_views_5)

rm(page_views_5); gc()

##################################################################################################################

page_views_6 <- fread("./input/pageviewfeatures/page_views6.csv", select = c("geo_location","document_id"))
page_views_6[ , docfreq := .N, by = list(geo_location, document_id)]
setkeyv(page_views_6, c('geo_location','document_id'))
page_views_66 <- unique(page_views_6)

rm(page_views_6); gc()

##################################################################################################################

page_views_7 <- fread("./input/pageviewfeatures/page_views7.csv", select = c("geo_location","document_id"))
page_views_7[ , docfreq := .N, by = list(geo_location, document_id)]
setkeyv(page_views_7, c('geo_location','document_id'))
page_views_77 <- unique(page_views_7)

rm(page_views_7); gc()

##################################################################################################################

page_views_8 <- fread("./input/pageviewfeatures/page_views8.csv", select = c("geo_location","document_id"))
page_views_8[ , docfreq := .N, by = list(geo_location, document_id)]
setkeyv(page_views_8, c('geo_location','document_id'))
page_views_88 <- unique(page_views_8)

rm(page_views_8); gc()

##################################################################################################################

page_views_9 <- fread("./input/pageviewfeatures/page_views9.csv", select = c("geo_location","document_id"))
page_views_9[ , docfreq := .N, by = list(geo_location, document_id)]
setkeyv(page_views_9, c('geo_location','document_id'))
page_views_99 <- unique(page_views_9)

rm(page_views_9); gc()

##################################################################################################################

page_views_10 <- fread("./input/pageviewfeatures/page_views10.csv", select = c("geo_location","document_id"))
page_views_10[ , docfreq := .N, by = list(geo_location, document_id)]
setkeyv(page_views_10, c('geo_location','document_id'))
page_views_1010 <- unique(page_views_10)

rm(page_views_10); gc()

##################################################################################################################

page_views_11 <- fread("./input/pageviewfeatures/page_views11.csv", select = c("geo_location","document_id"))
page_views_11[ , docfreq := .N, by = list(geo_location, document_id)]
setkeyv(page_views_11, c('geo_location','document_id'))
page_views_1111 <- unique(page_views_11)

rm(page_views_11); gc()

##################################################################################################################

page_views_12 <- fread("./input/pageviewfeatures/page_views12.csv", select = c("geo_location","document_id"))
page_views_12[ , docfreq := .N, by = list(geo_location, document_id)]
setkeyv(page_views_12, c('geo_location','document_id'))
page_views_1212 <- unique(page_views_12)

rm(page_views_12); gc()

##################################################################################################################

page_views_13 <- fread("./input/pageviewfeatures/page_views13.csv", select = c("geo_location","document_id"))
page_views_13[ , docfreq := .N, by = list(geo_location, document_id)]
setkeyv(page_views_13, c('geo_location','document_id'))
page_views_1313 <- unique(page_views_13)

rm(page_views_13); gc()

##################################################################################################################

page_views_14 <- fread("./input/pageviewfeatures/page_views14.csv", select = c("geo_location","document_id"))
page_views_14[ , docfreq := .N, by = list(geo_location, document_id)]
setkeyv(page_views_14, c('geo_location','document_id'))
page_views_1414 <- unique(page_views_14)

rm(page_views_14); gc()

##################################################################################################################
# 29060291


page_views <- rbind(page_views_00,page_views_011,page_views_22,page_views_33,page_views_44,page_views_55,page_views_66,page_views_77,page_views_88,page_views_99,page_views_1010,page_views_1111,page_views_1212,page_views_1313,page_views_1414); gc()

rm(page_views_00,page_views_011,page_views_22,page_views_33,page_views_44,page_views_55,page_views_66,page_views_77,page_views_88,page_views_99,page_views_1010,page_views_1111,page_views_1212,page_views_1313,page_views_1414); gc()

tail(page_views,50)


# 2034275446
system.time(page_views[, sum(docfreq), by = list(geo_location, document_id)])


setDT(page_views)[, paste0("location", 1:3) := tstrsplit(geo_location, ">")]

page_views$geo_location <- NULL

page_views[is.na(page_views)] <- 0

write_csv(page_views, "./input/pageviews_docgeolocation.csv")
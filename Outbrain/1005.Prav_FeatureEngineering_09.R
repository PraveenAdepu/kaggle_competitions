
documents_categories <- fread("./input/documents_categories.csv")
names(documents_categories)

documents_entities <- fread("./input/documents_entities.csv")
names(documents_entities)


documents_topics <- fread("./input/documents_topics.csv")
names(documents_topics)

head(documents_categories)

documents_categories_rank <- documents_categories %>%
  group_by(document_id) %>%
  mutate(category_rank = order(confidence_level,  decreasing=TRUE))

head(documents_categories_rank)

documents_categories_Toprank <- subset(documents_categories_rank , category_rank == 1)

length(unique(documents_categories$document_id))

head(documents_entities)
documents_entities_rank <- documents_entities %>%
  group_by(document_id) %>%
  mutate(entities_rank = order(confidence_level,  decreasing=TRUE))

head(documents_entities_rank)

documents_entities_Toprank <- subset(documents_entities_rank , entities_rank == 1)

length(unique(documents_entities$document_id))

head(documents_topics)
documents_topics_rank <- documents_topics %>%
  group_by(document_id) %>%
  mutate(topics_rank = order(confidence_level,  decreasing=TRUE))

head(documents_topics_rank)

documents_topics_Toprank <- subset(documents_topics_rank , topics_rank == 1)

length(unique(documents_topics$document_id))

rm(documents_categories, documents_categories_rank, documents_entities, documents_entities_rank
   , documents_topics, documents_topics_rank); gc()

head(documents_topics_Toprank)

documents_categories_Toprank$category_rank    <- NULL
documents_categories_Toprank$category_id <- NULL

documents_entities_Toprank$entities_rank    <- NULL
documents_entities_Toprank$entity_id <- NULL

documents_topics_Toprank$topics_rank      <- NULL
documents_topics_Toprank$topic_id <- NULL

write_csv(documents_categories_Toprank, './input/categories_topconfidence.csv')
write_csv(documents_entities_Toprank, './input/entities_topconfidence.csv')
write_csv(documents_topics_Toprank, './input/topics_topconfidence.csv')

gc()

#######################################################################################################


X_build <- read_csv( "./input/trainingSet12_fold1to4.csv")
X_val   <- read_csv( "./input/trainingSet12_fold5.csv")

names(X_build)
names(documents_categories_Toprank)
names(documents_categories_Toprank)[1] <- "event_document_id"
names(documents_categories_Toprank)[2] <- "event_Catconf"
names(documents_entities_Toprank)
names(documents_entities_Toprank)[1] <- "event_document_id"
names(documents_entities_Toprank)[2] <- "event_Entconf"

names(documents_topics_Toprank)
names(documents_topics_Toprank)[1] <- "event_document_id"
names(documents_topics_Toprank)[2] <- "event_topconf"



build <- left_join(X_build, documents_categories_Toprank, by = "event_document_id")
build <- left_join(build, documents_entities_Toprank, by = "event_document_id")
build <- left_join(build, documents_topics_Toprank, by = "event_document_id")


val <- left_join(X_val, documents_categories_Toprank, by = "event_document_id")
val <- left_join(val, documents_entities_Toprank, by = "event_document_id")
val <- left_join(val, documents_topics_Toprank, by = "event_document_id")


names(documents_categories_Toprank)
names(documents_categories_Toprank)[1] <- "document_id"
names(documents_categories_Toprank)[2] <- "Catconf"
names(documents_entities_Toprank)
names(documents_entities_Toprank)[1] <- "document_id"
names(documents_entities_Toprank)[2] <- "Entconf"

names(documents_topics_Toprank)
names(documents_topics_Toprank)[1] <- "document_id"
names(documents_topics_Toprank)[2] <- "topconf"

build <- left_join(build, documents_categories_Toprank, by = "document_id")
build <- left_join(build, documents_entities_Toprank, by = "document_id")
build <- left_join(build, documents_topics_Toprank, by = "document_id")


val <- left_join(val, documents_categories_Toprank, by = "document_id")
val <- left_join(val, documents_entities_Toprank, by = "document_id")
val <- left_join(val, documents_topics_Toprank, by = "document_id")

names(build)

build[is.na(build)] <- 0
val[is.na(val)]     <- 0

write_csv(build, "./input/trainingSet12_fold1to4.csv")
write_csv(val, "./input/trainingSet12_fold5.csv")


#########################################################################################


X_build   <- read_csv("./input/trainingSet11.csv")
X_val     <- read_csv("./input/testingSet11.csv")

categories_Bottomrank <- read_csv( "./input/documents_categories_Bottomrank.csv")
topics_Bottomrank     <- read_csv( "./input/documents_topics_Bottomrank.csv")


names(X_build)
names(categories_Bottomrank)
names(categories_Bottomrank)[1] <- "event_document_id"
names(categories_Bottomrank)[2] <- "event_LastCat_id"
names(topics_Bottomrank)
names(topics_Bottomrank)[1] <- "event_document_id"
names(topics_Bottomrank)[2] <- "event_Lasttopic_id"

build <- left_join(X_build, categories_Bottomrank, by = "event_document_id")
build <- left_join(build, topics_Bottomrank, by = "event_document_id")

val <- left_join(X_val, categories_Bottomrank, by = "event_document_id")
val <- left_join(val, topics_Bottomrank, by = "event_document_id")



names(categories_Bottomrank)
names(categories_Bottomrank)[1] <- "document_id"
names(categories_Bottomrank)[2] <- "LastCat_id"
names(topics_Bottomrank)
names(topics_Bottomrank)[1] <- "document_id"
names(topics_Bottomrank)[2] <- "Lasttopic_id"


build <- left_join(build, categories_Bottomrank, by = "document_id")
build <- left_join(build, topics_Bottomrank, by = "document_id")

val <- left_join(val, categories_Bottomrank, by = "document_id")
val <- left_join(val, topics_Bottomrank, by = "document_id")
#######################################################################

documents_categories_Toprank <- read_csv( "./input/topics_topconfidence.csv")
documents_entities_Toprank     <- read_csv( "./input/entities_topconfidence.csv")
documents_topics_Toprank     <- read_csv( "./input/topics_topconfidence.csv")


names(X_build)
names(documents_categories_Toprank)
names(documents_categories_Toprank)[1] <- "event_document_id"
names(documents_categories_Toprank)[2] <- "event_Catconf"
names(documents_entities_Toprank)
names(documents_entities_Toprank)[1] <- "event_document_id"
names(documents_entities_Toprank)[2] <- "event_Entconf"

names(documents_topics_Toprank)
names(documents_topics_Toprank)[1] <- "event_document_id"
names(documents_topics_Toprank)[2] <- "event_topconf"



build <- left_join(build, documents_categories_Toprank, by = "event_document_id")
build <- left_join(build, documents_entities_Toprank, by = "event_document_id")
build <- left_join(build, documents_topics_Toprank, by = "event_document_id")


val <- left_join(val, documents_categories_Toprank, by = "event_document_id")
val <- left_join(val, documents_entities_Toprank, by = "event_document_id")
val <- left_join(val, documents_topics_Toprank, by = "event_document_id")


names(documents_categories_Toprank)
names(documents_categories_Toprank)[1] <- "document_id"
names(documents_categories_Toprank)[2] <- "Catconf"
names(documents_entities_Toprank)
names(documents_entities_Toprank)[1] <- "document_id"
names(documents_entities_Toprank)[2] <- "Entconf"

names(documents_topics_Toprank)
names(documents_topics_Toprank)[1] <- "document_id"
names(documents_topics_Toprank)[2] <- "topconf"

build <- left_join(build, documents_categories_Toprank, by = "document_id")
build <- left_join(build, documents_entities_Toprank, by = "document_id")
build <- left_join(build, documents_topics_Toprank, by = "document_id")


val <- left_join(val, documents_categories_Toprank, by = "document_id")
val <- left_join(val, documents_entities_Toprank, by = "document_id")
val <- left_join(val, documents_topics_Toprank, by = "document_id")
##########################################################################################
build[is.na(build)] <- 0
val[is.na(val)]     <- 0

names(val)

head(build)
write_csv(build, "./input/trainingSet12.csv")
write_csv(val, "./input/testingSet12.csv")

rm(build, val); gc()

##########################################################################################


X_build <- read_csv( "./input/trainingSet12_fold1to4.csv")
X_val   <- read_csv( "./input/trainingSet12_fold5.csv")

names(X_build)
length(unique(as.integer(X_build$event_publish_dateToDate/90)))
length(unique(as.integer(X_build$publish_dateToDate/90)))
length(unique(as.integer(X_build$event_publish_dateTopublishdate/90)))

X_build$event_publish_dateToDate         <- as.integer(X_build$event_publish_dateToDate/90)
X_build$publish_dateToDate               <- as.integer(X_build$publish_dateToDate/90)
X_build$event_publish_dateTopublishdate  <- as.integer(X_build$event_publish_dateTopublishdate/90)


X_val$event_publish_dateToDate         <- as.integer(X_val$event_publish_dateToDate/90)
X_val$publish_dateToDate               <- as.integer(X_val$publish_dateToDate/90)
X_val$event_publish_dateTopublishdate  <- as.integer(X_val$event_publish_dateTopublishdate/90)

length(unique(X_build$event_publish_dateToDate))

write_csv(X_build, "./input/trainingSet13_fold1to4.csv")
write_csv(X_val, "./input/trainingSet13_fold5.csv")


########################################################################################

X_build <- read_csv( "./input/trainingSet12.csv")
X_val   <- read_csv( "./input/testingSet12.csv")

names(X_build)
length(unique(as.integer(X_build$event_publish_dateToDate/90)))
length(unique(as.integer(X_build$publish_dateToDate/90)))
length(unique(as.integer(X_build$event_publish_dateTopublishdate/90)))

X_build$event_publish_dateToDate         <- as.integer(X_build$event_publish_dateToDate/90)
X_build$publish_dateToDate               <- as.integer(X_build$publish_dateToDate/90)
X_build$event_publish_dateTopublishdate  <- as.integer(X_build$event_publish_dateTopublishdate/90)


X_val$event_publish_dateToDate         <- as.integer(X_val$event_publish_dateToDate/90)
X_val$publish_dateToDate               <- as.integer(X_val$publish_dateToDate/90)
X_val$event_publish_dateTopublishdate  <- as.integer(X_val$event_publish_dateTopublishdate/90)

length(unique(X_build$event_publish_dateToDate))

write_csv(X_build, "./input/trainingSet13.csv")
write_csv(X_val, "./input/testingSet13.csv")






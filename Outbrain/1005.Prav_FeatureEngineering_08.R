set.seed(2016)


documents_categories <- fread("./input/documents_categories.csv")
names(documents_categories)

documents_entities <- fread("./input/documents_entities.csv")
names(documents_entities)


documents_topics <- fread("./input/documents_topics.csv")
names(documents_topics)

head(documents_categories)

documents_categories_rank <- documents_categories %>%
  group_by(document_id) %>%
  mutate(category_rank = order(confidence_level,  decreasing=FALSE))

head(documents_categories_rank)

documents_categories_Bottomrank <- subset(documents_categories_rank , category_rank == 1)

length(unique(documents_categories$document_id))

head(documents_entities)
documents_entities_rank <- documents_entities %>%
  group_by(document_id) %>%
  mutate(entities_rank = order(confidence_level,  decreasing=FALSE))

head(documents_entities_rank)

documents_entities_Bottomrank <- subset(documents_entities_rank , entities_rank == 1)

length(unique(documents_entities$document_id))

head(documents_topics)
documents_topics_rank <- documents_topics %>%
  group_by(document_id) %>%
  mutate(topics_rank = order(confidence_level,  decreasing=FALSE))

head(documents_topics_rank)

documents_topics_Bottomrank <- subset(documents_topics_rank , topics_rank == 1)

length(unique(documents_topics$document_id))

rm(documents_categories, documents_categories_rank, documents_entities, documents_entities_rank
   , documents_topics, documents_topics_rank); gc()

documents_categories_Bottomrank$category_rank    <- NULL
documents_categories_Bottomrank$confidence_level <- NULL

documents_entities_Bottomrank$entities_rank    <- NULL
documents_entities_Bottomrank$confidence_level <- NULL

documents_topics_Bottomrank$topics_rank      <- NULL
documents_topics_Bottomrank$confidence_level <- NULL

gc()

head(documents_categories_Bottomrank)
colnames(documents_categories_Bottomrank)[which(names(documents_categories_Bottomrank) == "category_id")]  <- "Lastcategory_id"

head(documents_entities_Bottomrank)
colnames(documents_entities_Bottomrank)[which(names(documents_entities_Bottomrank) == "entity_id")]  <- "Lastentity_id"

head(documents_topics_Bottomrank)
colnames(documents_topics_Bottomrank)[which(names(documents_topics_Bottomrank) == "topic_id")]  <- "Lastopic_id"

write_csv(documents_categories_Bottomrank, "./input/documents_categories_Bottomrank.csv")
write_csv(documents_entities_Bottomrank, "./input/documents_entities_Bottomrank.csv")
write_csv(documents_topics_Bottomrank, "./input/documents_topics_Bottomrank.csv")
#####################################################################################################################################

X_build <- read_csv( "./input/trainingSet11_fold1to4.csv")
X_val   <- read_csv( "./input/trainingSet11_fold5.csv")

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

names(build)

build[is.na(build)] <- 0
val[is.na(val)]     <- 0

write_csv(build, "./input/trainingSet12_fold1to4.csv")
write_csv(val, "./input/trainingSet12_fold5.csv")

###############################################################################################################

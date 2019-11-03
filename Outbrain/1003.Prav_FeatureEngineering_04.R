
set.seed(2016)


train  <- fread("./input/clicks_train.csv") 
names(train)

test  <- fread("./input/clicks_test.csv") 
names(test)

events <- fread("./input/events.csv") 
names(events)
promoted_content <- fread("./input/promoted_content.csv") 
names(promoted_content)

documents_meta <- fread("./input/documents_meta.csv") 
names(documents_meta)

documents_meta$publish_date <- as.Date(documents_meta$publish_time)


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

documents_categories_Toprank$category_rank    <- NULL
documents_categories_Toprank$confidence_level <- NULL

documents_entities_Toprank$entities_rank    <- NULL
documents_entities_Toprank$confidence_level <- NULL

documents_topics_Toprank$topics_rank      <- NULL
documents_topics_Toprank$confidence_level <- NULL

gc()




head(train)
head(test)
head(events)
head(promoted_content)
head(documents_meta)
# head(documents_categories) documents are many to many relations
# head(documents_entities)   documents are many to many relations
# head(documents_topics)     documents are many to many relations



documents_meta$publish_time <- NULL
Prav_CVindices_5folds  <- fread("./CVSchema/Prav_CVindices_5folds.csv") 
names(Prav_CVindices_5folds)

# train -- 87,141,731
train <- left_join(train, Prav_CVindices_5folds, by = "display_id")
names(train)
train$day <- NULL

test$clicked   <- 0
test$CVindices <- 100

train_test <- rbind(train, test)

head(train_test,2)


# 119,366,893
train_test <- left_join(train_test, events, by = "display_id")

train_test <- left_join(train_test, documents_meta, by = "document_id")
train_test <- left_join(train_test, documents_categories_Toprank, by = "document_id")
train_test <- left_join(train_test, documents_entities_Toprank, by = "document_id")
train_test <- left_join(train_test, documents_topics_Toprank, by = "document_id")

colnames(train_test)[which(names(train_test) == "document_id")]  <- "event_document_id"
colnames(train_test)[which(names(train_test) == "source_id")]    <- "event_source_id"
colnames(train_test)[which(names(train_test) == "publisher_id")] <- "event_publisher_id"
colnames(train_test)[which(names(train_test) == "publish_date")] <- "event_publish_date"
colnames(train_test)[which(names(train_test) == "category_id")]  <- "event_category_id"
colnames(train_test)[which(names(train_test) == "entity_id")]    <- "event_entity_id"
colnames(train_test)[which(names(train_test) == "topic_id")]     <- "event_topic_id"



train_test <- left_join(train_test, promoted_content, by = c("ad_id"))
train_test <- left_join(train_test, documents_meta, by = "document_id")
train_test <- left_join(train_test, documents_categories_Toprank, by = "document_id")
train_test <- left_join(train_test, documents_entities_Toprank, by = "document_id")
train_test <- left_join(train_test, documents_topics_Toprank, by = "document_id")

head(train_test,20)

#######################################################################################################################

setDT(train_test)[, paste0("location", 1:3) := tstrsplit(geo_location, ">")]


train_test$Date =  as.POSIXct((train_test$timestamp+1465876799998)/1000, origin="1970-01-01", tz="UTC")

t.lub <- ymd_hms(train_test$Date)

h.lub <- hour(t.lub) + minute(t.lub)/60
d.lub <- day(t.lub)
m.lub <- minute(t.lub)/60

head(h.lub)
head(d.lub)
head(m.lub)

train_test$hour    <- h.lub
train_test$day     <- d.lub
train_test$minutes <- m.lub

train_test$day <- as.integer(train_test$day)



train_test$platform           <- as.numeric(train_test$platform)


train_test$geo_location <- NULL
train_test$timestamp    <- NULL


train_test$event_publish_dateToDate <- difftime(train_test$Date ,train_test$event_publish_date , units = c("days")) 
train_test$publish_dateToDate <- difftime(train_test$Date ,train_test$publish_date , units = c("days"))
train_test$event_publish_dateTopublishdate <- difftime(train_test$event_publish_date ,train_test$publish_date , units = c("days")) 

train_test$publish_date        <- NULL
train_test$Date                <- NULL
train_test$event_publish_date  <- NULL

train_test$event_publish_dateToDate        <- as.integer(gsub(" days","", train_test$event_publish_dateToDate))
train_test$publish_dateToDate              <- as.integer(gsub(" days","", train_test$publish_dateToDate))
train_test$event_publish_dateTopublishdate <- as.integer(gsub(" days","", train_test$event_publish_dateTopublishdate))


train_test[is.na(train_test)] <- 0

names(train_test)
head(train_test,20)

#########################################################################################################################

training <- train_test[train_test$CVindices != 100,]
testing  <- train_test[train_test$CVindices == 100,]

rm(train,test, promoted_content, documents_categories_Toprank, documents_entities_Toprank, documents_meta, documents_topics_Toprank, events, Prav_CVindices_5folds); gc()


testing$clicked   <- NULL
testing$CVindices <- NULL

#########################################################################################################################  
trainLeak <- read_csv("./input/trainingSet_Leak.csv")
testLeak <- read_csv("./input/testingSet_Leak.csv")

trainLeak$leak <- gsub("\\[|\\'|\\]","", trainLeak$leak)
testLeak$leak <- gsub("\\[|\\'|\\]","", testLeak$leak)

unique(trainLeak$leak)
unique(testLeak$leak)

training <- left_join(training, trainLeak, by = c("display_id","ad_id"))
testing  <- left_join(testing, testLeak, by = c("display_id","ad_id"))

unique(training$leak)
unique(testing$leak)

#########################################################################################################################  

names(training)
names(testing)

################################################################################################

# Sys.time()
# save.image(file = "Outbrain_Baseline01_20161216.RData" , safe = TRUE)
# Sys.time()

# Sys.time()
# load("Outbrain_Baseline01_20161216.RData"); gc()
# Sys.time()

################################################################################################


i = 5

X_build <- subset(training, CVindices != i, select = -c( CVindices))
X_val   <- subset(training, CVindices == i, select = -c( CVindices)) 

names(X_build)

write_csv(X_build,  './input/trainingSet05_fold1to4.csv')
write_csv(X_val, './input/trainingSet05_fold5.csv')


training$CVindices <- NULL


gc()

names(training)

write_csv(training,  './input/trainingSet5_20161215.csv')
write_csv(testing,  './input/testingSet5_20161215.csv')


head(training)




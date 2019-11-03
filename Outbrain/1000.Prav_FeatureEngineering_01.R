
set.seed(2016)


train  <- fread("./input/clicks_train.csv") 
names(train)
gc()

views  <- fread("./input/page_views.csv")
names(views)
head(views)
gc()

events <- fread("./input/events.csv") 
names(events)
head(events)
gc()


################################################################################################

 Sys.time()
 save.image(file = "Outbrain_Baseline01_20161211.RData" , safe = TRUE)
 Sys.time()

################################################################################################

# Sys.time()
# load("Allstate_Baseline01_20161014.RData"); gc()
# Sys.time()

################################################################################################
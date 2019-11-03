setwd("C:/Users/SriPrav/Documents/R/27Planet")
root_directory = "C:/Users/SriPrav/Documents/R/27Planet"

# paste(root_directory, "/input/events.csv", sep='')

# rm(list=ls())

require(data.table)
require(Matrix)
require(sqldf)
require(plyr)
require(dplyr)
require(ROCR)
require(Metrics)
require(pROC)
require(caret)
require(readr)

##################################################################################################

model01 <- read_csv("./submissions/Prav.dense169_02.fold10.csv") 
model02 <- read_csv("./submissions/Prav.Inception4_01.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (all_ensemble$slash_burn.x+all_ensemble$slash_burn.y)/2
all_ensemble$clear                <- (all_ensemble$clear.x+all_ensemble$clear.y)/2
all_ensemble$blooming             <- (all_ensemble$blooming.x+all_ensemble$blooming.y)/2
all_ensemble$primary              <- (all_ensemble$primary.x+all_ensemble$primary.y)/2
all_ensemble$cloudy               <- (all_ensemble$cloudy.x+all_ensemble$cloudy.y)/2
all_ensemble$conventional_mine    <- (all_ensemble$conventional_mine.x+all_ensemble$conventional_mine.y)/2
all_ensemble$water                <- (all_ensemble$water.x+all_ensemble$water.y)/2
all_ensemble$haze                 <- (all_ensemble$haze.x+all_ensemble$haze.y)/2
all_ensemble$cultivation          <- (all_ensemble$cultivation.x+all_ensemble$cultivation.y)/2
all_ensemble$partly_cloudy        <- (all_ensemble$partly_cloudy.x+all_ensemble$partly_cloudy.y)/2
all_ensemble$artisinal_mine       <- (all_ensemble$artisinal_mine.x+all_ensemble$artisinal_mine.y)/2
all_ensemble$habitation           <- (all_ensemble$habitation.x+all_ensemble$habitation.y)/2
all_ensemble$bare_ground          <- (all_ensemble$bare_ground.x+all_ensemble$bare_ground.y)/2
all_ensemble$blow_down            <- (all_ensemble$blow_down.x+all_ensemble$blow_down.y)/2
all_ensemble$agriculture          <- (all_ensemble$agriculture.x+all_ensemble$agriculture.y)/2
all_ensemble$road                 <- (all_ensemble$road.x+all_ensemble$road.y)/2
all_ensemble$selective_logging    <- (all_ensemble$selective_logging.x+all_ensemble$selective_logging.y)/2

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense169_02_Inception4_01_ensemble.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################


model01 <- read_csv("./submissions/Prav.dense169_02.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.Inception4_01.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (all_ensemble$slash_burn.x+all_ensemble$slash_burn.y)/2
all_ensemble$clear                <- (all_ensemble$clear.x+all_ensemble$clear.y)/2
all_ensemble$blooming             <- (all_ensemble$blooming.x+all_ensemble$blooming.y)/2
all_ensemble$primary              <- (all_ensemble$primary.x+all_ensemble$primary.y)/2
all_ensemble$cloudy               <- (all_ensemble$cloudy.x+all_ensemble$cloudy.y)/2
all_ensemble$conventional_mine    <- (all_ensemble$conventional_mine.x+all_ensemble$conventional_mine.y)/2
all_ensemble$water                <- (all_ensemble$water.x+all_ensemble$water.y)/2
all_ensemble$haze                 <- (all_ensemble$haze.x+all_ensemble$haze.y)/2
all_ensemble$cultivation          <- (all_ensemble$cultivation.x+all_ensemble$cultivation.y)/2
all_ensemble$partly_cloudy        <- (all_ensemble$partly_cloudy.x+all_ensemble$partly_cloudy.y)/2
all_ensemble$artisinal_mine       <- (all_ensemble$artisinal_mine.x+all_ensemble$artisinal_mine.y)/2
all_ensemble$habitation           <- (all_ensemble$habitation.x+all_ensemble$habitation.y)/2
all_ensemble$bare_ground          <- (all_ensemble$bare_ground.x+all_ensemble$bare_ground.y)/2
all_ensemble$blow_down            <- (all_ensemble$blow_down.x+all_ensemble$blow_down.y)/2
all_ensemble$agriculture          <- (all_ensemble$agriculture.x+all_ensemble$agriculture.y)/2
all_ensemble$road                 <- (all_ensemble$road.x+all_ensemble$road.y)/2
all_ensemble$selective_logging    <- (all_ensemble$selective_logging.x+all_ensemble$selective_logging.y)/2

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense169_02_Inception4_01_ensemble.fold10-test.csv', row.names=FALSE, quote = FALSE)







##################################################################################################

# Cross validation fold 
##################################################################################################

model01 <- read_csv("./submissions/Prav.dense169_02.fold10.csv") 
model02 <- read_csv("./submissions/Prav.Inception4_01.fold10.csv")
model03 <- read_csv("./submissions/Prav.dense161_01.fold10.csv") 

all_ensemble <- left_join(model01, model02, by = "image_name")
all_ensemble <- left_join(all_ensemble, model03, by = "image_name")
ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(model03)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.4 * all_ensemble$slash_burn.x       +0.3*all_ensemble$slash_burn.y       +0.3 * all_ensemble$slash_burn)
all_ensemble$clear                <- (0.4 * all_ensemble$clear.x            +0.3*all_ensemble$clear.y            +0.3 * all_ensemble$clear)
all_ensemble$blooming             <- (0.4 * all_ensemble$blooming.x         +0.3*all_ensemble$blooming.y         +0.3 * all_ensemble$blooming)
all_ensemble$primary              <- (0.4 * all_ensemble$primary.x          +0.3*all_ensemble$primary.y          +0.3 * all_ensemble$primary)
all_ensemble$cloudy               <- (0.4 * all_ensemble$cloudy.x           +0.3*all_ensemble$cloudy.y           +0.3 * all_ensemble$cloudy)
all_ensemble$conventional_mine    <- (0.4 * all_ensemble$conventional_mine.x+0.3*all_ensemble$conventional_mine.y+0.3 * all_ensemble$conventional_mine)
all_ensemble$water                <- (0.4 * all_ensemble$water.x            +0.3*all_ensemble$water.y            +0.3 * all_ensemble$water)
all_ensemble$haze                 <- (0.4 * all_ensemble$haze.x             +0.3*all_ensemble$haze.y             +0.3 * all_ensemble$haze)
all_ensemble$cultivation          <- (0.4 * all_ensemble$cultivation.x      +0.3*all_ensemble$cultivation.y      +0.3 * all_ensemble$cultivation)
all_ensemble$partly_cloudy        <- (0.4 * all_ensemble$partly_cloudy.x    +0.3*all_ensemble$partly_cloudy.y    +0.3 * all_ensemble$partly_cloudy)
all_ensemble$artisinal_mine       <- (0.4 * all_ensemble$artisinal_mine.x   +0.3*all_ensemble$artisinal_mine.y   +0.3 * all_ensemble$artisinal_mine)
all_ensemble$habitation           <- (0.4 * all_ensemble$habitation.x       +0.3*all_ensemble$habitation.y       +0.3 * all_ensemble$habitation)
all_ensemble$bare_ground          <- (0.4 * all_ensemble$bare_ground.x      +0.3*all_ensemble$bare_ground.y      +0.3 * all_ensemble$bare_ground)
all_ensemble$blow_down            <- (0.4 * all_ensemble$blow_down.x        +0.3*all_ensemble$blow_down.y        +0.3 * all_ensemble$blow_down)
all_ensemble$agriculture          <- (0.4 * all_ensemble$agriculture.x      +0.3*all_ensemble$agriculture.y      +0.3 * all_ensemble$agriculture)
all_ensemble$road                 <- (0.4 * all_ensemble$road.x             +0.3*all_ensemble$road.y             +0.3 * all_ensemble$road)
all_ensemble$selective_logging    <- (0.4 * all_ensemble$selective_logging.x+0.3*all_ensemble$selective_logging.y+0.3 * all_ensemble$selective_logging)

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense169_02_Inception4_01_dense161_01_ensemble.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################
############################################################################################################################


model01 <- read_csv("./submissions/Prav.dense169_02.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.Inception4_01.fold10-test.csv")
model03 <- read_csv("./submissions/Prav.dense161_01.fold10-test.csv") 

all_ensemble <- left_join(model01, model02, by = "image_name")
all_ensemble <- left_join(all_ensemble, model03, by = "image_name")
ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(model03)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.4 * all_ensemble$slash_burn.x       +0.3*all_ensemble$slash_burn.y       +0.3 * all_ensemble$slash_burn)
all_ensemble$clear                <- (0.4 * all_ensemble$clear.x            +0.3*all_ensemble$clear.y            +0.3 * all_ensemble$clear)
all_ensemble$blooming             <- (0.4 * all_ensemble$blooming.x         +0.3*all_ensemble$blooming.y         +0.3 * all_ensemble$blooming)
all_ensemble$primary              <- (0.4 * all_ensemble$primary.x          +0.3*all_ensemble$primary.y          +0.3 * all_ensemble$primary)
all_ensemble$cloudy               <- (0.4 * all_ensemble$cloudy.x           +0.3*all_ensemble$cloudy.y           +0.3 * all_ensemble$cloudy)
all_ensemble$conventional_mine    <- (0.4 * all_ensemble$conventional_mine.x+0.3*all_ensemble$conventional_mine.y+0.3 * all_ensemble$conventional_mine)
all_ensemble$water                <- (0.4 * all_ensemble$water.x            +0.3*all_ensemble$water.y            +0.3 * all_ensemble$water)
all_ensemble$haze                 <- (0.4 * all_ensemble$haze.x             +0.3*all_ensemble$haze.y             +0.3 * all_ensemble$haze)
all_ensemble$cultivation          <- (0.4 * all_ensemble$cultivation.x      +0.3*all_ensemble$cultivation.y      +0.3 * all_ensemble$cultivation)
all_ensemble$partly_cloudy        <- (0.4 * all_ensemble$partly_cloudy.x    +0.3*all_ensemble$partly_cloudy.y    +0.3 * all_ensemble$partly_cloudy)
all_ensemble$artisinal_mine       <- (0.4 * all_ensemble$artisinal_mine.x   +0.3*all_ensemble$artisinal_mine.y   +0.3 * all_ensemble$artisinal_mine)
all_ensemble$habitation           <- (0.4 * all_ensemble$habitation.x       +0.3*all_ensemble$habitation.y       +0.3 * all_ensemble$habitation)
all_ensemble$bare_ground          <- (0.4 * all_ensemble$bare_ground.x      +0.3*all_ensemble$bare_ground.y      +0.3 * all_ensemble$bare_ground)
all_ensemble$blow_down            <- (0.4 * all_ensemble$blow_down.x        +0.3*all_ensemble$blow_down.y        +0.3 * all_ensemble$blow_down)
all_ensemble$agriculture          <- (0.4 * all_ensemble$agriculture.x      +0.3*all_ensemble$agriculture.y      +0.3 * all_ensemble$agriculture)
all_ensemble$road                 <- (0.4 * all_ensemble$road.x             +0.3*all_ensemble$road.y             +0.3 * all_ensemble$road)
all_ensemble$selective_logging    <- (0.4 * all_ensemble$selective_logging.x+0.3*all_ensemble$selective_logging.y+0.3 * all_ensemble$selective_logging)

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense169_02_Inception4_01_dense161_01_ensemble.fold10-test.csv', row.names=FALSE, quote = FALSE)

###############################################################################################################################

##################################################################################################

# Cross validation fold 
##################################################################################################

model01 <- read_csv("./submissions/Prav.dense169_02.fold10.csv") 
model02 <- read_csv("./submissions/Prav.Inception4_01.fold10.csv")
model03 <- read_csv("./submissions/Prav.dense161_01.fold10.csv") 
model04 <- read_csv("./submissions/Prav.resnet152_01.fold10.csv") 

all_ensemble <- left_join(model01, model02, by = "image_name")
all_ensemble <- left_join(all_ensemble, model03, by = "image_name")
all_ensemble <- left_join(all_ensemble, model04, by = "image_name")
ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(model03)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.25 * all_ensemble$slash_burn.x       +0.25*all_ensemble$slash_burn.y       +0.25 * all_ensemble$slash_burn.x.x         +0.25 * all_ensemble$slash_burn.y.y           )
all_ensemble$clear                <- (0.25 * all_ensemble$clear.x            +0.25*all_ensemble$clear.y            +0.25 * all_ensemble$clear.x.x              +0.25 * all_ensemble$clear.y.y                )
all_ensemble$blooming             <- (0.25 * all_ensemble$blooming.x         +0.25*all_ensemble$blooming.y         +0.25 * all_ensemble$blooming.x.x           +0.25 * all_ensemble$blooming.y.y             )
all_ensemble$primary              <- (0.25 * all_ensemble$primary.x          +0.25*all_ensemble$primary.y          +0.25 * all_ensemble$primary.x.x            +0.25 * all_ensemble$primary.y.y              )
all_ensemble$cloudy               <- (0.25 * all_ensemble$cloudy.x           +0.25*all_ensemble$cloudy.y           +0.25 * all_ensemble$cloudy.x.x             +0.25 * all_ensemble$cloudy.y.y               )
all_ensemble$conventional_mine    <- (0.25 * all_ensemble$conventional_mine.x+0.25*all_ensemble$conventional_mine.y+0.25 * all_ensemble$conventional_mine.x.x  +0.25 * all_ensemble$conventional_mine.y.y    )
all_ensemble$water                <- (0.25 * all_ensemble$water.x            +0.25*all_ensemble$water.y            +0.25 * all_ensemble$water.x.x              +0.25 * all_ensemble$water.y.y                )
all_ensemble$haze                 <- (0.25 * all_ensemble$haze.x             +0.25*all_ensemble$haze.y             +0.25 * all_ensemble$haze.x.x               +0.25 * all_ensemble$haze.y.y                 )
all_ensemble$cultivation          <- (0.25 * all_ensemble$cultivation.x      +0.25*all_ensemble$cultivation.y      +0.25 * all_ensemble$cultivation.x.x        +0.25 * all_ensemble$cultivation.y.y          )
all_ensemble$partly_cloudy        <- (0.25 * all_ensemble$partly_cloudy.x    +0.25*all_ensemble$partly_cloudy.y    +0.25 * all_ensemble$partly_cloudy.x.x      +0.25 * all_ensemble$partly_cloudy.y.y        )
all_ensemble$artisinal_mine       <- (0.25 * all_ensemble$artisinal_mine.x   +0.25*all_ensemble$artisinal_mine.y   +0.25 * all_ensemble$artisinal_mine.x.x     +0.25 * all_ensemble$artisinal_mine.y.y       )
all_ensemble$habitation           <- (0.25 * all_ensemble$habitation.x       +0.25*all_ensemble$habitation.y       +0.25 * all_ensemble$habitation.x.x         +0.25 * all_ensemble$habitation.y.y           )
all_ensemble$bare_ground          <- (0.25 * all_ensemble$bare_ground.x      +0.25*all_ensemble$bare_ground.y      +0.25 * all_ensemble$bare_ground.x.x        +0.25 * all_ensemble$bare_ground.y.y          )
all_ensemble$blow_down            <- (0.25 * all_ensemble$blow_down.x        +0.25*all_ensemble$blow_down.y        +0.25 * all_ensemble$blow_down.x.x          +0.25 * all_ensemble$blow_down.y.y            )
all_ensemble$agriculture          <- (0.25 * all_ensemble$agriculture.x      +0.25*all_ensemble$agriculture.y      +0.25 * all_ensemble$agriculture.x.x        +0.25 * all_ensemble$agriculture.y.y          )
all_ensemble$road                 <- (0.25 * all_ensemble$road.x             +0.25*all_ensemble$road.y             +0.25 * all_ensemble$road.x.x               +0.25 * all_ensemble$road.y.y                 )
all_ensemble$selective_logging    <- (0.25 * all_ensemble$selective_logging.x+0.25*all_ensemble$selective_logging.y+0.25 * all_ensemble$selective_logging.x.x  +0.25 * all_ensemble$selective_logging.y.y    )

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.allensemlbe.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.dense169_02.fold10.csv") 
model02 <- read_csv("./submissions/Prav.dense169_04.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (all_ensemble$slash_burn.x+all_ensemble$slash_burn.y)/2
all_ensemble$clear                <- (all_ensemble$clear.x+all_ensemble$clear.y)/2
all_ensemble$blooming             <- (all_ensemble$blooming.x+all_ensemble$blooming.y)/2
all_ensemble$primary              <- (all_ensemble$primary.x+all_ensemble$primary.y)/2
all_ensemble$cloudy               <- (all_ensemble$cloudy.x+all_ensemble$cloudy.y)/2
all_ensemble$conventional_mine    <- (all_ensemble$conventional_mine.x+all_ensemble$conventional_mine.y)/2
all_ensemble$water                <- (all_ensemble$water.x+all_ensemble$water.y)/2
all_ensemble$haze                 <- (all_ensemble$haze.x+all_ensemble$haze.y)/2
all_ensemble$cultivation          <- (all_ensemble$cultivation.x+all_ensemble$cultivation.y)/2
all_ensemble$partly_cloudy        <- (all_ensemble$partly_cloudy.x+all_ensemble$partly_cloudy.y)/2
all_ensemble$artisinal_mine       <- (all_ensemble$artisinal_mine.x+all_ensemble$artisinal_mine.y)/2
all_ensemble$habitation           <- (all_ensemble$habitation.x+all_ensemble$habitation.y)/2
all_ensemble$bare_ground          <- (all_ensemble$bare_ground.x+all_ensemble$bare_ground.y)/2
all_ensemble$blow_down            <- (all_ensemble$blow_down.x+all_ensemble$blow_down.y)/2
all_ensemble$agriculture          <- (all_ensemble$agriculture.x+all_ensemble$agriculture.y)/2
all_ensemble$road                 <- (all_ensemble$road.x+all_ensemble$road.y)/2
all_ensemble$selective_logging    <- (all_ensemble$selective_logging.x+all_ensemble$selective_logging.y)/2

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense169_02_04_ensemble.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################


model01 <- read_csv("./submissions/Prav.dense169_02.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.dense169_04.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (all_ensemble$slash_burn.x+all_ensemble$slash_burn.y)/2
all_ensemble$clear                <- (all_ensemble$clear.x+all_ensemble$clear.y)/2
all_ensemble$blooming             <- (all_ensemble$blooming.x+all_ensemble$blooming.y)/2
all_ensemble$primary              <- (all_ensemble$primary.x+all_ensemble$primary.y)/2
all_ensemble$cloudy               <- (all_ensemble$cloudy.x+all_ensemble$cloudy.y)/2
all_ensemble$conventional_mine    <- (all_ensemble$conventional_mine.x+all_ensemble$conventional_mine.y)/2
all_ensemble$water                <- (all_ensemble$water.x+all_ensemble$water.y)/2
all_ensemble$haze                 <- (all_ensemble$haze.x+all_ensemble$haze.y)/2
all_ensemble$cultivation          <- (all_ensemble$cultivation.x+all_ensemble$cultivation.y)/2
all_ensemble$partly_cloudy        <- (all_ensemble$partly_cloudy.x+all_ensemble$partly_cloudy.y)/2
all_ensemble$artisinal_mine       <- (all_ensemble$artisinal_mine.x+all_ensemble$artisinal_mine.y)/2
all_ensemble$habitation           <- (all_ensemble$habitation.x+all_ensemble$habitation.y)/2
all_ensemble$bare_ground          <- (all_ensemble$bare_ground.x+all_ensemble$bare_ground.y)/2
all_ensemble$blow_down            <- (all_ensemble$blow_down.x+all_ensemble$blow_down.y)/2
all_ensemble$agriculture          <- (all_ensemble$agriculture.x+all_ensemble$agriculture.y)/2
all_ensemble$road                 <- (all_ensemble$road.x+all_ensemble$road.y)/2
all_ensemble$selective_logging    <- (all_ensemble$selective_logging.x+all_ensemble$selective_logging.y)/2

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense169_02_04_ensemble.fold10-test.csv', row.names=FALSE, quote = FALSE)

##################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.Inception4_01.fold10.csv") 
model02 <- read_csv("./submissions/Prav.Inception4_02.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (all_ensemble$slash_burn.x+all_ensemble$slash_burn.y)/2
all_ensemble$clear                <- (all_ensemble$clear.x+all_ensemble$clear.y)/2
all_ensemble$blooming             <- (all_ensemble$blooming.x+all_ensemble$blooming.y)/2
all_ensemble$primary              <- (all_ensemble$primary.x+all_ensemble$primary.y)/2
all_ensemble$cloudy               <- (all_ensemble$cloudy.x+all_ensemble$cloudy.y)/2
all_ensemble$conventional_mine    <- (all_ensemble$conventional_mine.x+all_ensemble$conventional_mine.y)/2
all_ensemble$water                <- (all_ensemble$water.x+all_ensemble$water.y)/2
all_ensemble$haze                 <- (all_ensemble$haze.x+all_ensemble$haze.y)/2
all_ensemble$cultivation          <- (all_ensemble$cultivation.x+all_ensemble$cultivation.y)/2
all_ensemble$partly_cloudy        <- (all_ensemble$partly_cloudy.x+all_ensemble$partly_cloudy.y)/2
all_ensemble$artisinal_mine       <- (all_ensemble$artisinal_mine.x+all_ensemble$artisinal_mine.y)/2
all_ensemble$habitation           <- (all_ensemble$habitation.x+all_ensemble$habitation.y)/2
all_ensemble$bare_ground          <- (all_ensemble$bare_ground.x+all_ensemble$bare_ground.y)/2
all_ensemble$blow_down            <- (all_ensemble$blow_down.x+all_ensemble$blow_down.y)/2
all_ensemble$agriculture          <- (all_ensemble$agriculture.x+all_ensemble$agriculture.y)/2
all_ensemble$road                 <- (all_ensemble$road.x+all_ensemble$road.y)/2
all_ensemble$selective_logging    <- (all_ensemble$selective_logging.x+all_ensemble$selective_logging.y)/2

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.Inception4_01_02_ensemble.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

############################################################################################################################


model01 <- read_csv("./submissions/Prav.Inception4_01.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.Inception4_02.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (all_ensemble$slash_burn.x+all_ensemble$slash_burn.y)/2
all_ensemble$clear                <- (all_ensemble$clear.x+all_ensemble$clear.y)/2
all_ensemble$blooming             <- (all_ensemble$blooming.x+all_ensemble$blooming.y)/2
all_ensemble$primary              <- (all_ensemble$primary.x+all_ensemble$primary.y)/2
all_ensemble$cloudy               <- (all_ensemble$cloudy.x+all_ensemble$cloudy.y)/2
all_ensemble$conventional_mine    <- (all_ensemble$conventional_mine.x+all_ensemble$conventional_mine.y)/2
all_ensemble$water                <- (all_ensemble$water.x+all_ensemble$water.y)/2
all_ensemble$haze                 <- (all_ensemble$haze.x+all_ensemble$haze.y)/2
all_ensemble$cultivation          <- (all_ensemble$cultivation.x+all_ensemble$cultivation.y)/2
all_ensemble$partly_cloudy        <- (all_ensemble$partly_cloudy.x+all_ensemble$partly_cloudy.y)/2
all_ensemble$artisinal_mine       <- (all_ensemble$artisinal_mine.x+all_ensemble$artisinal_mine.y)/2
all_ensemble$habitation           <- (all_ensemble$habitation.x+all_ensemble$habitation.y)/2
all_ensemble$bare_ground          <- (all_ensemble$bare_ground.x+all_ensemble$bare_ground.y)/2
all_ensemble$blow_down            <- (all_ensemble$blow_down.x+all_ensemble$blow_down.y)/2
all_ensemble$agriculture          <- (all_ensemble$agriculture.x+all_ensemble$agriculture.y)/2
all_ensemble$road                 <- (all_ensemble$road.x+all_ensemble$road.y)/2
all_ensemble$selective_logging    <- (all_ensemble$selective_logging.x+all_ensemble$selective_logging.y)/2

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.Inception4_01_02_ensemble.fold10-test.csv', row.names=FALSE, quote = FALSE)

##################################################################################################


##################################################################################################

model01 <- read_csv("./submissions/Prav.dense169_02_04_ensemble.fold10.csv") 
model02 <- read_csv("./submissions/Prav.Inception4_01_02_ensemble.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (all_ensemble$slash_burn.x+all_ensemble$slash_burn.y)/2
all_ensemble$clear                <- (all_ensemble$clear.x+all_ensemble$clear.y)/2
all_ensemble$blooming             <- (all_ensemble$blooming.x+all_ensemble$blooming.y)/2
all_ensemble$primary              <- (all_ensemble$primary.x+all_ensemble$primary.y)/2
all_ensemble$cloudy               <- (all_ensemble$cloudy.x+all_ensemble$cloudy.y)/2
all_ensemble$conventional_mine    <- (all_ensemble$conventional_mine.x+all_ensemble$conventional_mine.y)/2
all_ensemble$water                <- (all_ensemble$water.x+all_ensemble$water.y)/2
all_ensemble$haze                 <- (all_ensemble$haze.x+all_ensemble$haze.y)/2
all_ensemble$cultivation          <- (all_ensemble$cultivation.x+all_ensemble$cultivation.y)/2
all_ensemble$partly_cloudy        <- (all_ensemble$partly_cloudy.x+all_ensemble$partly_cloudy.y)/2
all_ensemble$artisinal_mine       <- (all_ensemble$artisinal_mine.x+all_ensemble$artisinal_mine.y)/2
all_ensemble$habitation           <- (all_ensemble$habitation.x+all_ensemble$habitation.y)/2
all_ensemble$bare_ground          <- (all_ensemble$bare_ground.x+all_ensemble$bare_ground.y)/2
all_ensemble$blow_down            <- (all_ensemble$blow_down.x+all_ensemble$blow_down.y)/2
all_ensemble$agriculture          <- (all_ensemble$agriculture.x+all_ensemble$agriculture.y)/2
all_ensemble$road                 <- (all_ensemble$road.x+all_ensemble$road.y)/2
all_ensemble$selective_logging    <- (all_ensemble$selective_logging.x+all_ensemble$selective_logging.y)/2

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.Inception12Dense24.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

############################################################################################################################


model01 <- read_csv("./submissions/Prav.dense169_02_04_ensemble.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.Inception4_01_02_ensemble.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (all_ensemble$slash_burn.x+all_ensemble$slash_burn.y)/2
all_ensemble$clear                <- (all_ensemble$clear.x+all_ensemble$clear.y)/2
all_ensemble$blooming             <- (all_ensemble$blooming.x+all_ensemble$blooming.y)/2
all_ensemble$primary              <- (all_ensemble$primary.x+all_ensemble$primary.y)/2
all_ensemble$cloudy               <- (all_ensemble$cloudy.x+all_ensemble$cloudy.y)/2
all_ensemble$conventional_mine    <- (all_ensemble$conventional_mine.x+all_ensemble$conventional_mine.y)/2
all_ensemble$water                <- (all_ensemble$water.x+all_ensemble$water.y)/2
all_ensemble$haze                 <- (all_ensemble$haze.x+all_ensemble$haze.y)/2
all_ensemble$cultivation          <- (all_ensemble$cultivation.x+all_ensemble$cultivation.y)/2
all_ensemble$partly_cloudy        <- (all_ensemble$partly_cloudy.x+all_ensemble$partly_cloudy.y)/2
all_ensemble$artisinal_mine       <- (all_ensemble$artisinal_mine.x+all_ensemble$artisinal_mine.y)/2
all_ensemble$habitation           <- (all_ensemble$habitation.x+all_ensemble$habitation.y)/2
all_ensemble$bare_ground          <- (all_ensemble$bare_ground.x+all_ensemble$bare_ground.y)/2
all_ensemble$blow_down            <- (all_ensemble$blow_down.x+all_ensemble$blow_down.y)/2
all_ensemble$agriculture          <- (all_ensemble$agriculture.x+all_ensemble$agriculture.y)/2
all_ensemble$road                 <- (all_ensemble$road.x+all_ensemble$road.y)/2
all_ensemble$selective_logging    <- (all_ensemble$selective_logging.x+all_ensemble$selective_logging.y)/2

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.Inception12Dense24.fold10-test.csv', row.names=FALSE, quote = FALSE)

##################################################################################################
##################################################################################################

model01 <- read_csv("./submissions/Prav.resnet50_01.fold10.csv") 
model02 <- read_csv("./submissions/Prav.Inceptionv3_01.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.25 * all_ensemble$slash_burn.x       +0.75*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.25 * all_ensemble$clear.x            +0.75*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.25 * all_ensemble$blooming.x         +0.75*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.25 * all_ensemble$primary.x          +0.75*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.25 * all_ensemble$cloudy.x           +0.75*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.25 * all_ensemble$conventional_mine.x+0.75*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.25 * all_ensemble$water.x            +0.75*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.25 * all_ensemble$haze.x             +0.75*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.25 * all_ensemble$cultivation.x      +0.75*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.25 * all_ensemble$partly_cloudy.x    +0.75*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.25 * all_ensemble$artisinal_mine.x   +0.75*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.25 * all_ensemble$habitation.x       +0.75*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.25 * all_ensemble$bare_ground.x      +0.75*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.25 * all_ensemble$blow_down.x        +0.75*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.25 * all_ensemble$agriculture.x      +0.75*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.25 * all_ensemble$road.x             +0.75*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.25 * all_ensemble$selective_logging.x+0.75*all_ensemble$selective_logging.y   )

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.Inception3_resnet50.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################


model01 <- read_csv("./submissions/Prav.resnet50_01.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.Inceptionv3_01.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.25 * all_ensemble$slash_burn.x       +0.75*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.25 * all_ensemble$clear.x            +0.75*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.25 * all_ensemble$blooming.x         +0.75*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.25 * all_ensemble$primary.x          +0.75*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.25 * all_ensemble$cloudy.x           +0.75*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.25 * all_ensemble$conventional_mine.x+0.75*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.25 * all_ensemble$water.x            +0.75*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.25 * all_ensemble$haze.x             +0.75*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.25 * all_ensemble$cultivation.x      +0.75*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.25 * all_ensemble$partly_cloudy.x    +0.75*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.25 * all_ensemble$artisinal_mine.x   +0.75*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.25 * all_ensemble$habitation.x       +0.75*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.25 * all_ensemble$bare_ground.x      +0.75*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.25 * all_ensemble$blow_down.x        +0.75*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.25 * all_ensemble$agriculture.x      +0.75*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.25 * all_ensemble$road.x             +0.75*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.25 * all_ensemble$selective_logging.x+0.75*all_ensemble$selective_logging.y   )

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.Inception3_resnet50.fold10-test.csv', row.names=FALSE, quote = FALSE)

##################################################################################################

model01 <- read_csv("./submissions/Prav.Inception3_resnet50.fold10.csv") 
model02 <- read_csv("./submissions/Prav.dense161_01.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (all_ensemble$slash_burn.x+all_ensemble$slash_burn.y)/2
all_ensemble$clear                <- (all_ensemble$clear.x+all_ensemble$clear.y)/2
all_ensemble$blooming             <- (all_ensemble$blooming.x+all_ensemble$blooming.y)/2
all_ensemble$primary              <- (all_ensemble$primary.x+all_ensemble$primary.y)/2
all_ensemble$cloudy               <- (all_ensemble$cloudy.x+all_ensemble$cloudy.y)/2
all_ensemble$conventional_mine    <- (all_ensemble$conventional_mine.x+all_ensemble$conventional_mine.y)/2
all_ensemble$water                <- (all_ensemble$water.x+all_ensemble$water.y)/2
all_ensemble$haze                 <- (all_ensemble$haze.x+all_ensemble$haze.y)/2
all_ensemble$cultivation          <- (all_ensemble$cultivation.x+all_ensemble$cultivation.y)/2
all_ensemble$partly_cloudy        <- (all_ensemble$partly_cloudy.x+all_ensemble$partly_cloudy.y)/2
all_ensemble$artisinal_mine       <- (all_ensemble$artisinal_mine.x+all_ensemble$artisinal_mine.y)/2
all_ensemble$habitation           <- (all_ensemble$habitation.x+all_ensemble$habitation.y)/2
all_ensemble$bare_ground          <- (all_ensemble$bare_ground.x+all_ensemble$bare_ground.y)/2
all_ensemble$blow_down            <- (all_ensemble$blow_down.x+all_ensemble$blow_down.y)/2
all_ensemble$agriculture          <- (all_ensemble$agriculture.x+all_ensemble$agriculture.y)/2
all_ensemble$road                 <- (all_ensemble$road.x+all_ensemble$road.y)/2
all_ensemble$selective_logging    <- (all_ensemble$selective_logging.x+all_ensemble$selective_logging.y)/2

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.low3.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

############################################################################################################################


model01 <- read_csv("./submissions/Prav.Inception4_01_02_ensemble023.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.dense161_01.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (all_ensemble$slash_burn.x+all_ensemble$slash_burn.y)/2
all_ensemble$clear                <- (all_ensemble$clear.x+all_ensemble$clear.y)/2
all_ensemble$blooming             <- (all_ensemble$blooming.x+all_ensemble$blooming.y)/2
all_ensemble$primary              <- (all_ensemble$primary.x+all_ensemble$primary.y)/2
all_ensemble$cloudy               <- (all_ensemble$cloudy.x+all_ensemble$cloudy.y)/2
all_ensemble$conventional_mine    <- (all_ensemble$conventional_mine.x+all_ensemble$conventional_mine.y)/2
all_ensemble$water                <- (all_ensemble$water.x+all_ensemble$water.y)/2
all_ensemble$haze                 <- (all_ensemble$haze.x+all_ensemble$haze.y)/2
all_ensemble$cultivation          <- (all_ensemble$cultivation.x+all_ensemble$cultivation.y)/2
all_ensemble$partly_cloudy        <- (all_ensemble$partly_cloudy.x+all_ensemble$partly_cloudy.y)/2
all_ensemble$artisinal_mine       <- (all_ensemble$artisinal_mine.x+all_ensemble$artisinal_mine.y)/2
all_ensemble$habitation           <- (all_ensemble$habitation.x+all_ensemble$habitation.y)/2
all_ensemble$bare_ground          <- (all_ensemble$bare_ground.x+all_ensemble$bare_ground.y)/2
all_ensemble$blow_down            <- (all_ensemble$blow_down.x+all_ensemble$blow_down.y)/2
all_ensemble$agriculture          <- (all_ensemble$agriculture.x+all_ensemble$agriculture.y)/2
all_ensemble$road                 <- (all_ensemble$road.x+all_ensemble$road.y)/2
all_ensemble$selective_logging    <- (all_ensemble$selective_logging.x+all_ensemble$selective_logging.y)/2

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.Inception12Dense24.fold10-test.csv', row.names=FALSE, quote = FALSE)

##################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.dense161_01.fold10.csv") 
model02 <- read_csv("./submissions/Prav.dense161_02.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.5 * all_ensemble$slash_burn.x       +0.5*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.5 * all_ensemble$clear.x            +0.5*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.5 * all_ensemble$blooming.x         +0.5*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.5 * all_ensemble$primary.x          +0.5*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.5 * all_ensemble$cloudy.x           +0.5*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.5 * all_ensemble$conventional_mine.x+0.5*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.5 * all_ensemble$water.x            +0.5*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.5 * all_ensemble$haze.x             +0.5*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.5 * all_ensemble$cultivation.x      +0.5*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.5 * all_ensemble$partly_cloudy.x    +0.5*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.5 * all_ensemble$artisinal_mine.x   +0.5*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.5 * all_ensemble$habitation.x       +0.5*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.5 * all_ensemble$bare_ground.x      +0.5*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.5 * all_ensemble$blow_down.x        +0.5*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.5 * all_ensemble$agriculture.x      +0.5*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.5 * all_ensemble$road.x             +0.5*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.5 * all_ensemble$selective_logging.x+0.5*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense161.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.dense161_01.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.dense161_02.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.5 * all_ensemble$slash_burn.x       +0.5*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.5 * all_ensemble$clear.x            +0.5*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.5 * all_ensemble$blooming.x         +0.5*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.5 * all_ensemble$primary.x          +0.5*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.5 * all_ensemble$cloudy.x           +0.5*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.5 * all_ensemble$conventional_mine.x+0.5*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.5 * all_ensemble$water.x            +0.5*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.5 * all_ensemble$haze.x             +0.5*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.5 * all_ensemble$cultivation.x      +0.5*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.5 * all_ensemble$partly_cloudy.x    +0.5*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.5 * all_ensemble$artisinal_mine.x   +0.5*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.5 * all_ensemble$habitation.x       +0.5*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.5 * all_ensemble$bare_ground.x      +0.5*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.5 * all_ensemble$blow_down.x        +0.5*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.5 * all_ensemble$agriculture.x      +0.5*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.5 * all_ensemble$road.x             +0.5*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.5 * all_ensemble$selective_logging.x+0.5*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense161.fold10-test.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.dense161.fold10.csv") 
model02 <- read_csv("./submissions/Prav.Inception12Dense24.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.3 * all_ensemble$slash_burn.x       +0.7*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.3 * all_ensemble$clear.x            +0.7*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.3 * all_ensemble$blooming.x         +0.7*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.3 * all_ensemble$primary.x          +0.7*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.3 * all_ensemble$cloudy.x           +0.7*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.3 * all_ensemble$conventional_mine.x+0.7*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.3 * all_ensemble$water.x            +0.7*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.3 * all_ensemble$haze.x             +0.7*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.3 * all_ensemble$cultivation.x      +0.7*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.3 * all_ensemble$partly_cloudy.x    +0.7*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.3 * all_ensemble$artisinal_mine.x   +0.7*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.3 * all_ensemble$habitation.x       +0.7*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.3 * all_ensemble$bare_ground.x      +0.7*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.3 * all_ensemble$blow_down.x        +0.7*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.3 * all_ensemble$agriculture.x      +0.7*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.3 * all_ensemble$road.x             +0.7*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.3 * all_ensemble$selective_logging.x+0.7*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.top2models.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################


model01 <- read_csv("./submissions/Prav.dense161.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.Inception12Dense24.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.3 * all_ensemble$slash_burn.x       +0.7*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.3 * all_ensemble$clear.x            +0.7*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.3 * all_ensemble$blooming.x         +0.7*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.3 * all_ensemble$primary.x          +0.7*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.3 * all_ensemble$cloudy.x           +0.7*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.3 * all_ensemble$conventional_mine.x+0.7*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.3 * all_ensemble$water.x            +0.7*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.3 * all_ensemble$haze.x             +0.7*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.3 * all_ensemble$cultivation.x      +0.7*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.3 * all_ensemble$partly_cloudy.x    +0.7*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.3 * all_ensemble$artisinal_mine.x   +0.7*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.3 * all_ensemble$habitation.x       +0.7*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.3 * all_ensemble$bare_ground.x      +0.7*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.3 * all_ensemble$blow_down.x        +0.7*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.3 * all_ensemble$agriculture.x      +0.7*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.3 * all_ensemble$road.x             +0.7*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.3 * all_ensemble$selective_logging.x+0.7*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.top2models.fold10-test.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################


############################################################################################################################


model01 <- read_csv("./submissions/Prav.dense169_04.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.dense169_04_h.fold10-test.csv")
model03 <- read_csv("./submissions/Prav.dense169_04_r.fold10-test.csv") 

all_ensemble <- left_join(model01, model02, by = "image_name")
all_ensemble <- left_join(all_ensemble, model03, by = "image_name")
ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(model03)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- ( all_ensemble$slash_burn.x       +all_ensemble$slash_burn.y       + all_ensemble$slash_burn       )/3
all_ensemble$clear                <- ( all_ensemble$clear.x            +all_ensemble$clear.y            + all_ensemble$clear            )/3
all_ensemble$blooming             <- ( all_ensemble$blooming.x         +all_ensemble$blooming.y         + all_ensemble$blooming         )/3
all_ensemble$primary              <- ( all_ensemble$primary.x          +all_ensemble$primary.y          + all_ensemble$primary          )/3
all_ensemble$cloudy               <- ( all_ensemble$cloudy.x           +all_ensemble$cloudy.y           + all_ensemble$cloudy           )/3
all_ensemble$conventional_mine    <- ( all_ensemble$conventional_mine.x+all_ensemble$conventional_mine.y+ all_ensemble$conventional_mine)/3
all_ensemble$water                <- ( all_ensemble$water.x            +all_ensemble$water.y            + all_ensemble$water            )/3
all_ensemble$haze                 <- ( all_ensemble$haze.x             +all_ensemble$haze.y             + all_ensemble$haze             )/3
all_ensemble$cultivation          <- ( all_ensemble$cultivation.x      +all_ensemble$cultivation.y      + all_ensemble$cultivation      )/3
all_ensemble$partly_cloudy        <- ( all_ensemble$partly_cloudy.x    +all_ensemble$partly_cloudy.y    + all_ensemble$partly_cloudy    )/3
all_ensemble$artisinal_mine       <- ( all_ensemble$artisinal_mine.x   +all_ensemble$artisinal_mine.y   + all_ensemble$artisinal_mine   )/3
all_ensemble$habitation           <- ( all_ensemble$habitation.x       +all_ensemble$habitation.y       + all_ensemble$habitation       )/3
all_ensemble$bare_ground          <- ( all_ensemble$bare_ground.x      +all_ensemble$bare_ground.y      + all_ensemble$bare_ground      )/3
all_ensemble$blow_down            <- ( all_ensemble$blow_down.x        +all_ensemble$blow_down.y        + all_ensemble$blow_down        )/3
all_ensemble$agriculture          <- ( all_ensemble$agriculture.x      +all_ensemble$agriculture.y      + all_ensemble$agriculture      )/3
all_ensemble$road                 <- ( all_ensemble$road.x             +all_ensemble$road.y             + all_ensemble$road             )/3
all_ensemble$selective_logging    <- ( all_ensemble$selective_logging.x+all_ensemble$selective_logging.y+ all_ensemble$selective_logging)/3

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense169_04_aug.fold10-test.csv', row.names=FALSE, quote = FALSE)

###############################################################################################################################


############################################################################################################################


model01 <- read_csv("./submissions/Prav.dense161_02.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.dense161_02_h.fold10-test.csv")
model03 <- read_csv("./submissions/Prav.dense161_02_r.fold10-test.csv") 

all_ensemble <- left_join(model01, model02, by = "image_name")
all_ensemble <- left_join(all_ensemble, model03, by = "image_name")
ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(model03)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- ( all_ensemble$slash_burn.x       +all_ensemble$slash_burn.y       + all_ensemble$slash_burn       )/3
all_ensemble$clear                <- ( all_ensemble$clear.x            +all_ensemble$clear.y            + all_ensemble$clear            )/3
all_ensemble$blooming             <- ( all_ensemble$blooming.x         +all_ensemble$blooming.y         + all_ensemble$blooming         )/3
all_ensemble$primary              <- ( all_ensemble$primary.x          +all_ensemble$primary.y          + all_ensemble$primary          )/3
all_ensemble$cloudy               <- ( all_ensemble$cloudy.x           +all_ensemble$cloudy.y           + all_ensemble$cloudy           )/3
all_ensemble$conventional_mine    <- ( all_ensemble$conventional_mine.x+all_ensemble$conventional_mine.y+ all_ensemble$conventional_mine)/3
all_ensemble$water                <- ( all_ensemble$water.x            +all_ensemble$water.y            + all_ensemble$water            )/3
all_ensemble$haze                 <- ( all_ensemble$haze.x             +all_ensemble$haze.y             + all_ensemble$haze             )/3
all_ensemble$cultivation          <- ( all_ensemble$cultivation.x      +all_ensemble$cultivation.y      + all_ensemble$cultivation      )/3
all_ensemble$partly_cloudy        <- ( all_ensemble$partly_cloudy.x    +all_ensemble$partly_cloudy.y    + all_ensemble$partly_cloudy    )/3
all_ensemble$artisinal_mine       <- ( all_ensemble$artisinal_mine.x   +all_ensemble$artisinal_mine.y   + all_ensemble$artisinal_mine   )/3
all_ensemble$habitation           <- ( all_ensemble$habitation.x       +all_ensemble$habitation.y       + all_ensemble$habitation       )/3
all_ensemble$bare_ground          <- ( all_ensemble$bare_ground.x      +all_ensemble$bare_ground.y      + all_ensemble$bare_ground      )/3
all_ensemble$blow_down            <- ( all_ensemble$blow_down.x        +all_ensemble$blow_down.y        + all_ensemble$blow_down        )/3
all_ensemble$agriculture          <- ( all_ensemble$agriculture.x      +all_ensemble$agriculture.y      + all_ensemble$agriculture      )/3
all_ensemble$road                 <- ( all_ensemble$road.x             +all_ensemble$road.y             + all_ensemble$road             )/3
all_ensemble$selective_logging    <- ( all_ensemble$selective_logging.x+all_ensemble$selective_logging.y+ all_ensemble$selective_logging)/3

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense161_02_aug.fold10-test.csv', row.names=FALSE, quote = FALSE)

###############################################################################################################################

############################################################################################################################


model01 <- read_csv("./submissions/Prav.Inception4_02.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.Inception4_02_h.fold10-test.csv")
model03 <- read_csv("./submissions/Prav.Inception4_02_r.fold10-test.csv") 

all_ensemble <- left_join(model01, model02, by = "image_name")
all_ensemble <- left_join(all_ensemble, model03, by = "image_name")
ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(model03)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- ( all_ensemble$slash_burn.x       +all_ensemble$slash_burn.y       + all_ensemble$slash_burn       )/3
all_ensemble$clear                <- ( all_ensemble$clear.x            +all_ensemble$clear.y            + all_ensemble$clear            )/3
all_ensemble$blooming             <- ( all_ensemble$blooming.x         +all_ensemble$blooming.y         + all_ensemble$blooming         )/3
all_ensemble$primary              <- ( all_ensemble$primary.x          +all_ensemble$primary.y          + all_ensemble$primary          )/3
all_ensemble$cloudy               <- ( all_ensemble$cloudy.x           +all_ensemble$cloudy.y           + all_ensemble$cloudy           )/3
all_ensemble$conventional_mine    <- ( all_ensemble$conventional_mine.x+all_ensemble$conventional_mine.y+ all_ensemble$conventional_mine)/3
all_ensemble$water                <- ( all_ensemble$water.x            +all_ensemble$water.y            + all_ensemble$water            )/3
all_ensemble$haze                 <- ( all_ensemble$haze.x             +all_ensemble$haze.y             + all_ensemble$haze             )/3
all_ensemble$cultivation          <- ( all_ensemble$cultivation.x      +all_ensemble$cultivation.y      + all_ensemble$cultivation      )/3
all_ensemble$partly_cloudy        <- ( all_ensemble$partly_cloudy.x    +all_ensemble$partly_cloudy.y    + all_ensemble$partly_cloudy    )/3
all_ensemble$artisinal_mine       <- ( all_ensemble$artisinal_mine.x   +all_ensemble$artisinal_mine.y   + all_ensemble$artisinal_mine   )/3
all_ensemble$habitation           <- ( all_ensemble$habitation.x       +all_ensemble$habitation.y       + all_ensemble$habitation       )/3
all_ensemble$bare_ground          <- ( all_ensemble$bare_ground.x      +all_ensemble$bare_ground.y      + all_ensemble$bare_ground      )/3
all_ensemble$blow_down            <- ( all_ensemble$blow_down.x        +all_ensemble$blow_down.y        + all_ensemble$blow_down        )/3
all_ensemble$agriculture          <- ( all_ensemble$agriculture.x      +all_ensemble$agriculture.y      + all_ensemble$agriculture      )/3
all_ensemble$road                 <- ( all_ensemble$road.x             +all_ensemble$road.y             + all_ensemble$road             )/3
all_ensemble$selective_logging    <- ( all_ensemble$selective_logging.x+all_ensemble$selective_logging.y+ all_ensemble$selective_logging)/3

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.Inception4_02_aug.fold10-test.csv', row.names=FALSE, quote = FALSE)

###############################################################################################################################

############################################################################################################################


model01 <- read_csv("./submissions/Prav.Inceptionv3_01.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.Inceptionv3_01_h.fold10-test.csv")
model03 <- read_csv("./submissions/Prav.Inceptionv3_01_r.fold10-test.csv") 

all_ensemble <- left_join(model01, model02, by = "image_name")
all_ensemble <- left_join(all_ensemble, model03, by = "image_name")
ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(model03)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- ( all_ensemble$slash_burn.x       +all_ensemble$slash_burn.y       + all_ensemble$slash_burn       )/3
all_ensemble$clear                <- ( all_ensemble$clear.x            +all_ensemble$clear.y            + all_ensemble$clear            )/3
all_ensemble$blooming             <- ( all_ensemble$blooming.x         +all_ensemble$blooming.y         + all_ensemble$blooming         )/3
all_ensemble$primary              <- ( all_ensemble$primary.x          +all_ensemble$primary.y          + all_ensemble$primary          )/3
all_ensemble$cloudy               <- ( all_ensemble$cloudy.x           +all_ensemble$cloudy.y           + all_ensemble$cloudy           )/3
all_ensemble$conventional_mine    <- ( all_ensemble$conventional_mine.x+all_ensemble$conventional_mine.y+ all_ensemble$conventional_mine)/3
all_ensemble$water                <- ( all_ensemble$water.x            +all_ensemble$water.y            + all_ensemble$water            )/3
all_ensemble$haze                 <- ( all_ensemble$haze.x             +all_ensemble$haze.y             + all_ensemble$haze             )/3
all_ensemble$cultivation          <- ( all_ensemble$cultivation.x      +all_ensemble$cultivation.y      + all_ensemble$cultivation      )/3
all_ensemble$partly_cloudy        <- ( all_ensemble$partly_cloudy.x    +all_ensemble$partly_cloudy.y    + all_ensemble$partly_cloudy    )/3
all_ensemble$artisinal_mine       <- ( all_ensemble$artisinal_mine.x   +all_ensemble$artisinal_mine.y   + all_ensemble$artisinal_mine   )/3
all_ensemble$habitation           <- ( all_ensemble$habitation.x       +all_ensemble$habitation.y       + all_ensemble$habitation       )/3
all_ensemble$bare_ground          <- ( all_ensemble$bare_ground.x      +all_ensemble$bare_ground.y      + all_ensemble$bare_ground      )/3
all_ensemble$blow_down            <- ( all_ensemble$blow_down.x        +all_ensemble$blow_down.y        + all_ensemble$blow_down        )/3
all_ensemble$agriculture          <- ( all_ensemble$agriculture.x      +all_ensemble$agriculture.y      + all_ensemble$agriculture      )/3
all_ensemble$road                 <- ( all_ensemble$road.x             +all_ensemble$road.y             + all_ensemble$road             )/3
all_ensemble$selective_logging    <- ( all_ensemble$selective_logging.x+all_ensemble$selective_logging.y+ all_ensemble$selective_logging)/3

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.Inceptionv3_01_aug.fold10-test.csv', row.names=FALSE, quote = FALSE)

###############################################################################################################################


##################################################################################################

model01 <- read_csv("./submissions/Prav.dense161_02.fold10.csv") 
model02 <- read_csv("./submissions/Prav.Inceptionv3_01.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.5 * all_ensemble$slash_burn.x       +0.5*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.5 * all_ensemble$clear.x            +0.5*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.5 * all_ensemble$blooming.x         +0.5*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.5 * all_ensemble$primary.x          +0.5*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.5 * all_ensemble$cloudy.x           +0.5*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.5 * all_ensemble$conventional_mine.x+0.5*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.5 * all_ensemble$water.x            +0.5*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.5 * all_ensemble$haze.x             +0.5*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.5 * all_ensemble$cultivation.x      +0.5*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.5 * all_ensemble$partly_cloudy.x    +0.5*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.5 * all_ensemble$artisinal_mine.x   +0.5*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.5 * all_ensemble$habitation.x       +0.5*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.5 * all_ensemble$bare_ground.x      +0.5*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.5 * all_ensemble$blow_down.x        +0.5*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.5 * all_ensemble$agriculture.x      +0.5*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.5 * all_ensemble$road.x             +0.5*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.5 * all_ensemble$selective_logging.x+0.5*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense161_Inc3.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.dense161_02_aug.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.Inceptionv3_01_aug.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.5 * all_ensemble$slash_burn.x       +0.5*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.5 * all_ensemble$clear.x            +0.5*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.5 * all_ensemble$blooming.x         +0.5*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.5 * all_ensemble$primary.x          +0.5*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.5 * all_ensemble$cloudy.x           +0.5*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.5 * all_ensemble$conventional_mine.x+0.5*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.5 * all_ensemble$water.x            +0.5*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.5 * all_ensemble$haze.x             +0.5*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.5 * all_ensemble$cultivation.x      +0.5*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.5 * all_ensemble$partly_cloudy.x    +0.5*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.5 * all_ensemble$artisinal_mine.x   +0.5*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.5 * all_ensemble$habitation.x       +0.5*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.5 * all_ensemble$bare_ground.x      +0.5*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.5 * all_ensemble$blow_down.x        +0.5*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.5 * all_ensemble$agriculture.x      +0.5*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.5 * all_ensemble$road.x             +0.5*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.5 * all_ensemble$selective_logging.x+0.5*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense161_Inc3_aug.fold10-test.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################


##################################################################################################

model01 <- read_csv("./submissions/Prav.dense169_04.fold10.csv") 
model02 <- read_csv("./submissions/Prav.Inception4_02.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.5 * all_ensemble$slash_burn.x       +0.5*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.5 * all_ensemble$clear.x            +0.5*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.5 * all_ensemble$blooming.x         +0.5*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.5 * all_ensemble$primary.x          +0.5*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.5 * all_ensemble$cloudy.x           +0.5*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.5 * all_ensemble$conventional_mine.x+0.5*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.5 * all_ensemble$water.x            +0.5*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.5 * all_ensemble$haze.x             +0.5*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.5 * all_ensemble$cultivation.x      +0.5*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.5 * all_ensemble$partly_cloudy.x    +0.5*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.5 * all_ensemble$artisinal_mine.x   +0.5*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.5 * all_ensemble$habitation.x       +0.5*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.5 * all_ensemble$bare_ground.x      +0.5*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.5 * all_ensemble$blow_down.x        +0.5*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.5 * all_ensemble$agriculture.x      +0.5*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.5 * all_ensemble$road.x             +0.5*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.5 * all_ensemble$selective_logging.x+0.5*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense169_Inc4.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################
##################################################################################################

model01 <- read_csv("./submissions/Prav.dense169_04_aug.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.Inception4_02_aug.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.5 * all_ensemble$slash_burn.x       +0.5*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.5 * all_ensemble$clear.x            +0.5*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.5 * all_ensemble$blooming.x         +0.5*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.5 * all_ensemble$primary.x          +0.5*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.5 * all_ensemble$cloudy.x           +0.5*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.5 * all_ensemble$conventional_mine.x+0.5*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.5 * all_ensemble$water.x            +0.5*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.5 * all_ensemble$haze.x             +0.5*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.5 * all_ensemble$cultivation.x      +0.5*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.5 * all_ensemble$partly_cloudy.x    +0.5*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.5 * all_ensemble$artisinal_mine.x   +0.5*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.5 * all_ensemble$habitation.x       +0.5*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.5 * all_ensemble$bare_ground.x      +0.5*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.5 * all_ensemble$blow_down.x        +0.5*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.5 * all_ensemble$agriculture.x      +0.5*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.5 * all_ensemble$road.x             +0.5*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.5 * all_ensemble$selective_logging.x+0.5*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense169_Inc4_aug.fold10-test.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.dense161_Inc3.fold10.csv") 
model02 <- read_csv("./submissions/Prav.dense169_Inc4.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.5 * all_ensemble$slash_burn.x       +0.5*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.5 * all_ensemble$clear.x            +0.5*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.5 * all_ensemble$blooming.x         +0.5*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.5 * all_ensemble$primary.x          +0.5*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.5 * all_ensemble$cloudy.x           +0.5*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.5 * all_ensemble$conventional_mine.x+0.5*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.5 * all_ensemble$water.x            +0.5*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.5 * all_ensemble$haze.x             +0.5*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.5 * all_ensemble$cultivation.x      +0.5*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.5 * all_ensemble$partly_cloudy.x    +0.5*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.5 * all_ensemble$artisinal_mine.x   +0.5*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.5 * all_ensemble$habitation.x       +0.5*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.5 * all_ensemble$bare_ground.x      +0.5*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.5 * all_ensemble$blow_down.x        +0.5*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.5 * all_ensemble$agriculture.x      +0.5*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.5 * all_ensemble$road.x             +0.5*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.5 * all_ensemble$selective_logging.x+0.5*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.twoaug.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.dense161_Inc3_aug.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.dense169_Inc4_aug.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.5 * all_ensemble$slash_burn.x       +0.5*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.5 * all_ensemble$clear.x            +0.5*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.5 * all_ensemble$blooming.x         +0.5*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.5 * all_ensemble$primary.x          +0.5*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.5 * all_ensemble$cloudy.x           +0.5*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.5 * all_ensemble$conventional_mine.x+0.5*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.5 * all_ensemble$water.x            +0.5*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.5 * all_ensemble$haze.x             +0.5*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.5 * all_ensemble$cultivation.x      +0.5*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.5 * all_ensemble$partly_cloudy.x    +0.5*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.5 * all_ensemble$artisinal_mine.x   +0.5*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.5 * all_ensemble$habitation.x       +0.5*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.5 * all_ensemble$bare_ground.x      +0.5*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.5 * all_ensemble$blow_down.x        +0.5*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.5 * all_ensemble$agriculture.x      +0.5*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.5 * all_ensemble$road.x             +0.5*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.5 * all_ensemble$selective_logging.x+0.5*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.twoaug_aug.fold10-test.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.top2models.fold10.csv") 
model02 <- read_csv("./submissions/Prav.twoaug.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.5 * all_ensemble$slash_burn.x       +0.5*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.5 * all_ensemble$clear.x            +0.5*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.5 * all_ensemble$blooming.x         +0.5*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.5 * all_ensemble$primary.x          +0.5*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.5 * all_ensemble$cloudy.x           +0.5*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.5 * all_ensemble$conventional_mine.x+0.5*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.5 * all_ensemble$water.x            +0.5*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.5 * all_ensemble$haze.x             +0.5*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.5 * all_ensemble$cultivation.x      +0.5*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.5 * all_ensemble$partly_cloudy.x    +0.5*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.5 * all_ensemble$artisinal_mine.x   +0.5*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.5 * all_ensemble$habitation.x       +0.5*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.5 * all_ensemble$bare_ground.x      +0.5*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.5 * all_ensemble$blow_down.x        +0.5*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.5 * all_ensemble$agriculture.x      +0.5*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.5 * all_ensemble$road.x             +0.5*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.5 * all_ensemble$selective_logging.x+0.5*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.top2ensemble.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.top2models.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.twoaug_aug.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.5 * all_ensemble$slash_burn.x       +0.5*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.5 * all_ensemble$clear.x            +0.5*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.5 * all_ensemble$blooming.x         +0.5*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.5 * all_ensemble$primary.x          +0.5*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.5 * all_ensemble$cloudy.x           +0.5*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.5 * all_ensemble$conventional_mine.x+0.5*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.5 * all_ensemble$water.x            +0.5*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.5 * all_ensemble$haze.x             +0.5*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.5 * all_ensemble$cultivation.x      +0.5*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.5 * all_ensemble$partly_cloudy.x    +0.5*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.5 * all_ensemble$artisinal_mine.x   +0.5*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.5 * all_ensemble$habitation.x       +0.5*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.5 * all_ensemble$bare_ground.x      +0.5*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.5 * all_ensemble$blow_down.x        +0.5*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.5 * all_ensemble$agriculture.x      +0.5*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.5 * all_ensemble$road.x             +0.5*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.5 * all_ensemble$selective_logging.x+0.5*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.top2ensemble_aug.fold10-test.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.resnet50_01.fold10.csv") 
model02 <- read_csv("./submissions/Prav.Inceptionv3_01.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.25 * all_ensemble$slash_burn.x       +0.75*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.25 * all_ensemble$clear.x            +0.75*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.25 * all_ensemble$blooming.x         +0.75*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.25 * all_ensemble$primary.x          +0.75*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.25 * all_ensemble$cloudy.x           +0.75*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.25 * all_ensemble$conventional_mine.x+0.75*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.25 * all_ensemble$water.x            +0.75*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.25 * all_ensemble$haze.x             +0.75*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.25 * all_ensemble$cultivation.x      +0.75*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.25 * all_ensemble$partly_cloudy.x    +0.75*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.25 * all_ensemble$artisinal_mine.x   +0.75*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.25 * all_ensemble$habitation.x       +0.75*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.25 * all_ensemble$bare_ground.x      +0.75*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.25 * all_ensemble$blow_down.x        +0.75*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.25 * all_ensemble$agriculture.x      +0.75*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.25 * all_ensemble$road.x             +0.75*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.25 * all_ensemble$selective_logging.x+0.75*all_ensemble$selective_logging.y   )

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.Inception3_resnet50.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################


model01 <- read_csv("./submissions/Prav.resnet50_01.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.Inceptionv3_01_aug.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.25 * all_ensemble$slash_burn.x       +0.75*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.25 * all_ensemble$clear.x            +0.75*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.25 * all_ensemble$blooming.x         +0.75*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.25 * all_ensemble$primary.x          +0.75*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.25 * all_ensemble$cloudy.x           +0.75*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.25 * all_ensemble$conventional_mine.x+0.75*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.25 * all_ensemble$water.x            +0.75*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.25 * all_ensemble$haze.x             +0.75*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.25 * all_ensemble$cultivation.x      +0.75*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.25 * all_ensemble$partly_cloudy.x    +0.75*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.25 * all_ensemble$artisinal_mine.x   +0.75*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.25 * all_ensemble$habitation.x       +0.75*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.25 * all_ensemble$bare_ground.x      +0.75*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.25 * all_ensemble$blow_down.x        +0.75*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.25 * all_ensemble$agriculture.x      +0.75*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.25 * all_ensemble$road.x             +0.75*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.25 * all_ensemble$selective_logging.x+0.75*all_ensemble$selective_logging.y   )

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.Inception3_resnet50_aug.fold10-test.csv', row.names=FALSE, quote = FALSE)

##################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.dense161_02.fold10.csv") 
model02 <- read_csv("./submissions/Prav.Inception3_resnet50.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.5 * all_ensemble$slash_burn.x       +0.5*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.5 * all_ensemble$clear.x            +0.5*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.5 * all_ensemble$blooming.x         +0.5*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.5 * all_ensemble$primary.x          +0.5*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.5 * all_ensemble$cloudy.x           +0.5*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.5 * all_ensemble$conventional_mine.x+0.5*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.5 * all_ensemble$water.x            +0.5*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.5 * all_ensemble$haze.x             +0.5*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.5 * all_ensemble$cultivation.x      +0.5*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.5 * all_ensemble$partly_cloudy.x    +0.5*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.5 * all_ensemble$artisinal_mine.x   +0.5*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.5 * all_ensemble$habitation.x       +0.5*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.5 * all_ensemble$bare_ground.x      +0.5*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.5 * all_ensemble$blow_down.x        +0.5*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.5 * all_ensemble$agriculture.x      +0.5*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.5 * all_ensemble$road.x             +0.5*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.5 * all_ensemble$selective_logging.x+0.5*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense161_Inc3res50.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.dense161_02_aug.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.Inception3_resnet50_aug.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.5 * all_ensemble$slash_burn.x       +0.5*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.5 * all_ensemble$clear.x            +0.5*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.5 * all_ensemble$blooming.x         +0.5*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.5 * all_ensemble$primary.x          +0.5*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.5 * all_ensemble$cloudy.x           +0.5*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.5 * all_ensemble$conventional_mine.x+0.5*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.5 * all_ensemble$water.x            +0.5*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.5 * all_ensemble$haze.x             +0.5*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.5 * all_ensemble$cultivation.x      +0.5*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.5 * all_ensemble$partly_cloudy.x    +0.5*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.5 * all_ensemble$artisinal_mine.x   +0.5*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.5 * all_ensemble$habitation.x       +0.5*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.5 * all_ensemble$bare_ground.x      +0.5*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.5 * all_ensemble$blow_down.x        +0.5*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.5 * all_ensemble$agriculture.x      +0.5*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.5 * all_ensemble$road.x             +0.5*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.5 * all_ensemble$selective_logging.x+0.5*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.dense161_Inc3res50_aug.fold10-test.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.dense161_Inc3res50.fold10.csv") 
model02 <- read_csv("./submissions/Prav.dense169_Inc4.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.3 * all_ensemble$slash_burn.x       +0.7*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.3 * all_ensemble$clear.x            +0.7*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.3 * all_ensemble$blooming.x         +0.7*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.3 * all_ensemble$primary.x          +0.7*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.3 * all_ensemble$cloudy.x           +0.7*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.3 * all_ensemble$conventional_mine.x+0.7*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.3 * all_ensemble$water.x            +0.7*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.3 * all_ensemble$haze.x             +0.7*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.3 * all_ensemble$cultivation.x      +0.7*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.3 * all_ensemble$partly_cloudy.x    +0.7*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.3 * all_ensemble$artisinal_mine.x   +0.7*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.3 * all_ensemble$habitation.x       +0.7*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.3 * all_ensemble$bare_ground.x      +0.7*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.3 * all_ensemble$blow_down.x        +0.7*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.3 * all_ensemble$agriculture.x      +0.7*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.3 * all_ensemble$road.x             +0.7*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.3 * all_ensemble$selective_logging.x+0.7*all_ensemble$selective_logging.y   )

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.twoaug2.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.dense161_Inc3res50_aug.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.dense169_Inc4_aug.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.3 * all_ensemble$slash_burn.x       +0.7*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.3 * all_ensemble$clear.x            +0.7*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.3 * all_ensemble$blooming.x         +0.7*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.3 * all_ensemble$primary.x          +0.7*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.3 * all_ensemble$cloudy.x           +0.7*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.3 * all_ensemble$conventional_mine.x+0.7*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.3 * all_ensemble$water.x            +0.7*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.3 * all_ensemble$haze.x             +0.7*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.3 * all_ensemble$cultivation.x      +0.7*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.3 * all_ensemble$partly_cloudy.x    +0.7*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.3 * all_ensemble$artisinal_mine.x   +0.7*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.3 * all_ensemble$habitation.x       +0.7*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.3 * all_ensemble$bare_ground.x      +0.7*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.3 * all_ensemble$blow_down.x        +0.7*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.3 * all_ensemble$agriculture.x      +0.7*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.3 * all_ensemble$road.x             +0.7*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.3 * all_ensemble$selective_logging.x+0.7*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.twoaug2_aug.fold10-test.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.top2models.fold10.csv") 
model02 <- read_csv("./submissions/Prav.twoaug2.fold10.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.5 * all_ensemble$slash_burn.x       +0.5*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.5 * all_ensemble$clear.x            +0.5*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.5 * all_ensemble$blooming.x         +0.5*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.5 * all_ensemble$primary.x          +0.5*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.5 * all_ensemble$cloudy.x           +0.5*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.5 * all_ensemble$conventional_mine.x+0.5*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.5 * all_ensemble$water.x            +0.5*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.5 * all_ensemble$haze.x             +0.5*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.5 * all_ensemble$cultivation.x      +0.5*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.5 * all_ensemble$partly_cloudy.x    +0.5*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.5 * all_ensemble$artisinal_mine.x   +0.5*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.5 * all_ensemble$habitation.x       +0.5*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.5 * all_ensemble$bare_ground.x      +0.5*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.5 * all_ensemble$blow_down.x        +0.5*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.5 * all_ensemble$agriculture.x      +0.5*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.5 * all_ensemble$road.x             +0.5*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.5 * all_ensemble$selective_logging.x+0.5*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.top2ensemble2.fold10.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

##################################################################################################

model01 <- read_csv("./submissions/Prav.top2models.fold10-test.csv") 
model02 <- read_csv("./submissions/Prav.twoaug2_aug.fold10-test.csv")

all_ensemble <- left_join(model01, model02, by = "image_name")

ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- (0.5 * all_ensemble$slash_burn.x       +0.5*all_ensemble$slash_burn.y          )
all_ensemble$clear                <- (0.5 * all_ensemble$clear.x            +0.5*all_ensemble$clear.y               )
all_ensemble$blooming             <- (0.5 * all_ensemble$blooming.x         +0.5*all_ensemble$blooming.y            )
all_ensemble$primary              <- (0.5 * all_ensemble$primary.x          +0.5*all_ensemble$primary.y             )
all_ensemble$cloudy               <- (0.5 * all_ensemble$cloudy.x           +0.5*all_ensemble$cloudy.y              )
all_ensemble$conventional_mine    <- (0.5 * all_ensemble$conventional_mine.x+0.5*all_ensemble$conventional_mine.y   )
all_ensemble$water                <- (0.5 * all_ensemble$water.x            +0.5*all_ensemble$water.y               )
all_ensemble$haze                 <- (0.5 * all_ensemble$haze.x             +0.5*all_ensemble$haze.y                )
all_ensemble$cultivation          <- (0.5 * all_ensemble$cultivation.x      +0.5*all_ensemble$cultivation.y         )
all_ensemble$partly_cloudy        <- (0.5 * all_ensemble$partly_cloudy.x    +0.5*all_ensemble$partly_cloudy.y       )
all_ensemble$artisinal_mine       <- (0.5 * all_ensemble$artisinal_mine.x   +0.5*all_ensemble$artisinal_mine.y      )
all_ensemble$habitation           <- (0.5 * all_ensemble$habitation.x       +0.5*all_ensemble$habitation.y          )
all_ensemble$bare_ground          <- (0.5 * all_ensemble$bare_ground.x      +0.5*all_ensemble$bare_ground.y         )
all_ensemble$blow_down            <- (0.5 * all_ensemble$blow_down.x        +0.5*all_ensemble$blow_down.y           )
all_ensemble$agriculture          <- (0.5 * all_ensemble$agriculture.x      +0.5*all_ensemble$agriculture.y         )
all_ensemble$road                 <- (0.5 * all_ensemble$road.x             +0.5*all_ensemble$road.y                )
all_ensemble$selective_logging    <- (0.5 * all_ensemble$selective_logging.x+0.5*all_ensemble$selective_logging.y   )
5
head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.top2ensemble2_aug.fold10-test.csv', row.names=FALSE, quote = FALSE)
############################################################################################################################

















############################################################################################################################
rm(list=ls())

model01 <- read_csv("./submissions/Prav.resnet50_01.fold1-test.csv") 
model02 <- read_csv("./submissions/Prav.resnet50_01_r.fold1-test.csv")
model03 <- read_csv("./submissions/Prav.resnet50_01_h.fold1-test.csv") 

all_ensemble <- left_join(model01, model02, by = "image_name")
all_ensemble <- left_join(all_ensemble, model03, by = "image_name")
ensemble.features <- setdiff(names(all_ensemble),"image_name")
names(all_ensemble)

head(model01)
head(model02)
head(model03)
head(all_ensemble)

cor(all_ensemble[, ensemble.features])


all_ensemble$slash_burn           <- ( all_ensemble$slash_burn.x       +all_ensemble$slash_burn.y       + all_ensemble$slash_burn       )/3
all_ensemble$clear                <- ( all_ensemble$clear.x            +all_ensemble$clear.y            + all_ensemble$clear            )/3
all_ensemble$blooming             <- ( all_ensemble$blooming.x         +all_ensemble$blooming.y         + all_ensemble$blooming         )/3
all_ensemble$primary              <- ( all_ensemble$primary.x          +all_ensemble$primary.y          + all_ensemble$primary          )/3
all_ensemble$cloudy               <- ( all_ensemble$cloudy.x           +all_ensemble$cloudy.y           + all_ensemble$cloudy           )/3
all_ensemble$conventional_mine    <- ( all_ensemble$conventional_mine.x+all_ensemble$conventional_mine.y+ all_ensemble$conventional_mine)/3
all_ensemble$water                <- ( all_ensemble$water.x            +all_ensemble$water.y            + all_ensemble$water            )/3
all_ensemble$haze                 <- ( all_ensemble$haze.x             +all_ensemble$haze.y             + all_ensemble$haze             )/3
all_ensemble$cultivation          <- ( all_ensemble$cultivation.x      +all_ensemble$cultivation.y      + all_ensemble$cultivation      )/3
all_ensemble$partly_cloudy        <- ( all_ensemble$partly_cloudy.x    +all_ensemble$partly_cloudy.y    + all_ensemble$partly_cloudy    )/3
all_ensemble$artisinal_mine       <- ( all_ensemble$artisinal_mine.x   +all_ensemble$artisinal_mine.y   + all_ensemble$artisinal_mine   )/3
all_ensemble$habitation           <- ( all_ensemble$habitation.x       +all_ensemble$habitation.y       + all_ensemble$habitation       )/3
all_ensemble$bare_ground          <- ( all_ensemble$bare_ground.x      +all_ensemble$bare_ground.y      + all_ensemble$bare_ground      )/3
all_ensemble$blow_down            <- ( all_ensemble$blow_down.x        +all_ensemble$blow_down.y        + all_ensemble$blow_down        )/3
all_ensemble$agriculture          <- ( all_ensemble$agriculture.x      +all_ensemble$agriculture.y      + all_ensemble$agriculture      )/3
all_ensemble$road                 <- ( all_ensemble$road.x             +all_ensemble$road.y             + all_ensemble$road             )/3
all_ensemble$selective_logging    <- ( all_ensemble$selective_logging.x+all_ensemble$selective_logging.y+ all_ensemble$selective_logging)/3

head(all_ensemble)

cols <- c("image_name","slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging")

Ensemble <- all_ensemble[, cols]

head(all_ensemble)
head(Ensemble)


write.csv(Ensemble, './submissions/Prav.resnet50_01_aug.fold1.csv', row.names=FALSE, quote = FALSE)

###############################################################################################################################

############################################################################################################################
rm(list=ls())

model01 <- read_csv("./submissions/Prav.resnet50_01_aug.fold1flag-test-sub.csv") 
model02 <- read_csv("./submissions/Prav.resnet50_01_aug.fold2flag-test-sub.csv")
model03 <- read_csv("./submissions/Prav.resnet50_01_aug.fold3flag-test-sub.csv") 

model04 <- read_csv("./submissions/Prav.resnet50_01_aug.fold4flag-test-sub.csv") 
model05 <- read_csv("./submissions/Prav.resnet50_01_aug.fold5flag-test-sub.csv")
model06 <- read_csv("./submissions/Prav.resnet50_01_aug.fold6flag-test-sub.csv") 

model07 <- read_csv("./submissions/Prav.resnet50_01_aug.fold7flag-test-sub.csv") 
model08 <- read_csv("./submissions/Prav.resnet50_01_aug.fold8flag-test-sub.csv")
model09 <- read_csv("./submissions/Prav.resnet50_01_aug.fold9flag-test-sub.csv") 


models <- rbind(model01, model02, model03, model04, model05, model06, model07, model08, model09)

MeanModels <- models %>%
              group_by(image_name) %>%
              summarise(  slash_burn = sum(slash_burn)
                        , clear      = sum(clear)
                        , blooming   = sum(blooming)
                        , primary    = sum(primary)
                        , cloudy     = sum(cloudy)
                        , conventional_mine = sum(conventional_mine)
                        , water  = sum(water)
                        , haze  = sum(haze)
                        , cultivation = sum(cultivation)
                        , partly_cloudy = sum(partly_cloudy)
                        , artisinal_mine = sum(artisinal_mine)
                        , habitation  = sum(habitation)
                        , bare_ground  = sum(bare_ground)
                        , blow_down   = sum(blow_down)
                        , agriculture = sum(agriculture)
                        , road = sum(road)
                        , selective_logging = sum(selective_logging)
                        
                        
                        )

head(MeanModels)

MeanModels_Update <- MeanModels

MeanModels_Update$slash_burn <- ifelse(MeanModels_Update$slash_burn >= 5, 1, 0)
MeanModels_Update$clear      <- ifelse(MeanModels_Update$clear >= 5, 1, 0)
MeanModels_Update$blooming   <- ifelse(MeanModels_Update$blooming >= 5, 1, 0)

MeanModels_Update$primary    <- ifelse(MeanModels_Update$primary >= 5, 1, 0)
MeanModels_Update$cloudy     <- ifelse(MeanModels_Update$cloudy >= 5, 1, 0)
MeanModels_Update$conventional_mine <- ifelse(MeanModels_Update$conventional_mine >= 5, 1, 0)

MeanModels_Update$water      <- ifelse(MeanModels_Update$water >= 5, 1, 0)
MeanModels_Update$haze       <- ifelse(MeanModels_Update$haze >= 5, 1, 0)
MeanModels_Update$cultivation  <- ifelse(MeanModels_Update$cultivation >= 5, 1, 0)

MeanModels_Update$partly_cloudy      <- ifelse(MeanModels_Update$partly_cloudy >= 5, 1, 0)
MeanModels_Update$artisinal_mine       <- ifelse(MeanModels_Update$artisinal_mine >= 5, 1, 0)
MeanModels_Update$habitation  <- ifelse(MeanModels_Update$habitation >= 5, 1, 0)

MeanModels_Update$bare_ground      <- ifelse(MeanModels_Update$bare_ground >= 5, 1, 0)
MeanModels_Update$blow_down       <- ifelse(MeanModels_Update$blow_down >= 5, 1, 0)
MeanModels_Update$agriculture  <- ifelse(MeanModels_Update$agriculture >= 5, 1, 0)

MeanModels_Update$road       <- ifelse(MeanModels_Update$road >= 5, 1, 0)
MeanModels_Update$selective_logging  <- ifelse(MeanModels_Update$selective_logging >= 5, 1, 0)
                
head(MeanModels_Update)

write.csv(MeanModels_Update, './submissions/Prav.resnet50_01_aug.fold0109.csv', row.names=FALSE, quote = FALSE)

###############################################################################################################################

############################################################################################################################
rm(list=ls())

model01 <- read_csv("./submissions/Prav.resnet50_01_aug.fold0109.csv") 
# model02 <- read_csv("./submissions/Prav.resnet50_01_aug.fold2flag-test-sub.csv")
# model03 <- read_csv("./submissions/Prav.resnet50_01_aug.fold3flag-test-sub.csv") 
# 
# model04 <- read_csv("./submissions/Prav.resnet50_01_aug.fold4flag-test-sub.csv") 
# model05 <- read_csv("./submissions/Prav.resnet50_01_aug.fold5flag-test-sub.csv")
# model06 <- read_csv("./submissions/Prav.resnet50_01_aug.fold6flag-test-sub.csv") 
# 
# model07 <- read_csv("./submissions/Prav.resnet50_01_aug.fold7flag-test-sub.csv") 
# model08 <- read_csv("./submissions/Prav.resnet50_01_aug.fold8flag-test-sub.csv")
# model09 <- read_csv("./submissions/Prav.resnet50_01_aug.fold9flag-test-sub.csv") 

model10 <- read_csv("./submissions/Prav.dense169.fold10flag-test-sub.csv") 
model11 <- read_csv("./submissions/Prav.dense169_02.fold10flag-test-sub.csv") 
model12 <- read_csv("./submissions/Prav.Inception4_01.fold10flag-test-sub.csv") 

model13 <- read_csv("./submissions/Prav.resnet152_01.fold10flag-test-sub.csv") 
model14 <- read_csv("./submissions/Prav.dense169_03.fold10flag-test-sub.csv") 
model15 <- read_csv("./submissions/Prav.dense169_04.fold10flag-test-sub.csv") 

model16 <- read_csv("./submissions/Prav.Inception4_02.fold10flag-test-sub.csv") 
model17 <- read_csv("./submissions/Prav.resnet50_01.fold10flag-test-sub.csv") 
model18 <- read_csv("./submissions/Prav.Inceptionv3_01.fold10flag-test-sub.csv") 

model19 <- read_csv("./submissions/Prav.dense161_02.fold10flag-test-sub.csv") 
model20 <- read_csv("./submissions/Prav.dense161.fold10flag-test-sub.csv") 





models <- rbind(model01,
                model10, model11, model12, model13, model14, model15, model16, model17, model18, model19, model20
                )

MeanModels <- models %>%
  group_by(image_name) %>%
  summarise(  slash_burn = sum(slash_burn)
              , clear      = sum(clear)
              , blooming   = sum(blooming)
              , primary    = sum(primary)
              , cloudy     = sum(cloudy)
              , conventional_mine = sum(conventional_mine)
              , water  = sum(water)
              , haze  = sum(haze)
              , cultivation = sum(cultivation)
              , partly_cloudy = sum(partly_cloudy)
              , artisinal_mine = sum(artisinal_mine)
              , habitation  = sum(habitation)
              , bare_ground  = sum(bare_ground)
              , blow_down   = sum(blow_down)
              , agriculture = sum(agriculture)
              , road = sum(road)
              , selective_logging = sum(selective_logging)
              
              
  )

head(MeanModels)

MeanModels_Update <- MeanModels

MeanModels_Update$slash_burn <- ifelse(MeanModels_Update$slash_burn >= 6, 1, 0)
MeanModels_Update$clear      <- ifelse(MeanModels_Update$clear >= 6, 1, 0)
MeanModels_Update$blooming   <- ifelse(MeanModels_Update$blooming >= 6, 1, 0)

MeanModels_Update$primary    <- ifelse(MeanModels_Update$primary >= 6, 1, 0)
MeanModels_Update$cloudy     <- ifelse(MeanModels_Update$cloudy >= 6, 1, 0)
MeanModels_Update$conventional_mine <- ifelse(MeanModels_Update$conventional_mine >= 6, 1, 0)

MeanModels_Update$water      <- ifelse(MeanModels_Update$water >=  6, 1, 0)
MeanModels_Update$haze       <- ifelse(MeanModels_Update$haze >=  6, 1, 0)
MeanModels_Update$cultivation  <- ifelse(MeanModels_Update$cultivation >=  6, 1, 0)

MeanModels_Update$partly_cloudy      <- ifelse(MeanModels_Update$partly_cloudy >=  6, 1, 0)
MeanModels_Update$artisinal_mine       <- ifelse(MeanModels_Update$artisinal_mine >=  6, 1, 0)
MeanModels_Update$habitation  <- ifelse(MeanModels_Update$habitation >=  6, 1, 0)

MeanModels_Update$bare_ground      <- ifelse(MeanModels_Update$bare_ground >=  6, 1, 0)
MeanModels_Update$blow_down       <- ifelse(MeanModels_Update$blow_down >=  6, 1, 0)
MeanModels_Update$agriculture  <- ifelse(MeanModels_Update$agriculture >=  6, 1, 0)

MeanModels_Update$road       <- ifelse(MeanModels_Update$road >=  6, 1, 0)
MeanModels_Update$selective_logging  <- ifelse(MeanModels_Update$selective_logging >=  6, 1, 0)

head(MeanModels_Update)

write.csv(MeanModels_Update, './submissions/Prav.allmodels.fold0110.csv', row.names=FALSE, quote = FALSE)


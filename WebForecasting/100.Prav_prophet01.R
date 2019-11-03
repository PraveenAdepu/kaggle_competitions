
rm(list=ls())
# fast data manipulation
library(data.table)     
library(readr)
# read in the data 
key_1   <- read_csv("./input/key_1.csv")
train_1 <- read_csv("./input/train_1.csv")

head(train_1)
# extract a lists  of pages and dates
train.date.cols = setdiff(names(train_1),"Page")

# reshape the training data into long format page, date and views
dt <- melt(train_1,
           d.vars = c("Page"),
           measure.vars = train.date.cols,
           variable.name = "ds",
           value.name = "y")
head(dt)
# replace NAs with 0 and calculate page median
dt[is.na(y), y :=0]
dt_sum <- dt[,.(Visits = median(y)) , by = Page]
setkey(dt_sum, Page)

# merge projection dates and key to create submission
key_1[, Page2:= substr(Page, 1, nchar(Page)-11)]
key_1[, Page:= NULL]
setnames(key_1, "Page2", "Page")
setkey(key_1, Page)

sub <- merge(key_1, dt_sum, all.x = TRUE)

write.csv(sub[,c('Id', 'Visits')], file=gzfile("sub_median.csv.gz"), row.names = FALSE)




set.seed(2016)


train  <- fread("./input/clicks_train.csv") 
names(train)

# views  <- fread("./input/page_views_sample.csv") 
# names(views)
# head(views)

events <- fread("./input/events.csv") 
names(events)
head(events)

train_events <- merge(train, events, by="display_id", all.x=TRUE) #87141731


train_events$Date =  as.POSIXct((train_events$timestamp+1465876799998)/1000, origin="1970-01-01", tz="UTC")

gc()
#train_events$hours = train_events$Date(strftime(train_events$Date, format="%T"))

t.lub <- ymd_hms(train_events$Date)

h.lub <- hour(t.lub) + minute(t.lub)/60
d.lub <- day(t.lub)
m.lub <- minute(t.lub)/60

head(h.lub)
head(d.lub)
head(m.lub)

train_events$hour    <- h.lub
train_events$day     <- d.lub
train_events$minutes <- m.lub
head(train_events)

train_events$day <- as.integer(train_events$day)

DisplayDays <-  sqldf("SELECT DISTINCT display_id, day FROM train_events")

head(DisplayDays)
gc()

#######################################################################################

unique(DisplayDays$day)

length(unique(train$display_id))

#######################################################################################
# 14 15 16 17 18 19 20 21 22 23 24 25 26 27
for (i in 14:27)
{
    cat(i, " - day processing folds\n")
    DailyDisplayDaysOriginal <- subset(DisplayDays, day == i)
    
    # Random shuffle dataset row wise
    DailyDisplayDays <- DailyDisplayDaysOriginal[sample(nrow(DailyDisplayDaysOriginal)),]
    
    folds <- createFolds(DailyDisplayDays$display_id, k = 5)
    
    split_up <- lapply(folds, function(ind, dat) dat[ind,], dat = DailyDisplayDays)
    dim(DailyDisplayDays)
    unlist(lapply(split_up, nrow))
    
    trainingFold01 <- as.data.frame(DailyDisplayDays[folds$Fold1, ])
    trainingFold01$CVindices <- 1
    
    trainingFold02 <- as.data.frame(DailyDisplayDays[folds$Fold2, ])
    trainingFold02$CVindices <- 2
    
    trainingFold03 <- as.data.frame(DailyDisplayDays[folds$Fold3, ])
    trainingFold03$CVindices <- 3
    
    trainingFold04 <- as.data.frame(DailyDisplayDays[folds$Fold4, ])
    trainingFold04$CVindices <- 4
    
    trainingFold05 <- as.data.frame(DailyDisplayDays[folds$Fold5, ])
    trainingFold05$CVindices <- 5
    
    names(trainingFold01)[1] <- "display_id"
    names(trainingFold02)[1] <- "display_id"
    names(trainingFold03)[1] <- "display_id"
    names(trainingFold04)[1] <- "display_id"
    names(trainingFold05)[1] <- "display_id"
    
    names(trainingFold05)
    trainingFolds <- rbind(trainingFold01, trainingFold02 , trainingFold03, trainingFold04, trainingFold05 )
    if(i == 14)
        {
         AlltrainingFolds <-  trainingFolds
        }
    else{
         AlltrainingFolds <- rbind(AlltrainingFolds, trainingFolds)
        }

}
AlltrainingFolds <-  AlltrainingFolds[with(AlltrainingFolds, order(display_id)), ]
cat("All done!")



write.csv(AlltrainingFolds, paste(root_directory, "./CVSchema/Prav_CVindices_5folds.csv", sep=''), row.names=FALSE, quote = FALSE)
# 
# # Unit testing
# unique(trainingFolds$CVindices)
# sqldf("SELECT day, CVindices,  count(*) Count from AlltrainingFolds Group by day,CVindices")

# day CVindices  Count
# 1   14         1 239631
# 2   14         2 239630
# 3   14         3 239632
# 4   14         4 239632
# 5   14         5 239632
# 6   15         1 263057
# 7   15         2 263056
# 8   15         3 263057
# 9   15         4 263059
# 10  15         5 263058
# 11  16         1 265879
# 12  16         2 265880
# 13  16         3 265879
# 14  16         4 265879
# 15  16         5 265878
# 16  17         1 257141
# 17  17         2 257141
# 18  17         3 257142
# 19  17         4 257141
# 20  17         5 257143
# 21  18         1 225559
# 22  18         2 225558
# 23  18         3 225556
# 24  18         4 225559
# 25  18         5 225558
# 26  19         1 217931
# 27  19         2 217932
# 28  19         3 217930
# 29  19         4 217930
# 30  19         5 217930
# 31  20         1 283715
# 32  20         2 283714
# 33  20         3 283714
# 34  20         4 283715
# 35  20         5 283715
# 36  21         1 275087
# 37  21         2 275087
# 38  21         3 275088
# 39  21         4 275088
# 40  21         5 275087
# 41  22         1 275347
# 42  22         2 275348
# 43  22         3 275348
# 44  22         4 275348
# 45  22         5 275347
# 46  23         1 274215
# 47  23         2 274216
# 48  23         3 274216
# 49  23         4 274216
# 50  23         5 274216
# 51  24         1 272723
# 52  24         2 272724
# 53  24         3 272724
# 54  24         4 272723
# 55  24         5 272724
# 56  25         1 233492
# 57  25         2 233490
# 58  25         3 233492
# 59  25         4 233491
# 60  25         5 233492
# 61  26         1 244747
# 62  26         2 244746
# 63  26         3 244747
# 64  26         4 244747
# 65  26         5 244747
# 66  27         1  46394
# 67  27         2  46394
# 68  27         3  46393
# 69  27         4  46393
# 70  27         5  46393


rm(list=ls())

source('fn.base.r')
library(data.table)

train <- fread( './input/train.csv' )
head(train)
set.seed(1111)
head(train,10)

###################################################################################################################
#Build 5 folds CV indices##########################################################################################
###################################################################################################################

dt1 <- train[, paste(qid2, collapse=" ") , keyby="qid1"  ]
dt2 <- train[, paste(qid1, collapse=" ") , keyby="qid2"  ]
head(dt1)
head(dt2)
colnames(dt1) <- c("id","val")
colnames(dt2) <- c("id","val")
dt <- rbind(dt1,dt2)
dt <- dt[, paste(val, collapse=" ") , keyby="id"  ]
rm(dt1,dt2);gc()
head(dt,10)
max(dt$id)

dt$id <- as.integer(dt$id)

dtall <- data.table( id = 1:537933  )
dtall <- merge( dtall , dt, by="id", all=T  )
head(dtall,20)
tail(dtall,20)

ids <- dtall$V1
cvs <- rep( 0 , nrow(dtall )  )

find_reg <- function( ind1 ){
  tmp <- numeric()
  for( i in ind1){
    tmp <- c( tmp , as.numeric( unlist( strsplit(ids[i], " ") ) )  )
  }
  return( unique(tmp) )
}

ids <- dtall$V1
cvs <- rep( 0 , nrow(dtall )  )
set.seed(111)
i1=20
for( i1 in 1:length(ids) ){
  if( !is.na(ids[i1]) && (cvs[i1]==0) ){
    if( (i1 %% 10000)==0 ){
      print(i1)
    }
    i2 <- as.numeric( unlist( strsplit(ids[i1], " ") ) )
    i3 <- find_reg(i2)
    i4 <- find_reg(i3)
    i5 <- find_reg(i4)
    i6 <- find_reg(i5)
    i7 <- find_reg(i6)
    i8 <- find_reg(i7)
    i9 <- find_reg(i8)
    i10 <- find_reg(i9)
    i11 <- find_reg(i10)
    i12 <- find_reg(i11)
    alli <- unique(c(i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12))
    f <- cvs[ alli ]
    f <- f[ f!=0 ][1]
    if( is.na(f) ){
      f = floor( runif( 1, 0, 5 ) )+1
    }
    cvs[alli] <- f
  }
}
dtall[, cv := cvs ]
head(dtall,40)
table( dtall$cv  );gc()
setkeyv( dtall, "id"  )

train[, cv1:= dtall[J(as.integer(train$qid1))]$cv  ]
train[, cv2:= dtall[J(as.integer(train$qid2))]$cv  ]
head(train,40)
mean( train$cv1 == train$cv2 )
train[ train$cv1 != train$cv2 ]

train[, CVindices := cv1]
train[, cv1 := NULL ]
train[, cv2 := NULL ]


train[, question1 := NULL ]
train[, question2 := NULL ]


rename(count(train, CVindices, is_duplicate), Freq = n)


train[, is_duplicate := NULL ]
head(train)

write.table( train, './CVSchema/CVindices_5folds.csv' , row.names=F, quote=F, sep=","  )
table( train$CVindices ); gc()

# 1     2     3     4     5 
# 73205 94671 78579 79375 78460

# CVindices is_duplicate  Freq
# <dbl>        <chr> <int>
#   1          1          0 48131
# 2          1            1 25074
# 3          2            0 56725
# 4          2            1 37946
# 5          3            0 50021
# 6          3            1 28558
# 7          4            0 50187
# 8          4            1 29188
# 9          5            0 49963
# 10         5            1 28497


###################################################################################################################
###################################################################################################################
###################################################################################################################

# 
# id   qid1   qid2                                                                                                            question1
# 1:  70814 121971 121972                                                                  How do I write an effective and professional email?
# 2:  71855 123589 123590                                              Where can I watch Naruto Shippuden episodes that are dubbed in English?
# 3:  73457 126053 126054                                                            Is it possible to change the IMEI number of a cell phone?
# 4:  75395  42032 129049                                                               How do you write a polite reminder email to your boss?
# 5: 224030 332002 212581                                                                       How can I have a stable income while studying?
# 6: 260112   9041 376001                                                     Can I find or track my lost mobile device using the IMEI number?
# 7: 284500 404758 389203                                                                                  What are examples of trace fossils?
# 8: 285145 123589 109940                                              Where can I watch Naruto Shippuden episodes that are dubbed in English?
# 9: 293210 123589 414946                                              Where can I watch Naruto Shippuden episodes that are dubbed in English?
# 10: 304348 389203 339459 How is it possible for cops to trace a lost mobile using the IMEI number even after the SIM card has been taken out?
# 11: 324686  51422 450824                                              How do I write an email to a client reminding him of a meeting with me?
# 12: 327982  42032  51422                                                               How do you write a polite reminder email to your boss?
# 13: 348788 389203 168908 How is it possible for cops to trace a lost mobile using the IMEI number even after the SIM card has been taken out?
# 14: 360192 278785 233118                                                                Is Naruto Shippuden dubbed in English on Crunchyroll?
# question2 is_duplicate cv1 cv2
# 1:                                                                        How do I create a professional looking email?            1   4   2
# 2:                                                          Where do I find Naruto Shippuden 403-407 episodes to watch?            0   5   2
# 3:                                                                                          Can imei number be changed?            0   4   2
# 4:                                                                          How can I write an informal email politely?            0   2   4
# 5:                                                                                    Is there any simple reminder app?            0   2   4
# 6:                                                                 I have lost my mobile. How do I find my IMEI number?            0   4   2
# 7: How is it possible for cops to trace a lost mobile using the IMEI number even after the SIM card has been taken out?            0   4   2
# 8:                               When is the release of Naruto Shippuden Episode 306+ in English dub, and on what site?            0   5   2
# 9:                                                                    Where can I watch an English dub of Naruto OVA 2?            0   5   2
# 10:                                                           I lost my Moto G mobile yesterday. I know the IMEI number.            0   2   4
# 11:                                                                      How can you write a farewell email to a client?            0   4   2
# 12:                                              How do I write an email to a client reminding him of a meeting with me?            0   2   4
# 13:                                                                How do I find a sim card number using an IMEI number?            0   2   4
# 14:                                                                    Has Naruto Shippuden been dubbed in English? Why?            0   2   5





###################################################################################################################
#Build 20 folds CV indices##########################################################################################
###################################################################################################################
ids <- dtall$V1
cvs <- rep( 0 , nrow(dtall )  )

find_reg <- function( ind1 ){
  tmp <- numeric()
  for( i in ind1){
    tmp <- c( tmp , as.numeric( unlist( strsplit(ids[i], " ") ) )  )
  }
  return( unique(tmp) )
}

ids <- dtall$V1
cvs <- rep( 0 , nrow(dtall )  )
set.seed(1)
i1=20
for( i1 in 1:length(ids) ){
  if( !is.na(ids[i1]) && (cvs[i1]==0) ){
    if( (i1 %% 10000)==0 ){
      print(i1)
    }
    i2 <- as.numeric( unlist( strsplit(ids[i1], " ") ) )
    i3 <- find_reg(i2)
    i4 <- find_reg(i3)
    i5 <- find_reg(i4)
    i6 <- find_reg(i5)
    i7 <- find_reg(i6)
    i8 <- find_reg(i7)
    i9 <- find_reg(i8)
    i10 <- find_reg(i9)
    i11 <- find_reg(i10)
    i12 <- find_reg(i11)
    alli <- unique(c(i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12))
    f <- cvs[ alli ]
    f <- f[ f!=0 ][1]
    if( is.na(f) ){
      f = floor( runif( 1, 0, 20 ) )+1
    }
    cvs[alli] <- f
  }
}
dtall[, cv := cvs ]
head(dtall,40)
table( dtall$cv  );gc()
setkeyv( dtall, "id"  )

train[, cv1:= dtall[J(train$itemID_1)]$cv  ]
train[, cv2:= dtall[J(train$itemID_2)]$cv  ]
head(train,40)
mean( train$cv1 == train$cv2 )
train[ train$cv1 != train$cv2 ]

train[, CVindices := cv1]
train[, cv1 := NULL ]
train[, cv2 := NULL ]
train[, generationMethod := NULL ]
train[, isDuplicate := NULL ]
write.table( train, '../input/CVindices20.csv' , row.names=F, quote=F, sep=","  )
###################################################################################################################
###################################################################################################################
###################################################################################################################



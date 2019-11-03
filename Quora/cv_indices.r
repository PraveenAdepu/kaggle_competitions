rm(list=ls())

source('fn.base.r')
library(data.table)

train <- fread( '../input/ItemPairs_train.csv' )
head(train)
set.seed(1111)
head(train,10)

###################################################################################################################
#Build 5 folds CV indices##########################################################################################
###################################################################################################################
dt1 <- train[, paste(itemID_2, collapse=" ") , keyby="itemID_1"  ]
dt2 <- train[, paste(itemID_1, collapse=" ") , keyby="itemID_2"  ]
colnames(dt1) <- c("id","val")
colnames(dt2) <- c("id","val")
dt <- rbind(dt1,dt2)
dt <- dt[, paste(val, collapse=" ") , keyby="id"  ]
rm(dt1,dt2);gc()
head(dt,10)

dtall <- data.table( id = 1:6112003  )
dtall <- merge( dtall , dt, by="id", all=T  )
head(dtall,20)


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
write.table( train, '../input/CVindices_5folds.csv' , row.names=F, quote=F, sep=","  )
table( train$CVindices ); gc()
###################################################################################################################
###################################################################################################################
###################################################################################################################








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



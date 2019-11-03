data("MovieLense", package = "recommenderlab")

# go through some pain to convert to a data.frame
user_to_df = function(i) {
  lst = as(MovieLense, "list")[[i]]
  data.frame(User = i, Movie = names(lst), Rating = as.numeric(lst), stringsAsFactors = FALSE)
}

rating_dfs = lapply(1:nrow(MovieLense), user_to_df)
movie_lens = do.call(rbind, rating_dfs)

head(movie_lens)
movie_lens$User = factor(movie_lens$User)
movie_lens$Movie = factor(movie_lens$Movie)

save(movie_lens, file = "./movie_lens.rdata", compress = "xz")

library(libFMexe)

set.seed(1)
train_rows = sample.int(nrow(movie_lens), nrow(movie_lens) * 2 / 3)
train = movie_lens[train_rows, ]
test  = movie_lens[-train_rows, ]

form = "Rating ~ User + Movie"
predFM = libFM(train, test, Rating ~ User + Movie,
               task = "r", dim = 10, iter = 500
               , exe_loc = "C:\\Users\\padepu\\Documents\\R\\11RedHat\\LibFM")


mean((predFM - test$Rating)^2)

pred_cv = libFM(X_build[, feature.names],
                X_val[, feature.names], 
                outcome ~ 
                  char_10        +   activity_category + char_1           + char_2           + char_3        +  
                  char_4         +   char_5            + char_6           + char_7           + char_8        +  
                  char_9         +   char10peopleCount + p_group_1        + p_char_1         +          
                  p_char_2       +   p_char_3          + p_char_4         + p_char_5         + p_char_6      +  
                  p_char_7       +   p_char_8          + p_char_9         + p_char_10        + p_char_11     +  
                  p_char_12      +   p_char_13         + p_char_14        + p_char_15        + p_char_16     +  
                  p_char_17      +   p_char_18         + p_char_19        + p_char_20        + p_char_21     +  
                  p_char_22      +   p_char_23         + p_char_24        + p_char_25        + p_char_26     +  
                  p_char_27      +   p_char_28         + p_char_29        + p_char_30        + p_char_31     +  
                  p_char_32      +   p_char_33         + p_char_34        + p_char_35        + p_char_36     +  
                  p_char_37      +   p_char_38          ,
                task = "r", dim = 10, iter = 1000, verbosity = 1, seed = 2016
                , exe_loc = "C:\\Users\\padepu\\Documents\\R\\11RedHat\\LibFM")

###############################################################################

train <- read_csv('./input/train.csv')
test  <- read_csv('./input/test.csv')

CVindices5folds <- read_csv("./CVSchema/Prav_CVindices_5folds.csv")
names(train)
summary(train$loss)
names(test)
test$loss <- -100
summary(test$loss)

trainingSet <- left_join(train, CVindices5folds, by = "id")
test$CVindices <- 0
testingSet  <- test

feature.names <- names(trainingSet[,-which(names(trainingSet) %in% c( "id","CVindices","loss"
))])

cont.names     <- grep("cont", names(trainingSet), value = T)

feature.names  <- setdiff(feature.names, cont.names)


for (f in feature.names) {
  
    cat("VARIABLE : ",f,"\n")
    trainingSet[[f]] <- as.factor(trainingSet[[f]])
    testingSet[[f]]  <- as.factor(testingSet[[f]])
  }

feature.names <- names(trainingSet[,-which(names(trainingSet) %in% c( "id","CVindices"
))])

cont.names     <- grep("cont", names(trainingSet), value = T)

feature.names  <- setdiff(feature.names, cont.names)

formula <-  as.formula(paste(" loss ~ ", paste(feature.names, collapse= "+")))

loss ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + 
  cat9 + cat10 + cat11 + cat12 + cat13 + cat14 + cat15 + cat16 + 
  cat17 + cat18 + cat19 + cat20 + cat21 + cat22 + cat23 + cat24 + 
  cat25 + cat26 + cat27 + cat28 + cat29 + cat30 + cat31 + cat32 + 
  cat33 + cat34 + cat35 + cat36 + cat37 + cat38 + cat39 + cat40 + 
  cat41 + cat42 + cat43 + cat44 + cat45 + cat46 + cat47 + cat48 + 
  cat49 + cat50 + cat51 + cat52 + cat53 + cat54 + cat55 + cat56 + 
  cat57 + cat58 + cat59 + cat60 + cat61 + cat62 + cat63 + cat64 + 
  cat65 + cat66 + cat67 + cat68 + cat69 + cat70 + cat71 + cat72 + 
  cat73 + cat74 + cat75 + cat76 + cat77 + cat78 + cat79 + cat80 + 
  cat81 + cat82 + cat83 + cat84 + cat85 + cat86 + cat87 + cat88 + 
  cat89 + cat90 + cat91 + cat92 + cat93 + cat94 + cat95 + cat96 + 
  cat97 + cat98 + cat99 + cat100 + cat101 + cat102 + cat103 + 
  cat104 + cat105 + cat106 + cat107 + cat108 + cat109 + cat110 + 
  cat111 + cat112 + cat113 + cat114 + cat115 + cat116 + cont1 + 
  cont2 + cont3 + cont4 + cont5 + cont6 + cont7 + cont8 + cont9 + 
  cont10 + cont11 + cont12 + cont13 + cont14 

trainingSet$loss <- log(trainingSet$loss)
i = 5
X_build <- subset(trainingSet, CVindices != i, select=-c(CVindices))
X_val   <- subset(trainingSet, CVindices == i, select=-c(CVindices))

X_train <- X_build[, feature.names]
X_valid <- X_val[,feature.names]
# X_train$loss <- log(X_train$loss)
# X_valid$loss <- log(X_valid$loss)



seed = 2016
set.seed(seed)

predFM = libFM(X_train, X_valid, loss ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + 
                 cat9 + cat10 + cat11 + cat12 + cat13 + cat14 + cat15 + cat16 + 
                 cat17 + cat18 + cat19 + cat20 + cat21 + cat22 + cat23 + cat24 + 
                 cat25 + cat26 + cat27 + cat28 + cat29 + cat30 + cat31 + cat32 + 
                 cat33 + cat34 + cat35 + cat36 + cat37 + cat38 + cat39 + cat40 + 
                 cat41 + cat42 + cat43 + cat44 + cat45 + cat46 + cat47 + cat48 + 
                 cat49 + cat50 + cat51 + cat52 + cat53 + cat54 + cat55 + cat56 + 
                 cat57 + cat58 + cat59 + cat60 + cat61 + cat62 + cat63 + cat64 + 
                 cat65 + cat66 + cat67 + cat68 + cat69 + cat70 + cat71 + cat72 + 
                 cat73 + cat74 + cat75 + cat76 + cat77 + cat78 + cat79 + cat80 + 
                 cat81 + cat82 + cat83 + cat84 + cat85 + cat86 + cat87 + cat88 + 
                 cat89 + cat90 + cat91 + cat92 + cat93 + cat94 + cat95 + cat96 + 
                 cat97 + cat98 + cat99 + cat100 + cat101 + cat102 + cat103 + 
                 cat104 + cat105 + cat106 + cat107 + cat108 + cat109 + cat110 + 
                 cat111 + cat112 + cat113 + cat114 + cat115 + cat116  ,
               task = "r", dim = 3, iter = 500
               , exe_loc = "C:\\Users\\padepu\\Documents\\R\\11RedHat\\LibFM")

score(predFM,X_val$loss,metric) # [1] 1279.474, [1] 1261.325, dim = 0, [1] 1978.766
# dim = 3, iter = 500  -- [1] 1228.722
# dim = 3, iter = 2000 -- [1] 1228.066
score(exp(predFM),exp(X_val$loss),metric) 

mean((predFM - test$Rating)^2)


feature.names <- names(trainingSet[,-which(names(trainingSet) %in% c( "id","CVindices"
))])

cont.names     <- grep("cont", names(trainingSet), value = T)

feature.names  <- setdiff(feature.names, cont.names)


X_build_mat = Matrix::sparse.model.matrix(loss ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + 
                                            cat9 + cat10 + cat11 + cat12 + cat13 + cat14 + cat15 + cat16 + 
                                            cat17 + cat18 + cat19 + cat20 + cat21 + cat22 + cat23 + cat24 + 
                                            cat25 + cat26 + cat27 + cat28 + cat29 + cat30 + cat31 + cat32 + 
                                            cat33 + cat34 + cat35 + cat36 + cat37 + cat38 + cat39 + cat40 + 
                                            cat41 + cat42 + cat43 + cat44 + cat45 + cat46 + cat47 + cat48 + 
                                            cat49 + cat50 + cat51 + cat52 + cat53 + cat54 + cat55 + cat56 + 
                                            cat57 + cat58 + cat59 + cat60 + cat61 + cat62 + cat63 + cat64 + 
                                            cat65 + cat66 + cat67 + cat68 + cat69 + cat70 + cat71 + cat72 + 
                                            cat73 + cat74 + cat75 + cat76 + cat77 + cat78 + cat79 + cat80 + 
                                            cat81 + cat82 + cat83 + cat84 + cat85 + cat86 + cat87 + cat88 + 
                                            cat89 + cat90 + cat91 + cat92 + cat93 + cat94 + cat95 + cat96 + 
                                            cat97 + cat98 + cat99 + cat100 + cat101 + cat102 + cat103 + 
                                            cat104 + cat105 + cat106 + cat107 + cat108 + cat109 + cat110 + 
                                            cat111 + cat112 + cat113 + cat114 + cat115 + cat116  - 1, X_build)
X_valid_mat = Matrix::sparse.model.matrix(loss ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + 
                                          cat9 + cat10 + cat11 + cat12 + cat13 + cat14 + cat15 + cat16 + 
                                          cat17 + cat18 + cat19 + cat20 + cat21 + cat22 + cat23 + cat24 + 
                                          cat25 + cat26 + cat27 + cat28 + cat29 + cat30 + cat31 + cat32 + 
                                          cat33 + cat34 + cat35 + cat36 + cat37 + cat38 + cat39 + cat40 + 
                                          cat41 + cat42 + cat43 + cat44 + cat45 + cat46 + cat47 + cat48 + 
                                          cat49 + cat50 + cat51 + cat52 + cat53 + cat54 + cat55 + cat56 + 
                                          cat57 + cat58 + cat59 + cat60 + cat61 + cat62 + cat63 + cat64 + 
                                          cat65 + cat66 + cat67 + cat68 + cat69 + cat70 + cat71 + cat72 + 
                                          cat73 + cat74 + cat75 + cat76 + cat77 + cat78 + cat79 + cat80 + 
                                          cat81 + cat82 + cat83 + cat84 + cat85 + cat86 + cat87 + cat88 + 
                                          cat89 + cat90 + cat91 + cat92 + cat93 + cat94 + cat95 + cat96 + 
                                          cat97 + cat98 + cat99 + cat100 + cat101 + cat102 + cat103 + 
                                          cat104 + cat105 + cat106 + cat107 + cat108 + cat109 + cat110 + 
                                          cat111 + cat112 + cat113 + cat114 + cat115 + cat116  - 1, X_valid)
pred_cv_bags   <- rep(0, nrow(X_val[, feature.names]))
bags = 2
for (b in 1:bags) 
{
  cat(b ," - bag Processing\n")
predFM = libFM(X_build_mat, X_valid_mat, X_build$loss, X_valid$loss
               , task = "r"
               , dim = 10
               #,regular = c(1, 1, 8)
               #,init_stdev = 0.001
               , verbosity = 1 
               , iter = 10
               , method = "mcmc"
               #, validation = X_valid
               , exe_loc = "C:\\Users\\padepu\\Documents\\R\\11RedHat\\LibFM"
        )
 cat("bag -", b, " ", metric, ": ", score( exp(predFM),exp(X_val$loss), metric), "\n", sep = "")
pred_cv_bags <- pred_cv_bags + exp(predFM)
}
pred_cv_bags <- pred_cv_bags / bags
cat("CV Fold -", i, " ", metric, ": ", score(pred_cv_bags,exp(X_val$loss),metric), "\n", sep = "")


head(pred_cv_bags)
head(X_val$loss)


bags = 1
for (b in 1:bags) 
{
cat(b ," - bag Processing\n")
predFM = libFM(X_train_mat, X_valid_mat, loss ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + 
                 cat9 + cat10 + cat11 + cat12 + cat13 + cat14 + cat15 + cat16 + 
                 cat17 + cat18 + cat19 + cat20 + cat21 + cat22 + cat23 + cat24 + 
                 cat25 + cat26 + cat27 + cat28 + cat29 + cat30 + cat31 + cat32 + 
                 cat33 + cat34 + cat35 + cat36 + cat37 + cat38 + cat39 + cat40 + 
                 cat41 + cat42 + cat43 + cat44 + cat45 + cat46 + cat47 + cat48 + 
                 cat49 + cat50 + cat51 + cat52 + cat53 + cat54 + cat55 + cat56 + 
                 cat57 + cat58 + cat59 + cat60 + cat61 + cat62 + cat63 + cat64 + 
                 cat65 + cat66 + cat67 + cat68 + cat69 + cat70 + cat71 + cat72 + 
                 cat73 + cat74 + cat75 + cat76 + cat77 + cat78 + cat79 + cat80 + 
                 cat81 + cat82 + cat83 + cat84 + cat85 + cat86 + cat87 + cat88 + 
                 cat89 + cat90 + cat91 + cat92 + cat93 + cat94 + cat95 + cat96 + 
                 cat97 + cat98 + cat99 + cat100 + cat101 + cat102 + cat103 + 
                 cat104 + cat105 + cat106 + cat107 + cat108 + cat109 + cat110 + 
                 cat111 + cat112 + cat113 + cat114 + cat115 + cat116 + cont1 + 
                 cont2 + cont3 + cont4 + cont5 + cont6 + cont7 + cont8 + cont9 + 
                 cont10 + cont11 + cont12 + cont13 + cont14
               , task = "r"
               , dim = 100
               #,init_stdev = 0.001
               , verbosity = 1 
               , iter = 100
               , method = "mcmc"
               , validation = X_valid
               , exe_loc = "C:\\Users\\padepu\\Documents\\R\\11RedHat\\LibFM"
               )
pred_cv_bags <- pred_cv_bags + predFM
}
pred_cv_bags <- pred_cv_bags / bags
score(pred_cv_bags,X_val$loss,metric)

head(pred_cv_bags)
head(X_val$loss)

# devtools::install_github("andland/libFMexe", force = TRUE)

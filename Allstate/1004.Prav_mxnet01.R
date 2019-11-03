# require(mlbench)
## Loading required package: mlbench
require(mxnet)


################################################################################

trainingSet <- read_csv('./input/train_deeplearningOhe.csv')
testingSet  <- read_csv('./input/test_deeplearningOhe.csv')


################################################################################



feature.names <- names(trainingSet[,-which(names(trainingSet) %in% c( "id","loss","CVindices"
                                                                    ))])


Ratio.names <- grep("Ratio", names(trainingSet), value = T)

feature.names <- setdiff(feature.names, Ratio.names)

trainingSet$loss <- log(trainingSet$loss)


###################################################################################################

demo.metric.mae <- mx.metric.custom("mae", function(label, pred) {
                      res <- mean(abs(exp(label)-exp(pred)))
                      return(res)
                      })



# defining the network
data = mx.symbol.Variable('data')
#flat = mx.symbol.Flatten(data=data)
fc1 = mx.symbol.FullyConnected(data=data, num_hidden=800)
act1 = mx.symbol.Activation(data=fc1, act_type="relu")
act11 = mx.symbol.Dropout(data= act1, p = 0.2)
fc2 = mx.symbol.FullyConnected(data=act11, num_hidden=400)
act2 = mx.symbol.Activation(data=fc2, act_type="relu")
act22 = mx.symbol.Dropout(data=act2,p = 0.1)
fc3 = mx.symbol.FullyConnected(data=act22, num_hidden=1)
# act3 = mx.symbol.Activation(data=fc3, act_type="relu")
# fc4 = mx.symbol.FullyConnected(data=act3, num_hidden=12)
net = mx.symbol.MAERegressionOutput(data=fc3)


# # Define the input data
# data <- mx.symbol.Variable("data")
# # A fully connected hidden layer
# # data: input source
# # num_hidden: number of neurons in this layer
# fc1 <- mx.symbol.FullyConnected(data, num_hidden=1)
# # Use linear regression for the output layer
# lro <- mx.symbol.MAERegressionOutput(fc1) #LinearRegressionOutput(fc1)

# train
device = mx.gpu();
cv          = 5
bags        = 1
n.rounds    = 100
seed        = 2016


######################################################################################

X_trainingSet       <- data.matrix(trainingSet[,feature.names])
X_trainingSetlabel  <- trainingSet$loss
X_testingSet        <- data.matrix(testingSet[,feature.names])


mx.set.seed(seed)
for (i in 5:cv)
  
{
  cat(i ,"fold Processing\n")
  
  cat("X_build fold Processing\n")
  X_build <- subset(trainingSet, CVindices != i, select = -c( CVindices))
  cat("X_val fold Processing\n")
  X_val   <- subset(trainingSet, CVindices == i, select = -c( CVindices)) 
  
  X_train <- data.matrix(X_build[,feature.names])
  X_label <- X_build$loss
  
  X_valid    <- data.matrix(X_val[,feature.names])
  X_vallabel <- X_val$loss
  
  
  pred_cv_bags   <- rep(0, nrow(X_val[, feature.names]))
  pred_test_bags <- rep(0, nrow(testingSet[,feature.names]))
  
  for (b in 1:bags) 
  {
    cat(b ," - bag Processing\n")
    
    model <- mx.model.FeedForward.create(
      X                  = X_train,
      y                  = X_label,
      eval.data          = list("data"= X_valid,"label"=X_vallabel),
      ctx                = device,    # device is either the cpu or gpu (graphical processor unit)
      symbol             = net,       # this is the network structure
      eval.metric        = demo.metric.mae,
      #val.metric        = mx.metric.accuracy,
      num.round          = n.rounds,       # how many batches to work with
      learning.rate      = 1e-2,      # 0.01 is a good start
      momentum           = 0.9,       # using second derivative
      wd                 = 0.0001,    # what is this for?
      initializer        = mx.init.normal(1/sqrt(nrow(X_build))),   # the standard devation is scaled with the number of
      #observations to prevent slow learning if by chance all weights are large or small
      #initializer        = mx.init.uniform(0.1),   # the standard devation is scaled with the number of 
      array.batch.size   = 500,
      #epoch.end.callback = mx.callback.save.checkpoint("titanic"),
      #batch.end.callback = mx.callback.log.train.metric(100),
      array.layout="rowmajor"
    );
    
    
    
    cat("X_val prediction Processing\n")
    pred_cv    <- predict(model, X_valid)
    cat("CV TestingSet prediction Processing\n")
    pred_test  <- predict(model, X_testingSet)
    
    pred_cv_bags   <- pred_cv_bags + exp(t(pred_cv))
    pred_test_bags <- pred_test_bags + exp(t(pred_test))
  }
  pred_cv_bags   <- pred_cv_bags / bags
  pred_test_bags <- pred_test_bags / bags
  
  cat("CV Fold-", i, " ", metric, ": ", score(exp(X_val$loss), pred_cv_bags, metric), "\n", sep = "")
  
  #rm(X_train, X_valid); gc()

  val_predictions  <- data.frame(id=X_val$id, loss = pred_cv_bags)
  test_predictions <- data.frame(id=testingSet$id, loss = pred_test_bags)
  
  # if(i == 1)
  # {
  #   write.csv(val_predictions,  './submissions/prav.mxnet01.fold1.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.mxnet01.fold1-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 2)
  # {
  #   write.csv(val_predictions,  './submissions/prav.mxnet01.fold2.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.mxnet01.fold2-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 3)
  # {
  #   write.csv(val_predictions,  './submissions/prav.mxnet01.fold3.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.mxnet01.fold3-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 4)
  # {
  #   write.csv(val_predictions,  './submissions/prav.mxnet01.fold4.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.mxnet01.fold4-test.csv', row.names=FALSE, quote = FALSE)
  # }
  # if(i == 5)
  # {
  #   write.csv(val_predictions,  './submissions/prav.mxnet01.fold5.csv', row.names=FALSE, quote = FALSE)
  #   write.csv(test_predictions, './submissions/prav.mxnet01.fold5-test.csv', row.names=FALSE, quote = FALSE)
  # }
  
}

# Full training


# #########################################################################################################
# Bagging - Full train
# #########################################################################################################



fulltest_ensemble <- rep(0, nrow(testingSet[,feature.names]))


for (b in 1:bags) {
  cat(b ," - bag Processing\n")
  cat("Bagging Full TrainingSet training\n")
  Fullmodel <- mx.model.FeedForward.create(
              X                  = X_trainingSet,
              y                  = X_trainingSetlabel,
              eval.data          = list("data"= X_trainingSet,"label"=X_trainingSetlabel),
              ctx                = device,    # device is either the cpu or gpu (graphical processor unit)
              symbol             = net,       # this is the network structure
              eval.metric        = demo.metric.mae,
              #val.metric        = mx.metric.accuracy,
              num.round          = n.rounds,       # how many batches to work with
              learning.rate      = 1e-2,      # 0.01 is a good start
              momentum           = 0.9,       # using second derivative
              wd                 = 0.0001,    # what is this for?
              initializer        = mx.init.normal(1/sqrt(nrow(X_build))),   # the standard devation is scaled with the number of
              #observations to prevent slow learning if by chance all weights are large or small
              #initializer        = mx.init.uniform(0.1),   # the standard devation is scaled with the number of 
              array.batch.size   = 500,
              #epoch.end.callback = mx.callback.save.checkpoint("titanic"),
              #batch.end.callback = mx.callback.log.train.metric(100),
              array.layout="rowmajor"
            );
  
  cat("Bagging Full Model prediction Processing\n")
  
  cat("CV TestingSet prediction Processing\n")
  predfull_test  <- predict(Fullmodel, X_testingSet)

  fulltest_ensemble <- fulltest_ensemble + exp(t(predfull_test))
  
}

fulltest_ensemble <- fulltest_ensemble / bags
testfull_predictions  <- data.frame(id=testingSet$id, loss = fulltest_ensemble)
write.csv(testfull_predictions, './submissions/prav.mxnet01.full.csv', row.names=FALSE, quote = FALSE)




# #########################################################################################################
# Bagging - Full train
# #########################################################################################################


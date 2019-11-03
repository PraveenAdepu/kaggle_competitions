# R version of some Machine Learning Method starter code using H2O. 
# The parameters are not tuned. 


### https://www.kaggle.com/nigelcarpenter/allstate-claims-severity/farons-xgb-starter-ported-to-r contains the 
### Code for xgboost in R. 

#Reading Data, old school read.csv. Using fread is faster. 
set.seed(0)
train<-read.csv('./input/train.csv')
test<-read.csv('./input/test.csv')

train<-train[,-1]
test_label<-test[,1]
test<-test[,-1]

index<-sample(1:(dim(train)[1]), 0.2*dim(train)[1], replace=FALSE)

train_frame<-train[-index,]
valid_frame<-train[index,]


valid_predict<-valid_frame[,-ncol(valid_frame)]
valid_loss<-valid_frame[,ncol(valid_frame)]

train_frame[,ncol(train_frame)]<-log(train_frame[,ncol(train_frame)])
valid_frame[,ncol(train_frame)]<-log(valid_frame[,ncol(valid_frame)])

library(h2o)
kd_h2o<-h2o.init(nthreads = 16, max_mem_size = "16g")

train_frame.hex<-as.h2o(train_frame)
valid_frame.hex<-as.h2o(valid_frame)
valid_predict.hex<-as.h2o(valid_predict)
test.hex<-as.h2o(test)

##Good old Random Forest
start<-proc.time()
model_rf<-h2o.randomForest(x=1:(ncol(train_frame.hex)-1), y=ncol(train_frame.hex), 
                           training_frame=train_frame.hex, validation_frame=valid_frame.hex, 
                           ntrees=5)
print(proc.time()-start)
pred_rf<-(as.matrix(predict(model_rf, valid_predict.hex)))
score_rf=mean(abs(exp(pred_rf)-valid_loss))

##Good Old gbm
start<-proc.time()
model_gbm<-h2o.gbm(x=1:(ncol(train_frame.hex)-1), y=ncol(train_frame.hex), 
                   training_frame=train_frame.hex, validation_frame=valid_frame.hex, 
                   ntrees=600, learn_rate = 0.05, max_depth=4)
print(proc.time()-start)
pred_gbm<-as.matrix(predict(model_gbm, valid_predict.hex))
score_gbm=mean(abs(exp(pred_gbm)-valid_loss))


#start<-proc.time()
#model_gbm_pois<-h2o.gbm(x=1:(ncol(train_frame.hex)-1), y=ncol(train_frame.hex), 
#	     training_frame=train_frame.hex, validation_frame=valid_frame.hex, 
#		ntrees=600, learn_rate = 0.02, max_depth=5, distribution="poisson")
#print(proc.time()-start)
#pred_gbm_pois<-as.matrix(predict(model_gbm_pois, valid_predict.hex))
#score_gbm_pois=mean(abs(exp(pred_gbm_pois)-valid_loss))


start<-proc.time()
model_learning<-h2o.deeplearning(x=1:(ncol(train_frame.hex)-1), y=ncol(train_frame.hex), 
                                 training_frame=train_frame.hex, validation_frame=valid_frame.hex,
                                 epochs=20, hidden=c(100, 100))
print(proc.time()-start)
pred_learning<-as.matrix(predict(model_learning, valid_predict.hex))
score_learning=mean(abs(exp(pred_learning)-valid_loss))


pred_ensemble=(pred_gbm+pred_learning)/2
score_ensemble=mean(abs(exp(pred_ensemble)-valid_loss))

pred_gbm_all<-(as.matrix(predict(model_gbm, test.hex)))
#pred_gbm_pois_all<-(as.matrix(predict(model_gbm_pois, test.hex)))
pred_learning_all<-(as.matrix(predict(model_learning, test.hex)))
pred_all<-exp((pred_gbm_all+pred_learning_all)/2)


submission = read.csv('../input/sample_submission.csv', colClasses = c("integer", "numeric"))
submission$loss = pred_all

write.csv(submission, 'h2o_blend.csv', row.names=FALSE)

h2o.shutdown(prompt=FALSE)

# Summary of Performance 
# Random Forest 
print(score_rf)
# GBM 
print(score_gbm)
# Deep Learning
print(score_learning)

# Ensemble 
print(score_ensemble)


# Score in Public Leaderboard is 1132.37
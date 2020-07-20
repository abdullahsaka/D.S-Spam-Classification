#importing packages
library(e1071)
library(SparseM)

# Read train file
require(Matrix)
dataLines_1 <- readLines("~/Rprojects/spam_hw2/articles.train")
m <- length(dataLines_1)
dataTokens_train = strsplit(dataLines_1, "[: ]")

Y_train = sapply(dataTokens_train, function(example) {as.numeric(example[1])})
Y1 <- ifelse(Y_train==1,1,-1)
Y2 <- ifelse(Y_train==2,1,-1)
Y3 <- ifelse(Y_train==3,1,-1)
Y4 <- ifelse(Y_train==4,1,-1)

X_list_train = lapply(dataTokens_train, function(example) {n = length(example) - 1; matrix(as.numeric(example[2:(n+1)]), ncol=2, byrow=T)})
X_list_train = mapply(cbind, x=1:length(X_list_train), y=X_list_train)
X_traindata = do.call('rbind', X_list_train)
train_X = sparseMatrix(x=X_traindata[,3], i=X_traindata[,1], j=X_traindata[,2])

#Create SVM Model
svm.model1 <- svm(train_X,Y1,type='C-classification')
predict1 <- predict(svm.model1,train_X)
cf1 <- table(predict1,Y1)
cf1 #accuracy is 75.47%

svm.model2 <- svm(train_X,Y2,type='C-classification')
predict2 <- predict(svm.model2,train_X)
cf2 <- table(predict2,Y2)
cf2 #accuracy is 75.05%

svm.model3 <- svm(train_X,Y3,type='C-classification')
predict3 <- predict(svm.model3,train_X)
cf3 <- table(predict3,Y3)
cf3 #accuracy is 76.2%

svm.model4 <- svm(train_X,Y4,type='C-classification')
predict4 <- predict(svm.model4,train_X)
cf4 <- table(predict4,Y4)
cf4 #accuracy is 80.77%

# Read test file
require(Matrix)
dataLines_2 <- readLines("/Users/sakahome/Rprojects/spam_hw2/articles.test")
m <- length(dataLines_2)
dataTokens_test = strsplit(dataLines_2, "[: ]")

Y_test = sapply(dataTokens_test, function(example) {as.numeric(example[1])})
Y_test1 <- ifelse(Y_test==1,1,-1)
Y_test2 <- ifelse(Y_test==2,1,-1)
Y_test3 <- ifelse(Y_test==3,1,-1)
Y_test4 <- ifelse(Y_test==4,1,-1)

X_list_test = lapply(dataTokens_test, function(example) {n = length(example) - 1; matrix(as.numeric(example[2:(n+1)]), ncol=2, byrow=T)})
X_list_test = mapply(cbind, x=1:length(X_list_test), y=X_list_test)
X_testdata = do.call('rbind', X_list_test)
test_X = sparseMatrix(x=X_testdata[,3], i=X_testdata[,1], j=X_testdata[,2])

#SVM Model in validation set
predict_test1 <- predict(svm.model1,test_X[,0:51949])
cf1 <- table(predict_test1,Y_test1)
(cf1[1,1]+cf1[2,2])/sum(cf1)

predict_test2 <- predict(svm.model2,test_X[,0:51949])
cf2 <- table(predict_test2,Y_test2)
(cf2[1,1]+cf2[2,2])/sum(cf2)

predict_test3 <- predict(svm.model3,test_X[,0:51949])
cf3 <- table(predict_test3,Y_test3)
(cf3[1,1]+cf3[2,2])/sum(cf3)

predict_test4 <- predict(svm.model4,test_X[,0:51949])
cf4 <- table(predict_test4,Y_test4)
(cf4[1,1]+cf4[2,2])/sum(cf4)

#Splitting data into training/testing sets using random sampling(Training: 75%, Testing: 25%)
nr<-nrow(train_X)
set.seed(7)
trnIndex = sample(1:nr, size = round(0.75*nr), replace=FALSE)
X_trn <- train_X[trnIndex, ] 
X_tst <- train_X[-trnIndex, ]

Y_train <- as.data.frame(Y_train)
Y_trn <- Y_train[trnIndex, ] 
Y_tst <- Y_train[-trnIndex, ]

Ytrn_1 <- ifelse(Y_trn==1,1,-1)
Ytrn_2 <- ifelse(Y_trn==2,1,-1)
Ytrn_3 <- ifelse(Y_trn==3,1,-1)
Ytrn_4 <- ifelse(Y_trn==4,1,-1)

Ytst_1 <- ifelse(Y_tst==1,1,-1)
Ytst_2 <- ifelse(Y_tst==2,1,-1)
Ytst_3 <- ifelse(Y_tst==3,1,-1)
Ytst_4 <- ifelse(Y_tst==4,1,-1)

# Run soft margin classifier model

C=c(0.125,0.25,0.50,1,2,4,8,16,32,64,128,256,512)
library(cat)
for (i in 1:length(C)) {
  cat("Cost is : ",C[i])
  cat("\n")
  svm.model <- svm(X_trn,Ytrn_1,type='C-classification',cost= C[i])
  print("The training confusion matrix:")
  predict_trn <- predict(svm.model,X_trn)
  print(table(predict_trn,Ytrn_1))
  cf <- table(predict_trn,Ytrn_1)
  cat("The training accuracy is ",((cf[1,1]+cf[2,2])/sum(cf)))
  cat("\n")
  print("The testing confusion matrix:")
  predict_tst <- predict(svm.model,X_tst)
  print(table(predict_tst,Ytst_1))
  cf1 <- table(predict_tst,Ytst_1)
  cat("The testing accuracy is ",((cf1[1,1]+cf1[2,2])/sum(cf1)))
  cat("\n")
}

# apply for classifier 2
for (i in 1:length(C)) {
  cat("Cost is : ",C[i])
  cat("\n")
  svm.model <- svm(X_trn,Ytrn_2,type='C-classification',cost= C[i])
  print("The training confusion matrix:")
  predict_trn <- predict(svm.model,X_trn)
  print(table(predict_trn,Ytrn_2))
  cf <- table(predict_trn,Ytrn_2)
  cat("The training accuracy is ",((cf[1,1]+cf[2,2])/sum(cf)))
  cat("\n")
  print("The testing confusion matrix:")
  predict_tst <- predict(svm.model,X_tst)
  print(table(predict_tst,Ytst_2))
  cf1 <- table(predict_tst,Ytst_2)
  cat("The testing accuracy is ",((cf1[1,1]+cf1[2,2])/sum(cf1)))
  cat("\n")
}

# apply for classifier 3
for (i in 1:length(C)) {
  cat("Cost is : ",C[i])
  cat("\n")
  svm.model <- svm(X_trn,Ytrn_3,type='C-classification',cost= C[i])
  print("The training confusion matrix:")
  predict_trn <- predict(svm.model,X_trn)
  print(table(predict_trn,Ytrn_3))
  cf <- table(predict_trn,Ytrn_3)
  cat("The training accuracy is ",((cf[1,1]+cf[2,2])/sum(cf)))
  cat("\n")
  print("The testing confusion matrix:")
  predict_tst <- predict(svm.model,X_tst)
  print(table(predict_tst,Ytst_3))
  cf1 <- table(predict_tst,Ytst_3)
  cat("The testing accuracy is ",((cf1[1,1]+cf1[2,2])/sum(cf1)))
  cat("\n")
}

# apply for classifier 4
for (i in 1:length(C)) {
  cat("Cost is : ",C[i])
  cat("\n")
  svm.model <- svm(X_trn,Ytrn_4,type='C-classification',cost= C[i])
  print("The training confusion matrix:")
  predict_trn <- predict(svm.model,X_trn)
  print(table(predict_trn,Ytrn_4))
  cf <- table(predict_trn,Ytrn_4)
  cat("The training accuracy is ",((cf[1,1]+cf[2,2])/sum(cf)))
  cat("\n")
  print("The testing confusion matrix:")
  predict_tst <- predict(svm.model,X_tst)
  print(table(predict_tst,Ytst_4))
  cf1 <- table(predict_tst,Ytst_4)
  cat("The testing accuracy is ",((cf1[1,1]+cf1[2,2])/sum(cf1)))
  cat("\n")
}


#partd(use 512 as cost parameter)
svm.model1 <- svm(train_X,Y1,type='C-classification',cost=512)
predict1 <- predict(svm.model1,train_X)
cf <- table(predict1,Y1)
(cf[1,1]+cf[2,2])/sum(cf)

predict_test1 <- predict(svm.model1,test_X[,0:51949])
cf1 <- table(predict_test1,Y_test1)
(cf1[1,1]+cf1[2,2])/sum(cf1)

#classifier 2
svm.model2 <- svm(train_X,Y2,type='C-classification',cost=512)
predict2 <- predict(svm.model2,train_X)
cf2 <- table(predict2,Y2)
(cf2[1,1]+cf2[2,2])/sum(cf2)

predict_test2 <- predict(svm.model2,test_X[,0:51949])
cf2 <- table(predict_test2,Y_test2)
(cf2[1,1]+cf2[2,2])/sum(cf2)

#classifier 3
svm.model3 <- svm(train_X,Y3,type='C-classification',cost=512)
predict3 <- predict(svm.model3,train_X)
cf3 <- table(predict3,Y3)
(cf3[1,1]+cf3[2,2])/sum(cf3)

predict_test3 <- predict(svm.model3,test_X[,0:51949])
cf3 <- table(predict_test3,Y_test3)
(cf3[1,1]+cf3[2,2])/sum(cf3)

#classifier 4
svm.model4 <- svm(train_X,Y4,type='C-classification',cost=512)
predict4 <- predict(svm.model4,train_X)
cf4 <- table(predict4,Y4)
(cf4[1,1]+cf4[2,2])/sum(cf4)

predict_test4 <- predict(svm.model4,test_X[,0:51949])
cf4 <- table(predict_test4,Y_test4)
(cf4[1,1]+cf4[2,2])/sum(cf4)

#part e
normtrain_X <- normalize.rows(train_X,method='euclidean')

svm.model1 <- svm(normtrain_X,Y1,type='C-classification',cost=512)
predict1 <- predict(svm.model1,normtrain_X)
cf1 <- table(predict1,Y1)
cf1
(cf1[1,1]+cf1[2,2])/sum(cf1)

svm.model2 <- svm(normtrain_X,Y2,type='C-classification',cost=512)
predict2 <- predict(svm.model2,normtrain_X)
cf2 <- table(predict2,Y2)
cf2
(cf2[1,1]+cf2[2,2])/sum(cf2)

svm.model3 <- svm(normtrain_X,Y3,type='C-classification',cost=512)
predict3 <- predict(svm.model3,normtrain_X)
cf3 <- table(predict3,Y3)
(cf3[1,1]+cf3[2,2])/sum(cf3)
cf3

svm.model4 <- svm(normtrain_X,Y4,type='C-classification',cost=512)
predict4 <- predict(svm.model4,normtrain_X)
cf4 <- table(predict4,Y4)
cf4
(cf4[1,1]+cf4[2,2])/sum(cf4)

normtest_X <- normalize.rows(test_X,method='euclidean',p=2)

predict1 <- predict(svm.model1,normtest_X[,0:51949])
cf1 <- table(predict1,Y_test1)
(cf1[1,1]+cf1[2,2])/sum(cf1)

predict2 <- predict(svm.model2,normtest_X[,0:51949])
cf2 <- table(predict2,Y_test2)
(cf2[1,1]+cf2[2,2])/sum(cf2)

predict3 <- predict(svm.model3,normtest_X[,0:51949])
cf3 <- table(predict3,Y_test3)
(cf3[1,1]+cf3[2,2])/sum(cf3)

#(part e) data preparation
Y_test = sapply(dataTokens_test, function(example) {as.numeric(example[1])})
Y_test_combo1 <- Y_test[1:1200]
Y_test_combo2 <- rbind(Y_test[1:600],Y_test[1201:1800])
Y_test_combo3 <- rbind(Y_test[1:600],Y_test[1801:2400])
Y_test_combo4 <- Y_test[601:1800]
Y_test_combo5 <- rbind(Y_test[601:1200],Y_test[1801:2400])
Y_test_combo6 <- Y_test[1200:2400]

X_test_combo1 <- test_X[1:1200,]
X_test_combo2 <- rbind(test_X[1:600,],test_X[1201:1800,])
X_test_combo3 <- rbind(test_X[1:600,],test_X[1801:2400,])
X_test_combo4 <- test_X[601:1800,]
X_test_combo5 <- rbind(test_X[601:1200,],test_X[1801:2400,])
X_test_combo6 <- test_X[1200:2400,]

Y_train_combo1 <- Y_train[1:2000,]
Y_train_combo2 <- rbind(Y_train[1:1000,],Y_train[2001:3000,])
Y_train_combo3 <- rbind(Y_train[1:1000,],Y_train[3001:4000,])
Y_train_combo4 <- Y_train[1001:3000,]
Y_train_combo5 <- rbind(Y_train[1001:2000,],Y_train[3001:4000,])
Y_train_combo6 <- Y_train[2001:4000,]

X_train_combo1 <- train_X[1:2000,]
X_train_combo2 <- rbind(train_X[1:1000,],train_X[2001:3000,])
X_train_combo3 <- rbind(train_X[1:1000,],train_X[3001:4000,])
X_train_combo4 <- train_X[1001:3000,]
X_train_combo5 <- rbind(train_X[1001:2000,],train_X[3001:4000,])
X_train_combo6 <- train_X[2001:4000,]

#(part e) data modeling
opt1 <- svm(X_train_combo1,Y_train_combo1,type='C-classification',cost=512)
predictopt1 <- predict(opt1,X_train_combo1)
predictopt1b <- predict(opt1, X_test_combo1[,0:51949])
cf <- table(predictopt1, Y_train_combo1)
cf
(cf[1,1]+cf[2,2])/sum(cf)
cf <- table(predictopt1b, Y_test_combo1)
(cf[1,1]+cf[2,2])/sum(cf)

opt2 <- svm(X_train_combo2,Y_train_combo2,type='C-classification',cost=512)
predictopt2 <- predict(opt2,X_train_combo2)
predictopt2b <- predict(opt2, X_test_combo2[,0:51949])
cf <- table(predictopt2, Y_train_combo2)
cf
(cf[1,1]+cf[2,2])/sum(cf)
cf <- table(predictopt2b, Y_test_combo2)
(cf[1,1]+cf[2,2])/sum(cf)

opt3 <- svm(X_train_combo3,Y_train_combo3,type='C-classification',cost=512)
predictopt3 <- predict(opt3,X_train_combo3)
predictopt3b <- predict(opt3, X_test_combo3[,0:51949])
cf <- table(predictopt3, Y_train_combo3)
cf
(cf[1,1]+cf[2,2])/sum(cf)
cf <- table(predictopt3b, Y_test_combo3)
(cf[1,1]+cf[2,2])/sum(cf)

opt4 <- svm(X_train_combo4,Y_train_combo4,type='C-classification',cost=512)
predictopt4 <- predict(opt4,X_train_combo4)
predictopt4b <- predict(opt4, X_test_combo4[,0:51949])
cf <- table(predictopt4, Y_train_combo4)
cf
(cf[1,1]+cf[2,2])/sum(cf)
cf <- table(predictopt4b, Y_test_combo4)
(cf[1,1]+cf[2,2])/sum(cf)

opt5 <- svm(X_train_combo5,Y_train_combo5,type='C-classification',cost=512)
predictopt5 <- predict(opt5,X_train_combo5)
predictopt5b <- predict(opt5, X_test_combo5[,0:51949])
cf <- table(predictopt5, Y_train_combo5)
cf
(cf[1,1]+cf[2,2])/sum(cf)
cf <- table(predictopt5b, Y_test_combo5)
(cf[1,1]+cf[2,2])/sum(cf)

opt6 <- svm(X_train_combo6,Y_train_combo6,type='C-classification',cost=512)
predictopt6 <- predict(opt6,X_train_combo6)
predictopt6b <- predict(opt6, X_test_combo6[,0:51949])
cf <- table(predictopt6, Y_train_combo6)
cf
(cf[1,1]+cf[2,2])/sum(cf)
cf <- table(predictopt6b, Y_test_combo6)
(cf[1,1]+cf[2,2])/sum(cf)

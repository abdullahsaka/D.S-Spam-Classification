# Importing required packages
library(e1071)
library(SparseM)
library(naivebayes)

# Read train file
require(Matrix)
dataLines_1 <- readLines("~/Rprojects/spam_hw2&3/articles.train")
m <- length(dataLines_1)
dataTokens_train = strsplit(dataLines_1, "[: ]")
Y_train = sapply(dataTokens_train, function(example) {as.numeric(example[1])})
X_list_train = lapply(dataTokens_train, function(example) {n = length(example) - 1; matrix(as.numeric(example[2:(n+1)]), ncol=2, byrow=T)})
X_list_train = mapply(cbind, x=1:length(X_list_train), y=X_list_train)
X_traindata = do.call('rbind', X_list_train)
X_traindata[,3]=1
train_X = sparseMatrix(x=X_traindata[,3], i=X_traindata[,1], j=X_traindata[,2])

# Read test file
require(Matrix)
dataLines_2 <- readLines("/Users/sakahome/Rprojects/spam_hw2&3/articles.test")
m <- length(dataLines_2)
dataTokens_test = strsplit(dataLines_2, "[: ]")
Y_test = sapply(dataTokens_test, function(example) {as.numeric(example[1])})
X_list_test = lapply(dataTokens_test, function(example) {n = length(example) - 1; matrix(as.numeric(example[2:(n+1)]), ncol=2, byrow=T)})
X_list_test = mapply(cbind, x=1:length(X_list_test), y=X_list_test)
X_testdata = do.call('rbind', X_list_test)
X_testdata[,3]=1
test_X = sparseMatrix(x=X_testdata[,3], i=X_testdata[,1], j=X_testdata[,2])

Y_train <- as.factor(Y_train)
colnames(train_X) <- paste0("V", seq_len(ncol(train_X)))

# Question 3 - part b: Apply Bernoulli Naive Bayes Model without laplace method
bnb <- bernoulli_naive_bayes(train_X, Y_train, laplace =0)
summary(bnb)

pred <- predict(bnb, newdata = train_X, type = "prob")
x1 <- table(pred,Y_train)
accuracy_wtht_laplace <- (x1[1,1]+x1[2,2]+x1[3,3]+x1[4,4])/sum(x1)
x1
accuracy_wtht_laplace

# Question 3 - part c: Apply Bernoulli Naive Bayes Model with laplace smoothing
bnb_laplace <- bernoulli_naive_bayes(train_X, Y_train, laplace =1)
summary(bnb_laplace)

pred <- predict(bnb_laplace, newdata = train_X, type = "class")
z1 <- table(pred,Y_train)
accuracy_wth_laplace <- (z1[1,1]+z1[2,2]+z1[3,3]+z1[4,4])/sum(z1)
z1
accuracy_wth_laplace

# Question 3 - part d: Apply Multinomial Naive Bayes Model
m <- length(dataLines_1)
dataTokens_train = strsplit(dataLines_1, "[: ]")
Y_train = sapply(dataTokens_train, function(example) {as.numeric(example[1])})
X_list_train = lapply(dataTokens_train, function(example) {n = length(example) - 1; matrix(as.numeric(example[2:(n+1)]), ncol=2, byrow=T)})
X_list_train = mapply(cbind, x=1:length(X_list_train), y=X_list_train)
X_traindata = do.call('rbind', X_list_train)
train_X = sparseMatrix(x=X_traindata[,3], i=X_traindata[,1], j=X_traindata[,2])
Y_train <- as.factor(Y_train)
colnames(train_X) <- paste0("V", seq_len(ncol(train_X)))

mltm <- multinomial_naive_bayes(train_X, Y_train, laplace =1)
summary(mltm)

pred <- predict(mltm, newdata = train_X, type = "class")
t1 <- table(pred,Y_train)
accuracy_mltm <- (t1[1,1]+t1[2,2]+t1[3,3]+t1[4,4])/sum(t1)
t1
accuracy_mltm

pred <- predict(mltm, newdata = test_X, type = "class")
t <- table(pred,Y_test)
accuracy <- (t[1,1]+t[2,2]+t[3,3]+t[4,4])/sum(t)
t
accuracy





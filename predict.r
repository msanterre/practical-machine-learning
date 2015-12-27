library(doMC)
library(caret)

# Use all the cores on this baby
registerDoMC(cores=8)

# We're trying to find classe
training <- read.csv("pml-training.csv", header=TRUE)
testing <- read.csv("pml-testing.csv", header=TRUE)

# Remove aggregate rows
training <- training[training[,"new_window"] == "no",]

# Remove the useless columns
training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]

set.seed(333)

# Remove columns with many NAs
nas <- sapply(2L:(NCOL(training)-1), function(i) {sum(is.na(training[,i]))})
naCols <- colnames(training)[which(nas != 0) + 1]
training <- training[ , -which(names(training) %in% naCols)]

# Remove the useless variables
nearZeroVarianceColumns <- nearZeroVar(training)
training <- training[,-nearZeroVarianceColumns]

inTrain <- createDataPartition(training$classe, p=0.7, list=FALSE)

training.train <- training[inTrain,]
training.test <- training[-inTrain,]

fitModel <- train(classe ~ ., data=training.train, method="rf", ntrees=750)

# Test out the predictions
pred <- predict(fitModel, training.test)
print(confusionMatrix(pred, training.test$classe))

# Predict the provided test cases

cols <- intersect(names(testing), names(training)) # Use the same columns as the training set
testing <- testing[,cols]

pmlWriteFiles = function(x){ # Utility method from the instructions
  n = length(x)
  for(i in 1:n){
    filename = paste0("problems/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

predictProblems <- predict(fitModel, testing)
pmlWriteFiles(predictProblems)

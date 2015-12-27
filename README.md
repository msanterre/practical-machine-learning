## Predicting barbell lift methods

### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

### Problem

With data collected from fitness trackers, we want to predict the barbell curl method that the user was doing while the sample was collected.


### Analysis

Glancing over the data, we can make a few assumptions:

- The first 7 columns (username, timestamps, windows, etc) are device metadata and don't seem useful for predicting the movement
- There seem to be rows that have the aggregate information about a specific window *(The usage of window is unspecified)*
- The raw data rows contain a lot of columns (used by aggregate rows) that contain no data
- There are 160 columns (features) in the training data


### Features

We can easily remove a lot of the features by removing the columns that the aggregate rows used. They are mostly all `NA` in the other rows, so we'll use this method to strip them out.

```
nas <- sapply(2L:(NCOL(training)-1), function(i) {sum(is.na(training[,i]))})
naCols <- colnames(training)[which(nas != 0) + 1]
training <- training[ , -which(names(training) %in% naCols)]
```

We should also remove the columns that have very low variance as they won't be very useful in training the model.

```
nearZeroVarianceColumns <- nearZeroVar(training)
training <- training[,-nearZeroVarianceColumns]
```

This leaves us with 53 features (including classe).


### Method

Since this is a classification problem, we need a classification model. I chose to use a random forest prediction model since they're simple to use and do quite well.
I split the training data this way: **70% in training set, 30% in the testing set**

I trained the model with this method:
```
train(classe ~ ., data=training.train, method="rf", ntrees=750)
```

When gauging the optimal number of trees, 750 seemed to offer a good tradeoff between speed and accuracy.


### Results

When running this model on the testing set data, **98.87% accuracy** was reported.

The full output of the confusion matrix:

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1633   10    0    0    0
         B    4 1099    9    0    0
         C    0    6  992   22    0
         D    0    0    4  921    5
         E    4    0    0    1 1053

Overall Statistics

               Accuracy : 0.9887
                 95% CI : (0.9856, 0.9913)
    No Information Rate : 0.2847
    P-Value [Acc > NIR] : < 2.2e-16

                  Kappa : 0.9857
 Mcnemar's Test P-Value : NA

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9951   0.9857   0.9871   0.9756   0.9953
Specificity            0.9976   0.9972   0.9941   0.9981   0.9989
Pos Pred Value         0.9939   0.9883   0.9725   0.9903   0.9953
Neg Pred Value         0.9981   0.9966   0.9973   0.9952   0.9989
Prevalence             0.2847   0.1935   0.1744   0.1638   0.1836
Detection Rate         0.2834   0.1907   0.1721   0.1598   0.1827
Detection Prevalence   0.2851   0.1930   0.1770   0.1614   0.1836
Balanced Accuracy      0.9963   0.9914   0.9906   0.9869   0.9971
```


### References

- https://class.coursera.org/predmachlearn-035/human_grading/view/courses/975205/assessments/4/submissions
- https://www.kaggle.com/wiki/RandomForests

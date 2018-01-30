# Iris Classification in R

Basic boilerplate code example for supervised learning classification tasks. Dataset obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Iris)

The following code uses  the [R caret package](https://cran.r-project.org/web/packages/caret/index.html)

```R
library("caret")
```

### Split data into train/test sets using createDataPartition()
```R
data_split <- createDataPartition(data$Species, p = 0.8, list = FALSE)

test <- data[-data_split,] # Save 20% of data for test validation here
dataset <- data[data_split,] # 80% of data 
```


## Data Summary
### Dataset Dimensions
```R
dim(dataset)
#[1] 120   5
```
Number of rows and columns 


### List datatypes for attributes
```R
sapply(dataset, class)
#Sepal.Length  Sepal.Width Petal.Length  Petal.Width      Species 
#   "numeric"    "numeric"    "numeric"    "numeric"     "factor" 
```
`sapply()` used to map function to each attribute; class function returns data class type for a given attribute


### Data Frame Header
```R
head(dataset)
#  Sepal.Length Sepal.Width Petal.Length Petal.Width     Species
#1          5.1         3.5          1.4         0.2 Iris-setosa
#2          4.9         3.0          1.4         0.2 Iris-setosa
#3          4.7         3.2          1.3         0.2 Iris-setosa
#5          5.0         3.6          1.4         0.2 Iris-setosa
#6          5.4         3.9          1.7         0.4 Iris-setosa
#8          5.0         3.4          1.5         0.2 Iris-setosa
```
Much like pandas `df.head()` in Python


### Y Class Levels
```R
levels(dataset$Species)
#[1] "Iris-setosa"     "Iris-versicolor" "Iris-virginica" 
```
Lists all unique class levels within the Species attribute of dataset


### Class Distribution
```R
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)
#                freq percentage
#Iris-setosa       40   33.33333
#Iris-versicolor   40   33.33333
#Iris-virginica    40   33.33333
```
Lists frequency and percentage of each individual class level within Species attribute

### Statistical Summary of Dataset
```R
summary(dataset)
#  Sepal.Length    Sepal.Width     Petal.Length    Petal.Width               Species  
# Min.   :4.300   Min.   :2.200   Min.   :1.100   Min.   :0.100   Iris-setosa    :40  
# 1st Qu.:5.175   1st Qu.:2.800   1st Qu.:1.575   1st Qu.:0.300   Iris-versicolor:40  
# Median :5.800   Median :3.000   Median :4.400   Median :1.300   Iris-virginica :40  
# Mean   :5.888   Mean   :3.091   Mean   :3.786   Mean   :1.222                       
# 3rd Qu.:6.425   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.825                       
# Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500         
```
Much like pandas `pd.describe()` in Python

### Separate training data (x) from labels (y)
```R
x <- dataset[,1:4] 
y <- dataset[,5]
```
The y label to predict for this model is Species, which is located at column 5 in the data frame. All other attribute columns are included as features in x

### Feature Boxplots
```R
par(mfrow=c(1,4))
    for(i in 1:4) {
      boxplot(x[,i], main=names(iris)[i])
    }
```

![Boxplots](https://github.com/trevorwitter/Iris-classification-R/blob/master/attribute_box_plots.jpg)

### Y Label Bar Plot
```R
plot(y)
```
![Bar Plots](https://github.com/trevorwitter/Iris-classification-R/blob/master/class_bar_plot.jpg)
Not particulary interesting looking in this case. This will be useful to identify [class imbalance](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/) in future projects. 


### Scatterplot Matrix
```R
featurePlot(x=x, y=y, plot="ellipse")
```
![Scatterplot Matrix](https://github.com/trevorwitter/Iris-classification-R/blob/master/scatter_plot_matrix.jpg)
`featurePlot()` with `plot="ellipse"` provides a quick and easy data-concentration ellipses in the off-diagonal panels

### Boxplots for each feature
```R
featurePlot(x=x, y=y, plot="box")
```
![Boxplots](https://github.com/trevorwitter/Iris-classification-R/blob/master/box_whisker_plot.jpg)
`featurePlot()` with `plot="box"` provides box plot with whiskers for each attribute grouped by y class


### Density Plots
```R
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)
```
![Density Plots](https://github.com/trevorwitter/Iris-classification-R/blob/master/density_plots.jpg)
`featurePlot()` with `plot="density"` provides density plots for each attribute grouped by y class

## Machine Learning Algorithm Evaluation
Algorithms assessed using 10-fold crossvalidation. Metric for assessment in this example is accuracy. This can be modified here for future projects
```R
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"
```

### Linear Discriminant Analyis
```R
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)
```

### Classification and Regression Trees
```R
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)
```

### k-Nearest Neighbors
```R
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)
```

### Support Vector Machines
```R
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)
```

### Random Forest
```R
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)
```

### Summarize Model Accuracies
```R
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)
#Call:
#summary.resamples(object = results)

#Models: lda, cart, knn, svm, rf 
#Number of resamples: 10 

#Accuracy 
#          Min.   1st Qu.    Median      Mean 3rd Qu. Max. NA's
#lda  0.9166667 0.9166667 1.0000000 0.9666667       1    1    0
#cart 0.8333333 0.9166667 1.0000000 0.9583333       1    1    0
#knn  0.8333333 0.9375000 1.0000000 0.9666667       1    1    0
#svm  0.8333333 1.0000000 1.0000000 0.9750000       1    1    0
#rf   0.8333333 0.9166667 0.9583333 0.9500000       1    1    0

#Kappa 
#      Min. 1st Qu. Median   Mean 3rd Qu. Max. NA's
#lda  0.875 0.87500 1.0000 0.9500       1    1    0
#cart 0.750 0.87500 1.0000 0.9375       1    1    0
#knn  0.750 0.90625 1.0000 0.9500       1    1    0
#svm  0.750 1.00000 1.0000 0.9625       1    1    0
#rf   0.750 0.87500 0.9375 0.9250       1    1    0
```

### Best Model Summary
```R
print(fit.lda)

#Linear Discriminant Analysis 

#120 samples
#  4 predictor
#  3 classes: 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica' 

#No pre-processing
#Resampling: Cross-Validated (10 fold) 
#Summary of sample sizes: 108, 108, 108, 108, 108, 108, ... 
#Resampling results:

#  Accuracy   Kappa
#  0.9666667  0.95 
```
96.6% accuracy on training data.. Not bad! Do keep in mind that this score is based on a small amount of predictions; the dataset is relatively small and when using 10 fold validation, evaluation is based on predictions made on 1/10th of the data.

### Evaluate Best Fit Model on Test Data
```R
predictions <- predict(fit.lda, test)
confusionMatrix(predictions, test$Species)

#Confusion Matrix and Statistics

#                 Reference
#Prediction        Iris-setosa Iris-versicolor Iris-virginica
#  Iris-setosa              10               0              0
#  Iris-versicolor           0              10              1
#  Iris-virginica            0               0              9

#Overall Statistics
                                          
#               Accuracy : 0.9667          
#                 95% CI : (0.8278, 0.9992)
#    No Information Rate : 0.3333          
#    P-Value [Acc > NIR] : 2.963e-13       
                                          
#                  Kappa : 0.95            
# Mcnemar's Test P-Value : NA              

#Statistics by Class:

#                     Class: Iris-setosa Class: Iris-versicolor Class: Iris-virginica
#Sensitivity                      1.0000                 1.0000                0.9000
#Specificity                      1.0000                 0.9500                1.0000
#Pos Pred Value                   1.0000                 0.9091                1.0000
#Neg Pred Value                   1.0000                 1.0000                0.9524
#Prevalence                       0.3333                 0.3333                0.3333
#Detection Rate                   0.3333                 0.3333                0.3000
#Detection Prevalence             0.3333                 0.3667                0.3000
#Balanced Accuracy                1.0000                 0.9750                0.9500
```

# Next Steps
The next update will focus on hyperparameter tuning for best fit model 

# Iris Classification in R

Basic boilerplate code example for supervised learning classification tasks. Dataset obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Iris)

#### Split data into train/test sets using createDataPartition()
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
sapply() used to map function to each attribute; class function returns data class type for a given attribute


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
Much like pandas' df.head() in Python


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
Much like pandas pd.describe() in Python

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
![Boxplots](https://render.githubusercontent.com/view/pdf?commit=5b9f8f6f976f6d443cb48313203776cea36bf531&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f747265766f727769747465722f497269732d636c617373696669636174696f6e2d522f356239663866366639373666366434343363623438333133323033373736636561333662663533312f6174747269627574655f626f785f706c6f74732e706466&nwo=trevorwitter%2FIris-classification-R&path=attribute_box_plots.pdf&repository_id=119458781&repository_type=Repository#7b3b09bb-b90a-4b88-bcd9-916a846e32dc)

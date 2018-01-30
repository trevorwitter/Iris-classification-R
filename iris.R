library("caret")

#load data from CSV file
data <- read.csv("iris.csv", header=FALSE) 
#Set data frame column names 
colnames(data) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")

# Split data into train/test sets using createDataPartition()
data_split <- createDataPartition(data$Species, p = 0.8, list = FALSE)

test <- data[-data_split,] # Save 20% of data for test validation here
dataset <- data[data_split,] # 80% of data 

# Data summary

# Dataset dimensions
dim(dataset)

# List of types for attributes
sapply(dataset, class)

# df header, just like python df.head() method
head(dataset)

# List of y class levels
levels(dataset$Species)

# Class distribution, as frequency and percentage
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)

# Statistcal summary of dataset
summary(dataset)

# Split dataset into x and y, y being class labels
x <- dataset[,1:4] 
y <- dataset[,5]

# Boxplot for each attribute in single image
par(mfrow=c(1,4))
    for(i in 1:4) {
      boxplot(x[,i], main=names(iris)[i])
    }

# barplot showing frequency of each class
plot(y)

# Multivariate plots
#Scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")

# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")

# density plots by class value for each attribute
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

# ML Algorithm Evaluation
# Algorithms will be assessed using 10-fold crossvalidation, setup here
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# build models

# Linear Discriminant Analysis
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)

# Classification and Regression Trees
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)

# k-Nearest Neighbors
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)

# Support Vector Machines
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)

#Random Forest
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)

# Summarize model accuracies
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

dotplot(results)

# Best Model Summary
print(fit.lda)

# Evaluate LDA model on test data
predictions <- predict(fit.lda, test)
confusionMatrix(predictions, test$Species)

# To add: hyperparameter tuning to improve model best fit model score 





# Iris Classification in R

Basic boilerplate code example for classification tasks. 

#### Split data into train/test sets using createDataPartition()
```R
data_split <- createDataPartition(data$Species, p = 0.8, list = FALSE)

test <- data[-data_split,] # Save 20% of data for test validation here
dataset <- data[data_split,] # 80% of data 
```

### Data Summary
#### Dataset Dimensions
```R
dim(dataset)
#[1] 120   5
```

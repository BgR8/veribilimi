# Importing dataset
#setwd("C:\\Users\\toshiba\\Desktop\\Machine Learning A-Z\\Part 1 - Data Preprocessing")

dataset = read.csv('Purchase_Prediction_Data.csv')
# dataset = dataset[,2:3]

# Splitting the dataset into the training and test
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])












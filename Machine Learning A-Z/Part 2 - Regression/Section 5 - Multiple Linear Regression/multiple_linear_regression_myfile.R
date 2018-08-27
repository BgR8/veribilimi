# Multiple Linear Regression 
setwd('C:\\Users\\toshiba\\Desktop\\Machine Learning A-Z\\Part 2 - Regression\\Section 5 - Multiple Linear Regression')

# Importing the dataset
dataset = read.csv('Sirketler_Kar_Bilgileri.csv')

# Encoding categorical data
dataset$Sehir = factor(dataset$Sehir,
                       levels = c('Ankara','Istanbul','Kocaeli'),
                       labels = c(1,2,3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Kar, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Multiple Linear Regression to the Training Set
regressor = lm(formula = Kar ~ ArgeHarcamasi + YonetimGiderleri + PazarlamaHarcamasi + Sehir, data = training_set)
regressor = lm(formula = Kar ~ ., data = training_set)

summary(regressor)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Building the optimal model using Backward Elimination
regressor = lm(formula = Kar ~ ArgeHarcamasi + YonetimGiderleri + PazarlamaHarcamasi,
               data = dataset)
regressor = lm(formula = Kar ~ ArgeHarcamasi + PazarlamaHarcamasi,
               data = dataset)
regressor = lm(formula = Kar ~ ArgeHarcamasi, data = dataset)


y_pred
# Random Forest Regreesyon

# Setting working directory
setwd('C:\\Users\\toshiba\\Desktop\\Machine Learning A-Z\\Part 2 - Regression\\Section 9 - Random Forest Regression')
# Importing the dataset
dataset = read.csv('PozisyonSeviyeMaas.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the Regression Model to the dataset
install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1], 
                         y = dataset$Maas,
                         ntree = 100)

# Predicting a new result
y_pred = predict(regressor, data.frame(Seviye = 6.5))

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Seviye), max(dataset$Seviye), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Seviye, y = dataset$Maas),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Seviye = x_grid))),
            colour = 'blue') +
  ggtitle('Random Forest Regresyon') +
  xlab('Seviye') +
  ylab('Maas')
# Regression Template

# Çalışma dizini ayarla
setwd('C:\\Users\\toshiba\\Desktop\\Machine Learning A-Z\\Part 2 - Regression\\Section 8 - Decision Tree Regression')

# Importing the dataset
dataset = read.csv('PozisyonSeviyeMaas.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the Decision Tree Regression to the dataset
#install.packages('rpart')
library(rpart)
regressor = rpart(formula = Maas ~ ., data = dataset,
                  control = rpart.control(minsplit = 1))

# Predicting a new result
y_pred = predict(regressor, data.frame(Seviye = 6.5))

# Visualising the Regression Model results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Seviye, y = dataset$Maas),
             colour = 'red') +
  geom_line(aes(x = dataset$Seviye, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Karar Ağacı Regresyonu') +
  xlab('Pozisyon Seviye') +
  ylab('Maas')

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Seviye), max(dataset$Seviye), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Seviye, y = dataset$Maas),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Seviye = x_grid))),
            colour = 'blue') +
  ggtitle('Karar Ağacı Regresyonu') +
  xlab('Pozisyon Seviye') +
  ylab('Maas')
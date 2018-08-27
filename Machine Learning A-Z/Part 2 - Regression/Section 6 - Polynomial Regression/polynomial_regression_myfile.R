# Data Preprocessing Template

# Setting working directory
setwd("C:\\Users\\toshiba\\Desktop\\Machine Learning A-Z\\Part 2 - Regression\\Section 6 - Polynomial Regression")

# Importing the dataset
dataset = read.csv('PozisyonSeviyeMaas.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Linear Regression to the dataset
lin_reg = lm(formula = Maas ~ ., data = dataset)
summary(lin_reg)

# Fitting Polinom Model
dataset$Seviye2 = dataset$Seviye^2
dataset$Seviye3 = dataset$Seviye^3

poly_reg = lm(formula = Maas ~ ., data = dataset)
summary(poly_reg)

# Lineer model grafiği
library(ggplot2)
ggplot()+
  geom_point(aes(x = dataset$Seviye, y = dataset$Maas),
             colour = 'red')+
  geom_line(aes(x = dataset$Seviye, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue')+
  ggtitle('Lineer Regresyon')+
  xlab('Seviye')+
  ylab('Maas')

# Polinom model grafiği
ggplot()+
  geom_point(aes(x = dataset$Seviye, y = dataset$Maas),
             colour = 'red')+
  geom_line(aes(x = dataset$Seviye, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue')+
  ggtitle('Polinom Regresyon')+
  xlab('Seviye')+
  ylab('Maas')

# Lineer model ile 6.5'in tahmini
y_pred = predict(lin_reg, data.frame(Level = 6.5))

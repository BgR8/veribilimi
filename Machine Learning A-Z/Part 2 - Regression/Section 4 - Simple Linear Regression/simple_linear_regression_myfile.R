setwd("C:\\Users\\toshiba\\Desktop\\Machine Learning A-Z\\Part 2 - Regression\\Section 4 - Simple Linear Regression")

# Simple Linear Regression
dataset = read.csv('Kidem_ve_Maas_VeriSeti.csv')
# dataset = dataset[,2:3]

# Splitting the dataset into the training and test
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Maas, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])

# Fitting Simple Linear Regression to the Training Set
regressor = lm(formula = Maas ~ Kidem, data = training_set)
summary(regressor)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)
View(y_pred)

# Visualising the Training set results
library(ggplot2)
ggplot()+
  geom_point(aes(x = training_set$Kidem, y = training_set$Maas),
             color='red')+
  geom_line(aes(x = training_set$Kidem, y = predict(regressor, newdata = training_set)),
            color = 'blue')+
  ggtitle('Kıdeme Göre Maaş Tahmini Regresyon Model Grafiği')+
  xlab('Kıdem(Yıl)')+
  ylab('Maaş(Yıllık - TL)')

# Visualising the Test set results
ggplot()+
  geom_point(aes(x = test_set$Kidem, y = test_set$Maas),
             color='red')+
  geom_line(aes(x = training_set$Kidem, y = predict(regressor, newdata = training_set)),
            color = 'blue')+
  ggtitle('Kıdeme Göre Maaş Tahmini Regresyon Model Grafiği (Test Seti)')+
  xlab('Kıdem(Yıl)')+
  ylab('Maaş(Yıllık - TL)')

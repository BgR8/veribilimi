# Apriori

# Data Preprocessing
# Setting working directory
setwd('C:\\Users\\toshiba\\Desktop\\Machine Learning A-Z\\Part 5 - Association Rule Learning\\Section 28 - Apriori\\Apriori')

# install.packages('arules')
library(arules)
dataset = read.csv('Birliktelik_Kurali_Market_Satis_Kayitlari.csv', header = FALSE, encoding = 'UTF-8')
dataset = read.transactions('Birliktelik_Kurali_Market_Satis_Kayitlari.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])
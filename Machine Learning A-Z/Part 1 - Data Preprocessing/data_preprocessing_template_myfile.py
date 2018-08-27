import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset

dataset = pd.read_csv('Purchase_Prediction_Data.csv')

# Aşağıdaki kodda köşeli parantez içindeki ilk : satırları, 
# ikinci : sütunları temsil eder. -1 tüm sütunları en son hariç al demektir.
#  Verimizde en son sütun Purchased idi yani hedef değişkenimiz idi.
X = dataset.iloc[:,:-1].values

# Şimdi bağımlı değişkenimizi y'ye atayalım
y = dataset.iloc[:,3].values


# Splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
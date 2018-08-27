# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 21:14:37 2018

@author: toshiba
"""
#Part 1 Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# 0,1,2 indeksli nitelikler hedef değişkene etkisi olmadığı için nitelikler matrisine almıyoruz.
# 3 ile 12 (dahil) arasındakileri nitelikler matrisine alıyruz
X = dataset.iloc[:, 3:13].values

# Hedef değişken 13. indekste onu da y ye atıyoruz
y = dataset.iloc[:, 13].values

# Veriyi train ve test olarak bölmeden önce kategorik nitelikleri kodlamalıyız
# İki kategorik nitelik var Geagraphy (ülkeler) ve Cinsiyer ()
# Kütüphaneleri indirme
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# İlk kategorik niteliğimiz Geography (X'de sütun indisi 1) dönüşüm işlemi
labelencoder_X_1 = LabelEncoder()
# Geography sütununa nümerik değerlere çevrilmiş indeks değerlerini atıyoruz
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# Kontrol için Ipython Console a X yazmamız yeterli.

# İkinci kategorik niteliğimiz Gender (X'de sütun indisi 2) dönüşüm işlemi
labelencoder_X_2 = LabelEncoder()
# Gender sütununa nümerik değerlere çevrilmiş indeks değerlerini atıyoruz
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Kontrol için Ipython Console a X yazmamız yeterli.

# Geography için gölge değişken oluşturuyoruz
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Artık 0,1,2 indisleri Geagraphy gölge değişkenlerine ait.

# Gölge değişken tuzağından (dummy variable trap) kaçınmak için bir tanesini düşürüyoruz.
# Bunun için X'i seçerken 0 indeks değerli sütunu hariç bırakarak kendisinden seçeceğiz.
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
# artık cross_validation yerine model_selection diyoruz 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# Derin öğrenmede mutlaka normalizasyon yapılmalıdır.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part 2  ANN Yapalım

# Keras kütüphanesini yüklemek
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(kernel_initializer = 'uniform', input_dim = 11, units = 6,  activation = 'relu'))
    classifier.add(Dense(kernel_initializer = 'uniform', units = 4,  activation = 'relu'))
    classifier.add(Dense(kernel_initializer = 'uniform', units = 1,  activation = 'sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# Fonksiyondan sonra fonksiyon dışında da bir nesne yaratmalıyız. Çünkü o fonksiyon içinde kaldı.
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)

# şimdi cv sonuçlarını alacak bir değişken yaratalım
# estimator az önce KerasClassifier sınıfından yarattığımız classifier, X_train ve y_train belli zaten
# cv=10 crossvalidation rakamı, n_jobs kaç tane işlemci çekirdeği kullansın -1 müsait olan hepsini kullan demek

accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv=10, n_jobs=-1)

mean = accuracies.mean()
variance = accuracies.std()













































# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

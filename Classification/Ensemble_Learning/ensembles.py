# Genel kütüphaneleri dahil et
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Çalışma dizni ayarla
os.chdir('C:\\Users\\toshiba\\SkyDrive\\veribilimi.co\\codes\\Classification\\Ensemble_Learning')

# Pandas ile veri setini dosyadan oku
data = pd.read_csv('C:\\Users\\toshiba\\SkyDrive\\veribilimi.co\\Datasets\\SosyalMedyaReklamKampanyasi.csv')

# Bağımsız nitelikleri X ile ifade edilen nitelikler matrisinde topla
X = data.iloc[:, [2, 3]].values

# Bağımlı değişkeni y niteliğinde topla
y = data.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
# veriyi eğitim ve test olarak ayırmak için scikit-learn kütüphanesi cross_validation modülü içindeki train_test_split fonksiyonunu dahil edelim
from sklearn.cross_validation import train_test_split

# Yukarıda veri setimizi X ve y olarak niteliklerden (sütun bazlı bölme) ayırmıştık. Şimdi hem X hem de y'i
# satır bazlı ikiye bölelim. Satırların % 80'i eğitim, % 20'si test için ayıralım. test_size = 0.20 buna işaret ediyor.
# Aşağıdaki kodlarla X'ten rastgele seçilecek %80 satırı X_train'e; kalan %20 ise X_test'e atanır.
# Aynı şekilde y'den rastgele seçilecek %80 satırı y_train'e; kalan %20 ise y_test'e atanır.
# Yalnız bu seçim işlemi hepsinde aynı satırlar seçilecek şekilde yapılır. Örneğin X_train'nin 5. satırı seilmişse diğerlerinde de 5. satır seçilir.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Sınıflandırma için karar ağacı kullanılacak. Karar Ağacı modeli için scikit-learn kütüphanesi tree modilinden DecisionTreeClassifier sınıfını dahil edelim.
from sklearn.tree import DecisionTreeClassifier
decisionTreeObject = DecisionTreeClassifier()
decisionTreeObject.fit(X_train,y_train)

# Test sonuçlarını hesaplayıp yazdıralım.
dt_test_sonuc = decisionTreeObject.score(X_test, y_test)
print("Karar Ağacı Doğruluk (test_seti): ",round(dt_test_sonuc,2))

# Aşırı öğrenme ezberleme var mı diye modeli eğitim setiyle test edelim
dt_egitim_sonuc = decisionTreeObject.score(X_train,y_train)
print("Karar Ağacı Doğruluk (eğitim_seti): ",round(dt_egitim_sonuc,2))


# Şimdi de herşey aynı kalsın sadece sınıflandırıcıyı random forest yapalım
# Scikit-learn kütüphanesi, ensemble modülünden RandomForestClassifier sınıfını dahil edelim
from sklearn.ensemble import RandomForestClassifier

randomForestObject = RandomForestClassifier(n_estimators=10)
randomForestObject.fit(X_train, y_train)

# Test sonuçlarını hesaplayıp yazdıralım.
df_test_sonuc = randomForestObject.score(X_test, y_test)
print("Random Forest Doğruluk (test_seti): ",round(df_test_sonuc,2))

# Aşırı öğrenme ezberleme var mı diye modeli eğitim setiyle test edelim
df_egitim_sonuc = randomForestObject.score(X_train,y_train)
print("Random Forest Doğruluk (test_seti): ", round(df_egitim_sonuc,2))


# Şimdi Ensemble metotları uygulayalım: Bagging
# Scikit-learn kütüphanesi, ensemble modülünden BaggingClassifier sınıfını dahil edelim
from sklearn.ensemble import BaggingClassifier

#Herşey aynı sadece sınıflandırıcıyı BaggingClassifier sınıfından yarattığımız model ile eğitip test edelim.
baggingObject = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=1.0, n_estimators=20)
baggingObject.fit(X_train, y_train)


# Şimdi bagging modelinin test sonuçlarına bakalım
baggingObject_sonuc = baggingObject.score(X_test, y_test)
print("Bagging Doğruluk (test_seti): ", round(baggingObject_sonuc,2))

# Şimdi ikinci Ensemble metodu uygulayalım: Boosting - Ada Boost
# Scikit-learn kütüphanesi, ensemble modülünden AdaBoostClassifier sınıfını dahil edelim
from sklearn.ensemble import AdaBoostClassifier

adaBoostingObject = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators= 10, learning_rate=1)
adaBoostingObject.fit(X_train, y_train)

# Şimdi ada boosting modelinin test sonuçlarına bakalım
adaBoostingObject_sonuc = adaBoostingObject.score(X_test, y_test)
print("Ada Boosting Doğruluk (test_seti): ", round(adaBoostingObject_sonuc,2))

# Aşırı öğrenme var mı kontrol edelim
adaBoostingObject_train = adaBoostingObject.score(X_train, y_train)
print("Ada Boosting Doğruluk (eğitim_seti): ", round(adaBoostingObject_train,2))






















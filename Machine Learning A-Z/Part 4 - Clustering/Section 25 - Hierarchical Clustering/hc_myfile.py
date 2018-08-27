# Hiyerarşik Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set working directory
os.chdir("C:\\Users\\toshiba\\Desktop\\Machine Learning A-Z\\Part 4 - Clustering\\Section 25 - Hierarchical Clustering")
# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using dendogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Müşteriler')
plt.ylabel('Öklid Mesafesi')
plt.show()

# Fitting hierarchical clustering to the mail dataset
from sklearn.cluster import AgglomerativeClustering
hiyerarsikKumeleyici =AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
kume_elemanlari = hiyerarsikKumeleyici.fit_predict(X)

# Visualising the clusters
plt.scatter(X[kume_elemanlari == 0, 0], X[kume_elemanlari == 0, 1], s = 100, c = 'red', label = 'Küme 1')
plt.scatter(X[kume_elemanlari == 1, 0], X[kume_elemanlari == 1, 1], s = 100, c = 'blue', label = 'Küme 2')
plt.scatter(X[kume_elemanlari == 2, 0], X[kume_elemanlari == 2, 1], s = 100, c = 'green', label = 'Küme 3')
plt.scatter(X[kume_elemanlari == 3, 0], X[kume_elemanlari == 3, 1], s = 100, c = 'cyan', label = 'Küme 4')
plt.scatter(X[kume_elemanlari == 4, 0], X[kume_elemanlari == 4, 1], s = 100, c = 'magenta', label = 'Küme 5')
plt.title('Hiyerarşik Kümeleme ile Müşteri Kümeleri')
plt.xlabel('Yıllık Gelir')
plt.ylabel('Harcama Skoru (1-100)')
plt.legend()
plt.show()
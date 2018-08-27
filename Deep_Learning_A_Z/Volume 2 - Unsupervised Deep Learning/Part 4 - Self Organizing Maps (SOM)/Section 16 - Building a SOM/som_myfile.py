# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 20:54:53 2018

@author: toshiba
"""
# Self-organizing Map
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))

X = sc.fit_transform(X)

# Training the SOM
# Çalışma dizinindeki minisom.py dosyasını import ederek içindeki Sınıfı kullanacağız
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)


# Visualization the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 07:05:43 2017

@author: toshiba
"""

liste1 = [3,7,2,0,1] # Orijinal liste oluşturulur
liste2 = [] # hesaplanan değerlerin tutulacağı ikinci boş liste oluşturulur

# Orijinal listenin her bir elemanına for döngüsü ile ulaşılır
# ve her eleman -2'nin üssü olarak hesaplanarak liste2'ye eklenir
# Liste elemanları hesaplama anında integer'a çevrilir
for i in liste1:
    liste2.append((-2)**int(i))
    
liste2.sort()
print(liste2)
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:38:00 2022

@author: rasha
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('pat_data.csv')
data

plt.scatter(data['spo2'], data['hr'])
plt.xlabel('spo2')
plt.ylabel('hr')
plt.show()

x=data.copy()

from sklearn import preprocessing
x_scaled=preprocessing.scale(x)
x_scaled

wcss = [] 
for i in range(1, 30): 
    kmeans = KMeans(i)
    kmeans.fit(x_scaled) 
    wcss.append(kmeans.inertia_)
wcss

plt.plot(range(1, 30), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()
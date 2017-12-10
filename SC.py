# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:00:03 2017

@author: kaish
"""

import numpy as np
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt 
from pyspc import hotelling, Tsquare_single, rules
from pyspc import spc
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from math import sqrt




results = []
with open('control.txt') as inputfile:
    for line in inputfile:
        results.append(line.strip().split(' '))
        
datas=np.array(results)
datas=datas.astype(np.float)

df = pd.DataFrame(data=datas, columns=['x1','x2','x3','x4','x5'])
axes = pd.plotting.scatter_matrix(df, alpha=0.9, figsize=(7, 7), c='blue',s=80)
plt.tight_layout()
plt.show()
plt.savefig('scatter_matrix.png')


fig=plt.figure(figsize=(7,7))
d = spc(datas) + Tsquare_single() 
print(d)


d2=PCA(n_components=5)

datas_standard=(datas - np.mean(datas, axis=0)) / np.std(datas, axis=0)

transformed=d2.fit_transform(datas_standard)

Lambda=d2.explained_variance_

UCL=3*sqrt(Lambda[0])
LCL=-3*sqrt(Lambda[0])


fig=plt.figure(figsize=(7,7))
plt.plot(transformed[:,0],marker='.',markersize='15',linestyle='--',c='blue')
plt.axhline(y=UCL, color='r', linestyle=':')
plt.axhline(y=LCL, color='r', linestyle=':')
plt.show


dist=euclidean_distances(datas_standard, datas_standard)

upper = []
lower = []
for j in range(0, len(dist)):
    for i in range(0, len(dist)):
        if j>i:
            lower.append(dist[j][i])
        elif j<i:
            upper.append(dist[j][i])
        else:
            pass
upperSum = sum(upper)
lowerSum = sum(lower)

sigma=(2/(20*(20-1)))*upperSum

d3=KernelPCA(n_components=5, kernel="rbf",gamma=1/(2*sigma**2))


transformed=d3.fit_transform(datas_standard)*np.sqrt(d3.lambdas_)

Lambda2=d3.lambdas_/20

UCL=3*sqrt(Lambda2[0])
LCL=-3*sqrt(Lambda2[0])


fig=plt.figure(figsize=(7,7))
plt.plot(transformed[:,0],marker='.',markersize='15',linestyle='--',c='blue')
plt.axhline(y=UCL, color='r', linestyle=':')
plt.axhline(y=LCL, color='r', linestyle=':')
plt.show


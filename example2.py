# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 10:04:27 2017

@author: kashen
"""

from sklearn.decomposition import PCA
from KPCA_self import kPCA
import matplotlib.pyplot as plt
import numpy as np
import data_example2 as data
from sklearn.decomposition import KernelPCA


plt.rc('font', family='serif')


plt.rc('text', usetex=False)

def plot(methods, X, Y, X0, Y0, line, rowspan):
    "Plots all results in the input list as a series of subplots"

    n_methods = len(methods)
    i = 0
    plt.hold(True)
    handles = []
    for denoised, name in methods:
        plt.subplot2grid((2, 2), (line, i), rowspan=rowspan)
        handle0, =plt.plot(X0,Y0,'.', color='r')
        handle1, = plt.plot(X, Y, '.', color="0.8")
        plt.title(name)
        handle2, = plt.plot(denoised[:,0], denoised[:,1], 'k.')
        i += 1
        handles.append(handle0)
        handles.append(handle1)
        handles.append(handle2)
    return handles

def pca_denoising(data):
    "Performs linear PCA denoising using sklearn"
    pca = PCA(n_components=1)
    low_dim_representation = pca.fit_transform(data)
    return pca.inverse_transform(low_dim_representation)

X, Y = data.get_curves(noise='normal', scale=0.2)
noisy_data = np.array([X, Y]).T

KPCA=KernelPCA(kernel="rbf",fit_inverse_transform=True)
X_new=KPCA.fit_transform(noisy_data)

fig=plt.figure(figsize=(10,10))



# To add a new method, simply add it to both methods list
# Curves

methods = [
    (kPCA(noisy_data).obtain_preimages(4, 0.5), 'Kernel PCA'),
   # ( KPCA.inverse_transform(X_new),'Kernel PCA'),
   # (fit_curve(noisy_data), 'Principal Curves'),
    (pca_denoising(noisy_data), 'Linear PCA')
]

Xo, Yo=data.get_curves(points=1000, radius=2, noise=None, original=True)
plot(methods, X, Y, Xo, Yo, 0, 1)


# Square
X, Y = data.get_square(noise='normal', scale=0.2)
noisy_data = np.array([X, Y]).T
KPCA2=KernelPCA(kernel="rbf",fit_inverse_transform=True)
X_new2=KPCA2.fit_transform(noisy_data)

methods = [
    (kPCA(noisy_data).obtain_preimages(4, 0.6), 'Kernel PCA'),
    #( KPCA2.inverse_transform(X_new2),'Kernel PCA'),
    #(fit_curve(noisy_data), 'Principal Curves'),
    #(fit_curve(noisy_data, circle=True), 'Principal Curves (from circle)'),
    (pca_denoising(noisy_data), 'Linear PCA')
]

Xo, Yo = data.get_square(points=1000, noise=None, original=True)
handles = plot(methods, X, Y, Xo, Yo, 1, 1)
plt.figlegend(handles[:3], ['Original data', 'Noised data,', 'Denoised data'],
    loc='upper left')
plt.show()

